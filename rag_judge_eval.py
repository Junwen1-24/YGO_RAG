"""
Quantitative RAG evaluation via a separate "Judge" LLM (professor-style grading).

Default judge model is configurable via YGO_JUDGE_MODEL or --judge-model.
The API model id is often a dated snapshot (e.g. gpt-5.4-2026-03-05), not the marketing name.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from query_ygo_judge import ensure_openai_api_key

JUDGE_SYSTEM_PROMPT = """System Role:

You are a Senior Yu-Gi-Oh! Head Judge and an AI Quality Assurance Specialist. Your task is to quantitatively evaluate the performance of a RAG (Retrieval-Augmented Generation) system designed to answer complex YGO ruling questions.

The Input Data:

I will provide you with:

User Question: The specific gameplay scenario.

Official Answer (Ground Truth): The "Up to date" Konami ruling.

Retrieved Context: The snippets the RAG found in the database.

RAG Generated Answer: The answer produced by the model (gpt-4o-mini).

Your Evaluation Metrics (Score each 0.0 to 1.0):

Context Precision (0.0 - 1.0): Did the RAG retrieve the correct card and the relevant rule section (e.g., Damage Step rules for a Damage Step question)?

Faithfulness (0.0 - 1.0): Is the generated answer derived only from the retrieved context? Deduct points if the model uses outside knowledge or hallucinates rules not present in the snippets.

Answer Correctness (0.0 - 1.0): Does the final ruling (Yes/No, Can/Cannot) match the Official Answer? (1.0 = Perfect match, 0.5 = Correct result but wrong reasoning, 0.0 = Wrong result).

PSCT Logic (0.0 - 1.0): Did the model correctly identify and use the PSCT (Problem-Solving Card Text) structure from the retrieved card text? In particular:
- If the model quotes clauses around ':' for conditions and ';' for costs WHEN those symbols actually appear in the retrieved [CARD] text, and maps them correctly to the scenario, score PSCT Logic higher.
- If those symbols do NOT appear in any retrieved [CARD] snippet and the model explicitly marks them as \"Absent in snippet\" (or equivalent wording) and does NOT invent extra conditions/costs, treat that as correct/neutral rather than an error.
- Deduct points only when the model misreads PSCT structure (e.g. treats an effect as a cost, or claims ':'/' ;' structure that is not present in the retrieved snippets), or when it contradicts the actual card text.

Output Format (STRICT JSON):

Return your evaluation ONLY as a JSON object with this structure:
{
  "metrics": {
    "context_precision": 0.0,
    "faithfulness": 0.0,
    "answer_correctness": 0.0,
    "psct_logic": 0.0
  },
  "total_score_weighted": 0.0,
  "judge_reasoning": "A one-sentence explanation of the score."
}

Rules:
- Output JSON only. No markdown fences, no commentary before or after the JSON.
- total_score_weighted must be the mean of the four metrics (equal weight)."""


def default_judge_model() -> str:
    # gpt-5.4-thinking may 404; dated snapshot is the API id for GPT-5.4 Thinking-class models.
    return (os.getenv("YGO_JUDGE_MODEL") or "gpt-5.4-2026-03-05").strip()


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _clamp01(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))


def _parse_judge_json(raw: str) -> dict[str, Any]:
    text = _strip_json_fence(raw)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Judge output is not a JSON object.")
    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Missing or invalid 'metrics' object.")

    keys = ("context_precision", "faithfulness", "answer_correctness", "psct_logic")
    out_metrics: dict[str, float] = {}
    for k in keys:
        out_metrics[k] = _clamp01(metrics.get(k))

    total_raw = data.get("total_score_weighted")
    if total_raw is None or total_raw == "":
        total = sum(out_metrics.values()) / 4.0
    else:
        total = _clamp01(total_raw)

    reasoning = data.get("judge_reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)

    return {
        "metrics": out_metrics,
        "total_score_weighted": float(total),
        "judge_reasoning": reasoning.strip(),
    }


def judge_rag_output(
    *,
    user_question: str,
    official_answer: str,
    retrieved_context: str,
    rag_generated_answer: str,
    judge_model: str | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """
    Call the Judge LLM and return a dict with metrics, total_score_weighted, judge_reasoning.

    On failure, returns {"error": str, ...} with empty metrics.
    """
    ensure_openai_api_key()
    model = (judge_model or default_judge_model()).strip()
    if not model:
        model = "gpt-5.4-2026-03-05"

    user_block = (
        f"User Question:\n{user_question}\n\n"
        f"Official Answer (Ground Truth):\n{official_answer}\n\n"
        f"Retrieved Context:\n{retrieved_context}\n\n"
        f"RAG Generated Answer:\n{rag_generated_answer}\n"
    )

    llm = ChatOpenAI(model=model, temperature=temperature)
    response = llm.invoke(
        [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=user_block),
        ]
    )
    raw = str(response.content).strip()
    try:
        parsed = _parse_judge_json(raw)
        parsed["judge_raw"] = raw
        parsed["judge_model"] = model
        parsed["judge_error"] = ""
        return parsed
    except (json.JSONDecodeError, ValueError) as exc:
        err = f"{type(exc).__name__}: {exc}"
        return {
            "judge_error": err,
            "metrics": {
                "context_precision": 0.0,
                "faithfulness": 0.0,
                "answer_correctness": 0.0,
                "psct_logic": 0.0,
            },
            "total_score_weighted": 0.0,
            "judge_reasoning": "",
            "judge_raw": raw,
            "judge_model": model,
        }
