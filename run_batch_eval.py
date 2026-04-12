#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

try:
    from openai import APITimeoutError
except Exception:  # pragma: no cover
    APITimeoutError = TimeoutError  # type: ignore[misc,assignment]

from query_ygo_judge import ask_ygo_judge
from rag_judge_eval import default_judge_model, judge_rag_output


def _serialize_contexts(result: dict[str, Any]) -> str:
    cards = result.get("cards", []) or []
    rules = result.get("rules", []) or []
    qa_docs = result.get("qa", []) or []
    contexts: list[dict[str, Any]] = []

    for doc in cards:
        contexts.append(
            {
                "type": "card",
                "text": doc.page_content,
                "metadata": dict(doc.metadata or {}),
            }
        )
    for doc in rules:
        contexts.append(
            {
                "type": "rule",
                "text": doc.page_content,
                "metadata": dict(doc.metadata or {}),
            }
        )
    for doc in qa_docs:
        contexts.append(
            {
                "type": "qa",
                "text": doc.page_content,
                "metadata": dict(doc.metadata or {}),
            }
        )

    # Keep JSON in one CSV cell with UTF-8 text preserved.
    return json.dumps(contexts, ensure_ascii=False)


def ask_with_retry(
    query: str,
    tags: str | None,
    retries: int,
    retry_backoff_sec: float,
    rules_index_dir: str,
    cards_index_dir: str,
    qa_index_dir: str,
    embedding_model: str,
    chat_model: str,
    temperature: float,
    use_rag: bool,
    rules_k: int,
    cards_k: int,
    qa_k: int,
    fetch_k: int,
    keyword_k: int,
    coverage_check: bool,
    second_pass_on_gap: bool,
) -> tuple[str, str, float, str]:
    last_error = ""
    for attempt in range(1, retries + 1):
        started = time.perf_counter()
        try:
            result = ask_ygo_judge(
                question=query,
                tags=tags,
                rules_index_dir=rules_index_dir,
                cards_index_dir=cards_index_dir,
                qa_index_dir=qa_index_dir,
                embedding_model=embedding_model,
                chat_model=chat_model,
                temperature=temperature,
                use_rag=use_rag,
                rules_k=rules_k,
                cards_k=cards_k,
                qa_k=qa_k,
                fetch_k=fetch_k,
                keyword_k=keyword_k,
                coverage_check=coverage_check,
                second_pass_on_gap=second_pass_on_gap,
            )
            elapsed = time.perf_counter() - started
            generated_answer = str(result.get("answer", "")).strip()
            retrieved_contexts = _serialize_contexts(result)
            return generated_answer, retrieved_contexts, elapsed, ""
        except APITimeoutError as exc:
            last_error = f"timeout: {exc}"
        except TimeoutError as exc:
            last_error = f"timeout: {exc}"
        except Exception as exc:  # pragma: no cover
            # Retry generic transient API errors too during batch processing.
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < retries:
            sleep_s = retry_backoff_sec * (2 ** (attempt - 1))
            time.sleep(sleep_s)

    return "", "[]", 0.0, last_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch eval on data/ygo_eval_dataset.csv (RAG or ablation).")
    parser.add_argument("--input", default="data/ygo_eval_dataset.csv", help="Ground-truth CSV path.")
    parser.add_argument("--output", default="eval_results/test_results_raw.csv", help="Output CSV path.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per question on timeout/API errors.")
    parser.add_argument(
        "--retry-backoff-sec",
        type=float,
        default=2.0,
        help="Base backoff seconds (exponential) between retries.",
    )
    parser.add_argument("--rules-index-dir", default="faiss_rules_index")
    parser.add_argument("--cards-index-dir", default="faiss_cards_index")
    parser.add_argument("--qa-index-dir", default="faiss_qa_index")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--chat-model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--rules-k", type=int, default=5, help="Final number of retrieved rule chunks.")
    parser.add_argument("--cards-k", type=int, default=5, help="Final number of retrieved card chunks.")
    parser.add_argument("--qa-k", type=int, default=2, help="Final number of retrieved Q&A snippets.")
    parser.add_argument(
        "--fetch-k",
        type=int,
        default=10,
        help="Initial vector/MMR candidates before dedupe and rerank.",
    )
    parser.add_argument(
        "--keyword-k",
        type=int,
        default=12,
        help="Keyword-retrieval candidates (hybrid search).",
    )
    parser.add_argument(
        "--no-coverage-check",
        action="store_true",
        help="Skip LLM coverage audit before answering (no second retrieval pass).",
    )
    parser.add_argument(
        "--no-second-pass",
        action="store_true",
        help="Keep coverage check but never run a second retrieval pass.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional number of rows to run from the top of the CSV (0 = all).",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Ablation: call the LLM only (no FAISS retrieval). retrieved_contexts will be [].",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="After each answer, call the Judge LLM (GPT-style grader) for quantitative metrics.",
    )
    parser.add_argument(
        "--judge-model",
        default="",
        help="Judge chat model (default: YGO_JUDGE_MODEL env or gpt-5.4-2026-03-05).",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.0,
        help="Temperature for the Judge LLM (default: 0).",
    )
    args = parser.parse_args()
    judge_model = (args.judge_model or "").strip() or default_judge_model()
    coverage_check = not args.no_coverage_check
    second_pass_on_gap = not args.no_second_pass

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    rows = list(csv.DictReader(input_path.open("r", encoding="utf-8", newline="")))
    if not rows:
        raise ValueError(f"Input CSV has no data rows: {input_path}")
    if "Question" not in rows[0]:
        raise KeyError("Input CSV must contain a 'Question' column.")
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ID",
        "Tags",
        "Question",
        "Official_Answer",
        "generated_answer",
        "retrieved_contexts",
        "processing_time",
        "status",
        "error",
    ]
    judge_fields = [
        "context_precision",
        "faithfulness",
        "answer_correctness",
        "psct_logic",
        "total_score_weighted",
        "judge_reasoning",
        "judge_error",
        "judge_model",
    ]
    if args.judge:
        fieldnames = fieldnames + judge_fields

    success = 0
    fail = 0
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        total = len(rows)
        for idx, row in enumerate(rows, start=1):
            question = (row.get("Question") or "").strip()
            if not question:
                fail += 1
                empty_row: dict[str, str] = {
                    "ID": row.get("ID", ""),
                    "Tags": row.get("Tags", ""),
                    "Question": row.get("Question", ""),
                    "Official_Answer": row.get("Official_Answer", ""),
                    "generated_answer": "",
                    "retrieved_contexts": "[]",
                    "processing_time": "0.0000",
                    "status": "failed",
                    "error": "empty question",
                }
                if args.judge:
                    for k in judge_fields:
                        empty_row[k] = ""
                writer.writerow(empty_row)
                continue

            generated_answer, retrieved_contexts, elapsed, error = ask_with_retry(
                query=question,
                tags=(row.get("Tags") or "").strip() or None,
                retries=args.retries,
                retry_backoff_sec=args.retry_backoff_sec,
                rules_index_dir=args.rules_index_dir,
                cards_index_dir=args.cards_index_dir,
                qa_index_dir=args.qa_index_dir,
                embedding_model=args.embedding_model,
                chat_model=args.chat_model,
                temperature=args.temperature,
                use_rag=not args.no_rag,
                rules_k=args.rules_k,
                cards_k=args.cards_k,
                qa_k=args.qa_k,
                fetch_k=args.fetch_k,
                keyword_k=args.keyword_k,
                coverage_check=coverage_check,
                second_pass_on_gap=second_pass_on_gap,
            )
            ok = bool(generated_answer)
            if ok:
                success += 1
            else:
                fail += 1

            row_out: dict[str, str] = {
                "ID": row.get("ID", ""),
                "Tags": row.get("Tags", ""),
                "Question": row.get("Question", ""),
                "Official_Answer": row.get("Official_Answer", ""),
                "generated_answer": generated_answer,
                "retrieved_contexts": retrieved_contexts,
                "processing_time": f"{elapsed:.4f}",
                "status": "ok" if ok else "failed",
                "error": error,
            }

            if args.judge:
                try:
                    j = judge_rag_output(
                        user_question=question,
                        official_answer=(row.get("Official_Answer") or "").strip(),
                        retrieved_context=retrieved_contexts,
                        rag_generated_answer=generated_answer,
                        judge_model=judge_model,
                        temperature=args.judge_temperature,
                    )
                    m = j.get("metrics") or {}
                    row_out["context_precision"] = f"{float(m.get('context_precision', 0.0)):.4f}"
                    row_out["faithfulness"] = f"{float(m.get('faithfulness', 0.0)):.4f}"
                    row_out["answer_correctness"] = f"{float(m.get('answer_correctness', 0.0)):.4f}"
                    row_out["psct_logic"] = f"{float(m.get('psct_logic', 0.0)):.4f}"
                    row_out["total_score_weighted"] = f"{float(j.get('total_score_weighted', 0.0)):.4f}"
                    row_out["judge_reasoning"] = str(j.get("judge_reasoning", ""))
                    row_out["judge_error"] = str(j.get("judge_error", ""))
                    row_out["judge_model"] = str(j.get("judge_model", judge_model))
                except Exception as exc:  # pragma: no cover
                    row_out["context_precision"] = ""
                    row_out["faithfulness"] = ""
                    row_out["answer_correctness"] = ""
                    row_out["psct_logic"] = ""
                    row_out["total_score_weighted"] = ""
                    row_out["judge_reasoning"] = ""
                    row_out["judge_error"] = f"{type(exc).__name__}: {exc}"
                    row_out["judge_model"] = judge_model

            writer.writerow(row_out)
            print(f"[{idx}/{total}] {'ok' if ok else 'failed'}")

    print(f"Done. success={success}, failed={fail}, output={output_path}")


if __name__ == "__main__":
    main()
