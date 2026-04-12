#!/usr/bin/env python3
"""Run the Judge LLM on an existing batch-eval CSV (no RAG re-run)."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

try:
    from openai import RateLimitError
except Exception:  # pragma: no cover
    RateLimitError = type("RateLimitError", (Exception,), {})  # type: ignore[misc,assignment]

from rag_judge_eval import default_judge_model, judge_rag_output

JUDGE_FIELDS = [
    "context_precision",
    "faithfulness",
    "answer_correctness",
    "psct_logic",
    "total_score_weighted",
    "judge_reasoning",
    "judge_error",
    "judge_model",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Add Judge LLM scores to a test_results_*.csv file.")
    parser.add_argument("--input", default="eval_results/test_results_raw.csv", help="Input CSV from run_batch_eval.py.")
    parser.add_argument(
        "--output",
        default="",
        help="Output CSV (default: input path with _judged before .csv, e.g. test_results_raw_judged.csv).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max rows to process (0 = all).")
    parser.add_argument("--judge-model", default="", help="Override YGO_JUDGE_MODEL / default.")
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=0.0,
        help="Optional delay between Judge API calls (rate limits).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Retries per row on rate limit / transient API errors.",
    )
    parser.add_argument(
        "--retry-backoff-sec",
        type=float,
        default=3.0,
        help="Base backoff seconds (exponential) after 429 / transient errors.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    out_path = Path(args.output) if args.output.strip() else input_path.with_name(
        input_path.stem + "_judged" + input_path.suffix
    )

    judge_model = (args.judge_model or "").strip() or default_judge_model()

    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    for jf in JUDGE_FIELDS:
        if jf not in fieldnames:
            fieldnames.append(jf)

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = 0
    fail = 0
    total = len(rows)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for idx, row in enumerate(rows, start=1):
            q = (row.get("Question") or "").strip()
            if not q:
                for jf in JUDGE_FIELDS:
                    row[jf] = ""
                writer.writerow(row)
                fail += 1
                print(f"[{idx}/{total}] skip (empty question)")
                continue

            last_err = ""
            j = None
            for attempt in range(1, max(1, args.retries) + 1):
                try:
                    j = judge_rag_output(
                        user_question=q,
                        official_answer=(row.get("Official_Answer") or "").strip(),
                        retrieved_context=(row.get("retrieved_contexts") or "[]").strip(),
                        rag_generated_answer=(row.get("generated_answer") or "").strip(),
                        judge_model=judge_model,
                        temperature=args.judge_temperature,
                    )
                    last_err = ""
                    break
                except RateLimitError as exc:
                    last_err = f"{type(exc).__name__}: {exc}"
                    if attempt < args.retries:
                        time.sleep(args.retry_backoff_sec * (2 ** (attempt - 1)))
                except Exception as exc:  # pragma: no cover
                    last_err = f"{type(exc).__name__}: {exc}"
                    retryable = "429" in str(exc).lower() or "rate limit" in str(exc).lower()
                    if retryable and attempt < args.retries:
                        time.sleep(args.retry_backoff_sec * (2 ** (attempt - 1)))
                    else:
                        break

            if j is not None and not last_err:
                m = j.get("metrics") or {}
                row["context_precision"] = f"{float(m.get('context_precision', 0.0)):.4f}"
                row["faithfulness"] = f"{float(m.get('faithfulness', 0.0)):.4f}"
                row["answer_correctness"] = f"{float(m.get('answer_correctness', 0.0)):.4f}"
                row["psct_logic"] = f"{float(m.get('psct_logic', 0.0)):.4f}"
                row["total_score_weighted"] = f"{float(j.get('total_score_weighted', 0.0)):.4f}"
                row["judge_reasoning"] = str(j.get("judge_reasoning", ""))
                row["judge_error"] = str(j.get("judge_error", ""))
                row["judge_model"] = str(j.get("judge_model", judge_model))
                ok += 1
            else:
                row["context_precision"] = ""
                row["faithfulness"] = ""
                row["answer_correctness"] = ""
                row["psct_logic"] = ""
                row["total_score_weighted"] = ""
                row["judge_reasoning"] = ""
                row["judge_error"] = last_err or "judge_rag_output returned no result"
                row["judge_model"] = judge_model
                fail += 1

            writer.writerow(row)
            err = row.get("judge_error", "")
            print(f"[{idx}/{total}] judge ok" if not err else f"[{idx}/{total}] judge err: {err[:80]}")

            if args.sleep_sec > 0 and idx < total:
                time.sleep(args.sleep_sec)

    print(f"Done. rows={total}, judge_ok={ok}, judge_fail={fail}, output={out_path}")


if __name__ == "__main__":
    main()
