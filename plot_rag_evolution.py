#!/usr/bin/env python3
"""
Generate the definitive "Evolution of the RAG Pipeline" bar chart.

Default: embedded milestone means (from the project's judged evaluation runs).
Optional: --recompute to average metrics from judged CSVs on disk when present.

Requires: pip install matplotlib
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Any

# Canonical milestone labels (short for x-axis)
MILESTONES: list[dict[str, Any]] = [
    {
        "id": "V0",
        "label": "V0\nBaseline",
        "files": ["eval_results/test_results_raw_judged.csv"],
        "metrics": {
            "context_precision": 0.2812,
            "faithfulness": 0.3960,
            "answer_correctness": 0.3729,
            "psct_logic": 0.0950,
            "total_score_weighted": 0.2863,
        },
    },
    {
        "id": "V1",
        "label": "V1\nHybrid+MMR",
        "files": ["eval_results/test_results_raw_hybrid_judged.csv"],
        "metrics": {
            "context_precision": 0.6797,
            "faithfulness": 0.6727,
            "answer_correctness": 0.4138,
            "psct_logic": 0.3192,
            "total_score_weighted": 0.5214,
        },
    },
    {
        "id": "V2",
        "label": "V2\n+Exact card\n+prompts",
        "files": ["eval_results/test_results_raw_hybrid_fresh_judged.csv"],
        "metrics": {
            "context_precision": 0.6898,
            "faithfulness": 0.7946,
            "answer_correctness": 0.3582,
            "psct_logic": 0.4646,
            "total_score_weighted": 0.5768,
        },
    },
    {
        "id": "V3a",
        "label": "V3a\n+Coverage\n+2nd pass",
        "files": ["eval_results/test_results_full_eval_judged.csv"],
        "metrics": {
            "context_precision": 0.7069,
            "faithfulness": 0.8728,
            "answer_correctness": 0.2367,
            "psct_logic": 0.4693,
            "total_score_weighted": 0.5714,
        },
    },
    {
        "id": "V3b",
        "label": "V3b\n+Structured\n+ruling",
        "files": ["eval_results/test_results_full_eval_v2_judged.csv"],
        "metrics": {
            "context_precision": 0.7027,
            "faithfulness": 0.8765,
            "answer_correctness": 0.3463,
            "psct_logic": 0.2988,
            "total_score_weighted": 0.5561,
        },
    },
    {
        "id": "V3d",
        "label": "V3d\n+PSCT judge\nrubric",
        "files": ["eval_results/test_results_full_eval_v3_judged_psct_tuned.csv"],
        "metrics": {
            "context_precision": 0.7528,
            "faithfulness": 0.8565,
            "answer_correctness": 0.3322,
            "psct_logic": 0.5904,
            "total_score_weighted": 0.6330,
        },
    },
    {
        "id": "V5",
        "label": "V5\n+Q&A index\n(definitive)",
        "files": ["eval_results/test_results_full_eval_v5_judged.csv"],
        "metrics": {
            "context_precision": 0.9548,
            "faithfulness": 0.9579,
            "answer_correctness": 0.8868,
            "psct_logic": 0.5998,
            "total_score_weighted": 0.8499,
        },
    },
]

METRIC_KEYS = [
    "context_precision",
    "faithfulness",
    "answer_correctness",
    "psct_logic",
    "total_score_weighted",
]
METRIC_LABELS = [
    "Context precision",
    "Faithfulness",
    "Answer correctness",
    "PSCT logic",
    "Total (weighted)",
]


def _avg_from_csv(path: Path, keys: list[str]) -> dict[str, float] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    out: dict[str, float] = {}
    for k in keys:
        vals = []
        for r in rows:
            s = (r.get(k) or "").strip()
            if s:
                try:
                    vals.append(float(s))
                except ValueError:
                    pass
        if not vals:
            return None
        out[k] = mean(vals)
    return out


def _resolve_metrics(recompute: bool, project_root: Path) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    for m in MILESTONES:
        entry = dict(m)
        if recompute:
            merged: dict[str, float] | None = None
            for name in m["files"]:
                p = project_root / name
                got = _avg_from_csv(p, METRIC_KEYS)
                if got is not None:
                    merged = got
                    break
            if merged is not None:
                entry["metrics"] = merged
        resolved.append(entry)
    return resolved


def plot_chart(
    milestones: list[dict[str, Any]],
    out_path: Path,
    title: str,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(milestones)
    x = np.arange(n, dtype=float)
    n_metrics = len(METRIC_KEYS)
    width = 0.14
    offsets = np.linspace(-(n_metrics - 1) / 2 * width, (n_metrics - 1) / 2 * width, n_metrics)

    fig, ax = plt.subplots(figsize=(14.5, 6.2), layout="constrained")
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6", "#34495e"]

    for i, key in enumerate(METRIC_KEYS):
        vals = [float(m["metrics"][key]) for m in milestones]
        ax.bar(
            x + offsets[i],
            vals,
            width,
            label=METRIC_LABELS[i],
            color=colors[i],
            edgecolor="white",
            linewidth=0.6,
        )

    ax.set_ylabel("Judge score (mean, 0–1)")
    ax.set_xlabel("Pipeline milestone")
    ax.set_xticks(x)
    ax.set_xticklabels([m["label"] for m in milestones], fontsize=9)
    ax.set_ylim(0.0, 1.05)
    ax.axhline(1.0, color="#bdc3c7", linestyle="--", linewidth=0.8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        fontsize=9,
    )
    ax.grid(axis="y", alpha=0.35, linestyle=":")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RAG pipeline evolution (judge metrics).")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Overwrite embedded means with averages from judged CSVs in the repo when found.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures/evolution_rag_pipeline.png"),
        help="Output PNG path.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--title",
        default="Evolution of the YGO RAG pipeline (Judge LLM metrics, n=120)",
        help="Chart title.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    milestones = _resolve_metrics(args.recompute, root)

    try:
        plot_chart(milestones, args.out, args.title, args.dpi)
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required: pip install matplotlib\n" f"({exc})"
        ) from exc

    print(f"Wrote: {args.out.resolve()}")


if __name__ == "__main__":
    main()
