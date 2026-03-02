from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from baselines import all_equal, heuristic_flatpack
from best_priority import priority
from data import DEFAULT_TEST_SET, load_metadata
from evaluator import evaluate_dataset


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the extracted FlatPack priority() function on the local test set."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_TEST_SET)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally evaluate only the first N instances from the dataset.",
    )
    return parser.parse_args()


def _write_instance_rows(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _evaluate_priority_fn(dataset: Path, priority_fn, limit: int | None) -> dict[str, object]:
    started_at = time.perf_counter()
    scores = evaluate_dataset(str(dataset), priority_fn, limit=limit)
    total_runtime = time.perf_counter() - started_at
    mean_score = float(np.mean(scores)) if scores else 0.0
    return {
        "scores": scores,
        "summary": {
            "mean_occupied_proportion": mean_score,
            "mean_occupancy_percent": mean_score * 100.0,
            "gap_to_full_fraction": 1.0 - mean_score,
            "gap_to_full_percent": (1.0 - mean_score) * 100.0,
            "evotune_score": round(mean_score * 100.0, 3) - 100.0,
            "total_runtime_sec": total_runtime,
            "mean_duration_sec": total_runtime / len(scores) if scores else 0.0,
        },
    }


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.dataset)
    run_started_at = time.perf_counter()
    best_priority_eval = _evaluate_priority_fn(args.dataset, priority, args.limit)
    all_equal_eval = _evaluate_priority_fn(args.dataset, all_equal, args.limit)
    heuristic_flatpack_eval = _evaluate_priority_fn(args.dataset, heuristic_flatpack, args.limit)

    limit = len(best_priority_eval["scores"])
    rows = []
    for idx in range(limit):
        best_priority_score = best_priority_eval["scores"][idx]
        all_equal_score = all_equal_eval["scores"][idx]
        heuristic_flatpack_score = heuristic_flatpack_eval["scores"][idx]
        rows.append(
            {
                "instance_idx": idx,
                "upper_bound_optimal_score": metadata.upper_bound_optimal_scores[idx],
                "best_priority_occupancy_percent": round(best_priority_score * 100.0, 6),
                "all_equal_occupancy_percent": round(all_equal_score * 100.0, 6),
                "heuristic_flatpack_occupancy_percent": round(heuristic_flatpack_score * 100.0, 6),
                "best_priority_gap_fraction": round(1.0 - best_priority_score, 6),
                "all_equal_gap_fraction": round(1.0 - all_equal_score, 6),
                "heuristic_flatpack_gap_fraction": round(1.0 - heuristic_flatpack_score, 6),
            }
        )

    best_priority_summary = best_priority_eval["summary"]
    all_equal_summary = all_equal_eval["summary"]
    heuristic_flatpack_summary = heuristic_flatpack_eval["summary"]
    summary = {
        "heuristic_source": {
            "paper": "Algorithm Discovery With LLMs: Evolutionary Search Meets Reinforcement Learning",
            "arxiv": "2504.05108",
            "appendix": "A.9",
            "paper_label": "listing:best_program_fp",
            "model": "Llama 3.2 1B Instruct",
        },
        "dataset": str(args.dataset),
        "num_instances": limit,
        "mean_upper_bound_optimal_score": float(np.mean(metadata.upper_bound_optimal_scores[:limit])) if limit else 0.0,
        "heuristics": {
            "best_priority": best_priority_summary,
            "all_equal": all_equal_summary,
            "heuristic_flatpack": heuristic_flatpack_summary,
        },
        "comparisons": {
            "best_priority_vs_all_equal": {
                "absolute_gap_reduction_fraction": (
                    all_equal_summary["gap_to_full_fraction"] - best_priority_summary["gap_to_full_fraction"]
                ),
                "relative_gap_reduction_percent": (
                    (all_equal_summary["gap_to_full_fraction"] - best_priority_summary["gap_to_full_fraction"])
                    / all_equal_summary["gap_to_full_fraction"]
                    * 100.0
                ),
            },
            "best_priority_vs_heuristic_flatpack": {
                "absolute_gap_reduction_fraction": (
                    heuristic_flatpack_summary["gap_to_full_fraction"] - best_priority_summary["gap_to_full_fraction"]
                ),
                "relative_gap_reduction_percent": (
                    (
                        heuristic_flatpack_summary["gap_to_full_fraction"]
                        - best_priority_summary["gap_to_full_fraction"]
                    )
                    / heuristic_flatpack_summary["gap_to_full_fraction"]
                    * 100.0
                ),
            },
        },
        "paper_reference": {
            "metric": "optimality gap fraction, lower is better",
            "human_designed_heuristic": 0.1092,
            "evotune": 0.0829,
            "funsearch": 0.0898,
        },
        "total_runtime_sec": time.perf_counter() - run_started_at,
    }

    dataset_stem = args.dataset.stem
    summary_path = args.results_dir / f"{dataset_stem}_summary.json"
    rows_path = args.results_dir / f"{dataset_stem}_instances.csv"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    _write_instance_rows(rows_path, rows)

    print(json.dumps(summary, indent=2))
    print(f"Saved summary to {summary_path}")
    print(f"Saved per-instance metrics to {rows_path}")


if __name__ == "__main__":
    main()
