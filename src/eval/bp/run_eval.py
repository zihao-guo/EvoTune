from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from baselines import best_fit
from best_priority import priority
from data import DEFAULT_DATASET, l1_bound, l1_bound_dataset, load_dataset
from solver import pack_instance


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the extracted Bin Packing priority() function on a local evaluation dataset."
    )
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally evaluate only the first N instances from the dataset.",
    )
    return parser.parse_args()


def _write_instance_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _evaluate_priority_fn(dataset_items, priority_fn) -> dict[str, object]:
    rows = []
    bins_used = []
    lower_bounds = []
    durations = []

    for name, instance in dataset_items:
        started_at = time.perf_counter()
        used_bins, _ = pack_instance(instance["items"], instance["capacity"], priority_fn)
        duration = time.perf_counter() - started_at
        lower_bound = l1_bound(instance["items"], instance["capacity"])

        bins_used.append(used_bins)
        lower_bounds.append(lower_bound)
        durations.append(duration)
        rows.append(
            {
                "instance_name": name,
                "num_items": int(instance["num_items"]),
                "capacity": int(instance["capacity"]),
                "bins_used": used_bins,
                "l1_lower_bound": round(lower_bound, 6),
                "excess_percent": round((used_bins - lower_bound) / lower_bound * 100.0, 6),
                "duration_sec": round(duration, 6),
            }
        )

    mean_bins_used = float(np.mean(bins_used))
    mean_lower_bound = float(np.mean(lower_bounds))
    excess_percent = (mean_bins_used - mean_lower_bound) / mean_lower_bound * 100.0
    score = round(-excess_percent * 100.0, 2)
    return {
        "rows": rows,
        "summary": {
            "mean_bins_used": mean_bins_used,
            "mean_l1_lower_bound": mean_lower_bound,
            "excess_percent_vs_l1": excess_percent,
            "evotune_score": score,
            "mean_duration_sec": float(np.mean(durations)) if durations else 0.0,
            "total_duration_sec": float(np.sum(durations)),
        },
    }


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset)
    dataset_items = list(dataset.items())
    if args.limit is not None:
        dataset_items = dataset_items[: args.limit]

    run_started_at = time.perf_counter()
    best_priority_eval = _evaluate_priority_fn(dataset_items, priority)
    best_fit_eval = _evaluate_priority_fn(dataset_items, best_fit)

    comparison_rows = []
    best_fit_by_name = {
        row["instance_name"]: row for row in best_fit_eval["rows"]
    }
    best_priority_by_name = {
        row["instance_name"]: row for row in best_priority_eval["rows"]
    }
    for instance_name, best_priority_row in best_priority_by_name.items():
        best_fit_row = best_fit_by_name[instance_name]
        comparison_rows.append(
            {
                "instance_name": instance_name,
                "num_items": best_priority_row["num_items"],
                "capacity": best_priority_row["capacity"],
                "l1_lower_bound": best_priority_row["l1_lower_bound"],
                "best_priority_bins_used": best_priority_row["bins_used"],
                "best_fit_bins_used": best_fit_row["bins_used"],
                "best_priority_excess_percent": best_priority_row["excess_percent"],
                "best_fit_excess_percent": best_fit_row["excess_percent"],
                "absolute_excess_reduction_percent": round(
                    best_fit_row["excess_percent"] - best_priority_row["excess_percent"],
                    6,
                ),
            }
        )

    best_priority_summary = best_priority_eval["summary"]
    best_fit_summary = best_fit_eval["summary"]
    relative_improvement_vs_best_fit = (
        (best_fit_summary["excess_percent_vs_l1"] - best_priority_summary["excess_percent_vs_l1"])
        / best_fit_summary["excess_percent_vs_l1"]
        * 100.0
    )

    summary = {
        "heuristic_source": {
            "paper": "Algorithm Discovery With LLMs: Evolutionary Search Meets Reinforcement Learning",
            "arxiv": "2504.05108",
            "appendix": "A.9",
            "paper_label": "listing:best_program_bin",
            "model": "Phi 3.5 Instruct",
        },
        "dataset": args.dataset,
        "num_instances": len(dataset_items),
        "mean_dataset_l1_lower_bound_full": l1_bound_dataset(dataset),
        "heuristics": {
            "best_priority": best_priority_summary,
            "best_fit": best_fit_summary,
        },
        "comparisons": {
            "best_priority_vs_best_fit": {
                "absolute_excess_reduction_percent": (
                    best_fit_summary["excess_percent_vs_l1"] - best_priority_summary["excess_percent_vs_l1"]
                ),
                "relative_excess_reduction_percent": relative_improvement_vs_best_fit,
            },
        },
        "paper_reference": {
            "metric": "optimality gap percent, lower is better",
            "human_designed_heuristic": 5.37,
            "evotune": 2.06,
            "funsearch": 2.96,
        },
        "total_runtime_sec": time.perf_counter() - run_started_at,
    }

    summary_path = args.results_dir / f"{args.dataset}_summary.json"
    rows_path = args.results_dir / f"{args.dataset}_instances.csv"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    _write_instance_rows(rows_path, comparison_rows)

    print(json.dumps(summary, indent=2))
    print(f"Saved summary to {summary_path}")
    print(f"Saved per-instance metrics to {rows_path}")


if __name__ == "__main__":
    main()
