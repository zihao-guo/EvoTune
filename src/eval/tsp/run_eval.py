from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from best_heuristic import heuristics
from data import ITER_LIMIT_MAP, PERTURBATION_MOVES_MAP, load_instances, load_optimal_objectives
from gls_solver import solve_tsp, warm_up_numba


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = ROOT / "data" / "dataset"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the extracted Granite TSP heuristic inside a standalone GLS solver."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 200])
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally evaluate only the first N instances for each problem size.",
    )
    return parser.parse_args()


def _write_instance_rows(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    warm_up_numba()

    optimal_objectives = load_optimal_objectives(
        data_dir=args.data_dir,
        split=args.split,
        drop_rate=args.drop_rate,
        problem_sizes=args.sizes,
    )

    run_started_at = time.perf_counter()
    summary: dict[str, object] = {
        "heuristic_source": {
            "paper": "Algorithm Discovery With LLMs: Evolutionary Search Meets Reinforcement Learning",
            "arxiv": "2504.05108",
            "appendix": "A.9",
            "paper_label": "listing:best_program_tsp",
            "model": "ibm-granite/granite-3.1-2b-instruct",
        },
        "paper_reference": {
            "metric": "optimality gap percent, lower is better",
            "best_single_program": 2.446,
        },
        "split": args.split,
        "drop_rate": args.drop_rate,
        "sizes": args.sizes,
        "per_size": {},
    }

    all_gap_values = []
    all_rows: list[dict[str, float | int]] = []

    for size in args.sizes:
        instances = load_instances(args.data_dir, args.split, size, args.drop_rate)
        if args.limit is not None:
            instances = instances[: args.limit]

        costs = []
        durations = []
        for idx, instance in enumerate(instances):
            started_at = time.perf_counter()
            _, cost, _ = solve_tsp(
                distmat=instance.distmat,
                heuristic_fn=heuristics,
                perturbation_moves=PERTURBATION_MOVES_MAP[size],
                iter_limit=ITER_LIMIT_MAP[size],
            )
            duration = time.perf_counter() - started_at
            costs.append(cost)
            durations.append(duration)
            all_rows.append(
                {
                    "problem_size": size,
                    "instance_idx": idx,
                    "tour_cost": round(cost, 6),
                    "duration_sec": round(duration, 6),
                }
            )

        mean_cost = float(np.mean(costs))
        mean_duration = float(np.mean(durations))
        optimal_mean_cost = float(optimal_objectives[size])
        gap_percent = (mean_cost / optimal_mean_cost - 1.0) * 100.0
        all_gap_values.append(gap_percent)

        summary["per_size"][str(size)] = {
            "num_instances": len(instances),
            "perturbation_moves": PERTURBATION_MOVES_MAP[size],
            "iter_limit": ITER_LIMIT_MAP[size],
            "mean_cost": mean_cost,
            "optimal_mean_cost": optimal_mean_cost,
            "optimality_gap_percent": gap_percent,
            "mean_duration_sec": mean_duration,
            "total_duration_sec": float(np.sum(durations)),
        }

    summary["average_optimality_gap_percent"] = float(np.mean(all_gap_values))
    summary["total_runtime_sec"] = time.perf_counter() - run_started_at

    summary_path = args.results_dir / f"{args.split}_summary.json"
    rows_path = args.results_dir / f"{args.split}_instances.csv"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    _write_instance_rows(rows_path, all_rows)

    print(json.dumps(summary, indent=2))
    print(f"Saved summary to {summary_path}")
    print(f"Saved per-instance metrics to {rows_path}")


if __name__ == "__main__":
    main()
