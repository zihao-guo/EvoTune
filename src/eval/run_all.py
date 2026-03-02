from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = EVAL_ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect a single summary for the TSP, Bin Packing, and FlatPack eval sandboxes."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-run the per-task eval scripts before collecting the unified summary.",
    )
    return parser.parse_args()


def _run_script(script: Path) -> None:
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)


def _load_json(path: Path) -> dict[str, object]:
    with path.open() as handle:
        return json.load(handle)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_float(value: object, digits: int = 6) -> str:
    if value in ("", None):
        return "-"
    return f"{float(value):.{digits}f}"


def _format_percent(value: object, digits: int = 2) -> str:
    if value in ("", None):
        return "-"
    return f"{float(value):.{digits}f}%"


def _build_markdown_report(summary: dict[str, object]) -> str:
    tasks = summary["tasks"]
    tsp = tasks["tsp"]
    bp = tasks["bp"]
    fp = tasks["fp"]

    lines = [
        "# Evaluation Summary",
        "",
        f"Generated at (UTC): `{summary['generated_at_utc']}`",
        "",
        "## Table 1. Local Optimality Gap Results vs Baseline or Reference",
        "",
        "| Task | Dataset | Optimality gap | Gap reference | Local result | Baseline / Reference | Baseline gap | Reference gap | Absolute delta | Relative improvement |",
        "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
        (
            f"| TSP | test100+test200 | optimality gap (%) | optimal tour | "
            f"{_format_float(tsp['local_metric']['value'])} | "
            f"paper best single program | "
            f"- | "
            f"{_format_float(tsp['paper_reference']['best_single_program'])} | "
            f"{_format_float(tsp['comparison']['absolute_gap_delta_percent_points'])} | - |"
        ),
        (
            f"| BP | OR3_val | optimality gap (%) | L1 lower bound | "
            f"{_format_float(bp['local_metric']['value'])} | "
            f"best-fit | "
            f"{_format_float(bp['baseline_metric']['value'])} | "
            f"- | "
            f"{_format_float(bp['comparison']['absolute_optimality_gap_reduction_percent'])} | "
            f"{_format_percent(bp['comparison']['relative_optimality_gap_reduction_percent'])} |"
        ),
        (
            f"| FP | test_flatpack_dynamic_0_seed | optimality gap (fraction) | full occupancy | "
            f"{_format_float(fp['local_metric']['value'])} | "
            f"all_equal | "
            f"{_format_float(fp['baseline_metrics']['all_equal'])} | "
            f"- | "
            f"{_format_float(fp['comparison']['best_priority_vs_all_equal']['absolute_optimality_gap_reduction_fraction'])} | "
            f"{_format_percent(fp['comparison']['best_priority_vs_all_equal']['relative_optimality_gap_reduction_percent'])} |"
        ),
        (
            f"| FP | test_flatpack_dynamic_0_seed | optimality gap (fraction) | full occupancy | "
            f"{_format_float(fp['local_metric']['value'])} | "
            f"heuristic_flatpack | "
            f"{_format_float(fp['baseline_metrics']['heuristic_flatpack'])} | "
            f"- | "
            f"{_format_float(fp['comparison']['best_priority_vs_heuristic_flatpack']['absolute_optimality_gap_reduction_fraction'])} | "
            f"{_format_percent(fp['comparison']['best_priority_vs_heuristic_flatpack']['relative_optimality_gap_reduction_percent'])} |"
        ),
        "",
        "## Table 2. Comparison with Paper-Reported Optimality Gaps",
        "",
        "| Task | Paper optimality gap | Gap reference | Paper human-designed | Paper EvoTune | Paper FunSearch | Local reproduction |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        (
            f"| TSP | optimality gap (%) | optimal tour | - | "
            f"{_format_float(tsp['paper_reference']['best_single_program'])} "
            f"(best single program) | - | "
            f"{_format_float(tsp['local_metric']['value'])} |"
        ),
        (
            f"| BP | optimality gap (%) | L1 lower bound | "
            f"{_format_float(bp['paper_reference']['human_designed_heuristic'], 2)} | "
            f"{_format_float(bp['paper_reference']['evotune'], 2)} | "
            f"{_format_float(bp['paper_reference']['funsearch'], 2)} | "
            f"{_format_float(bp['local_metric']['value'])} |"
        ),
        (
            f"| FP | optimality gap (fraction) | full occupancy | "
            f"{_format_float(fp['paper_reference']['human_designed_heuristic'], 4)} | "
            f"{_format_float(fp['paper_reference']['evotune'], 4)} | "
            f"{_format_float(fp['paper_reference']['funsearch'], 4)} | "
            f"{_format_float(fp['local_metric']['value'])} |"
        ),
        "",
        "## Notes",
        "",
        "- All reported results are expressed as optimality gap. Lower is better.",
        "- TSP optimality gap is measured against the optimal tour.",
        "- BP optimality gap is measured against the L1 lower bound used by the original repository and paper.",
        "- FP optimality gap is defined as `1 - occupied_proportion`, i.e. gap to full occupancy.",
        "- BP local result compares the extracted `best_priority` against the `best-fit` baseline.",
        "- FP local result compares the extracted `best_priority` against both `all_equal` and `heuristic_flatpack` baselines.",
    ]
    return "\n".join(lines) + "\n"


def build_summary() -> tuple[dict[str, object], list[dict[str, object]]]:
    tsp_summary = _load_json(EVAL_ROOT / "tsp" / "results" / "test_summary.json")
    bp_summary = _load_json(EVAL_ROOT / "bp" / "results" / "OR3_val_summary.json")
    fp_summary = _load_json(EVAL_ROOT / "fp" / "results" / "test_flatpack_dynamic_0_seed_summary.json")

    tsp_local_gap = tsp_summary["average_optimality_gap_percent"]
    tsp_paper_gap = tsp_summary["paper_reference"]["best_single_program"]

    bp_best = bp_summary["heuristics"]["best_priority"]["excess_percent_vs_l1"]
    bp_baseline = bp_summary["heuristics"]["best_fit"]["excess_percent_vs_l1"]

    fp_best = fp_summary["heuristics"]["best_priority"]["gap_to_full_fraction"]
    fp_all_equal = fp_summary["heuristics"]["all_equal"]["gap_to_full_fraction"]
    fp_human = fp_summary["heuristics"]["heuristic_flatpack"]["gap_to_full_fraction"]

    comparison_table = [
        {
            "task": "TSP",
            "dataset": "test100+test200",
            "metric": "optimality_gap_percent",
            "gap_reference": "optimal_tour",
            "best_program_model": tsp_summary["heuristic_source"]["model"],
            "local_best": round(tsp_local_gap, 6),
            "baseline_or_reference": "paper_best_single_program",
            "baseline_gap": "",
            "reference_gap": round(tsp_paper_gap, 6),
            "absolute_delta": round(tsp_local_gap - tsp_paper_gap, 6),
            "relative_improvement_percent": "",
            "paper_human": "",
            "paper_evotune": "",
            "paper_funsearch": "",
        },
        {
            "task": "BP",
            "dataset": bp_summary["dataset"],
            "metric": "optimality_gap_percent",
            "gap_reference": "l1_lower_bound",
            "best_program_model": bp_summary["heuristic_source"]["model"],
            "local_best": round(bp_best, 6),
            "baseline_or_reference": "best_fit",
            "baseline_gap": round(bp_baseline, 6),
            "reference_gap": "",
            "absolute_delta": round(bp_baseline - bp_best, 6),
            "relative_improvement_percent": round(
                bp_summary["comparisons"]["best_priority_vs_best_fit"][
                    "relative_excess_reduction_percent"
                ],
                6,
            ),
            "paper_human": bp_summary["paper_reference"]["human_designed_heuristic"],
            "paper_evotune": bp_summary["paper_reference"]["evotune"],
            "paper_funsearch": bp_summary["paper_reference"]["funsearch"],
        },
        {
            "task": "FP",
            "dataset": Path(fp_summary["dataset"]).name,
            "metric": "optimality_gap_fraction",
            "gap_reference": "full_occupancy",
            "best_program_model": fp_summary["heuristic_source"]["model"],
            "local_best": round(fp_best, 6),
            "baseline_or_reference": "all_equal",
            "baseline_gap": round(fp_all_equal, 6),
            "reference_gap": "",
            "absolute_delta": round(fp_all_equal - fp_best, 6),
            "relative_improvement_percent": round(
                fp_summary["comparisons"]["best_priority_vs_all_equal"][
                    "relative_gap_reduction_percent"
                ],
                6,
            ),
            "paper_human": fp_summary["paper_reference"]["human_designed_heuristic"],
            "paper_evotune": fp_summary["paper_reference"]["evotune"],
            "paper_funsearch": fp_summary["paper_reference"]["funsearch"],
        },
        {
            "task": "FP",
            "dataset": Path(fp_summary["dataset"]).name,
            "metric": "optimality_gap_fraction",
            "gap_reference": "full_occupancy",
            "best_program_model": fp_summary["heuristic_source"]["model"],
            "local_best": round(fp_best, 6),
            "baseline_or_reference": "heuristic_flatpack",
            "baseline_gap": round(fp_human, 6),
            "reference_gap": "",
            "absolute_delta": round(fp_human - fp_best, 6),
            "relative_improvement_percent": round(
                fp_summary["comparisons"]["best_priority_vs_heuristic_flatpack"][
                    "relative_gap_reduction_percent"
                ],
                6,
            ),
            "paper_human": fp_summary["paper_reference"]["human_designed_heuristic"],
            "paper_evotune": fp_summary["paper_reference"]["evotune"],
            "paper_funsearch": fp_summary["paper_reference"]["funsearch"],
        },
    ]

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": {
            "tsp": {
                "summary_path": str(EVAL_ROOT / "tsp" / "results" / "test_summary.json"),
                "local_metric": {
                    "name": "average_optimality_gap_percent",
                    "value": tsp_local_gap,
                },
                "gap_reference": "optimal_tour",
                "paper_reference": tsp_summary["paper_reference"],
                "comparison": {
                    "reference_name": "paper_best_single_program",
                    "absolute_gap_delta_percent_points": tsp_local_gap - tsp_paper_gap,
                },
            },
            "bp": {
                "summary_path": str(EVAL_ROOT / "bp" / "results" / "OR3_val_summary.json"),
                "local_metric": {
                    "name": "optimality_gap_percent",
                    "value": bp_best,
                },
                "gap_reference": "l1_lower_bound",
                "baseline_metric": {
                    "name": "best_fit",
                    "value": bp_baseline,
                },
                "comparison": {
                    "absolute_optimality_gap_reduction_percent": bp_summary["comparisons"][
                        "best_priority_vs_best_fit"
                    ]["absolute_excess_reduction_percent"],
                    "relative_optimality_gap_reduction_percent": bp_summary["comparisons"][
                        "best_priority_vs_best_fit"
                    ]["relative_excess_reduction_percent"],
                },
                "paper_reference": bp_summary["paper_reference"],
            },
            "fp": {
                "summary_path": str(EVAL_ROOT / "fp" / "results" / "test_flatpack_dynamic_0_seed_summary.json"),
                "local_metric": {
                    "name": "optimality_gap_fraction",
                    "value": fp_best,
                },
                "gap_reference": "full_occupancy",
                "baseline_metrics": {
                    "all_equal": fp_all_equal,
                    "heuristic_flatpack": fp_human,
                },
                "comparison": {
                    "best_priority_vs_all_equal": {
                        "absolute_optimality_gap_reduction_fraction": fp_summary["comparisons"][
                            "best_priority_vs_all_equal"
                        ]["absolute_gap_reduction_fraction"],
                        "relative_optimality_gap_reduction_percent": fp_summary["comparisons"][
                            "best_priority_vs_all_equal"
                        ]["relative_gap_reduction_percent"],
                    },
                    "best_priority_vs_heuristic_flatpack": {
                        "absolute_optimality_gap_reduction_fraction": fp_summary["comparisons"][
                            "best_priority_vs_heuristic_flatpack"
                        ]["absolute_gap_reduction_fraction"],
                        "relative_optimality_gap_reduction_percent": fp_summary["comparisons"][
                            "best_priority_vs_heuristic_flatpack"
                        ]["relative_gap_reduction_percent"],
                    },
                },
                "paper_reference": fp_summary["paper_reference"],
            },
        },
        "comparison_table": comparison_table,
    }
    return summary, comparison_table


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    if args.refresh:
        _run_script(EVAL_ROOT / "tsp" / "run_eval.py")
        _run_script(EVAL_ROOT / "bp" / "run_eval.py")
        _run_script(EVAL_ROOT / "fp" / "run_eval.py")

    summary, comparison_table = build_summary()
    summary_path = args.results_dir / "summary.json"
    table_path = args.results_dir / "comparison_table.csv"
    report_path = args.results_dir / "report.md"

    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    _write_rows(table_path, comparison_table)
    report_path.write_text(_build_markdown_report(summary))

    print(json.dumps(summary, indent=2))
    print(f"Saved summary to {summary_path}")
    print(f"Saved comparison table to {table_path}")
    print(f"Saved markdown report to {report_path}")


if __name__ == "__main__":
    main()
