"""Standalone py3.12 bridge: compile + solve + score a Crex candidate.

This script is invoked as a subprocess from the py3.10 EvoTune worker,
using the dedicated interpreter that can import the vendored cp312 PyVRP
build (``/home/zguo/miniconda/envs/rlft/bin/python3.12``). It must remain
standalone: stdlib + pyvrp (+ numpy transitively via pyvrp) only. It must
NOT import ``packing`` -- the py3.10 and py3.12 environments are separate
and this file is imported by neither; it's only ever run as ``__main__``.

Frozen CLI (see docs/superpowers/plans/2026-07-28-hgs-crex-evolution.md,
Task 3)::

    <rlft-python3.12> hgs_eval_py312.py \
        --sandbox <abs path to sandbox pyvrp_rep root> --candidate <cpp path> \
        --variant CVRP --instances <dir with *.vrp> --baselines <dir with *.sol> \
        --num-instances 50 --max-iterations 1000 --seed 0 --compile 1 \
        --json-out <path>

JSON out (frozen shape)::

    {"ok": bool, "stage": "compile"|"solve"|"done",
     "compile_stderr_tail": str, "n_instances": int, "n_feasible": int,
     "mean_gap_percent": float,
     "per_instance": [{"name","obj","baseline","gap_percent","feasible"}, ...],
     "wall_seconds": float}

Core solve logic (parse_vrp_instance / build_manual_model /
calculate_route_cost) is ported near-verbatim from
``ReEvo/problems/hgs_share/eval.py``, which handles all six VRP variants
generically via the TYPE-header driven sets below. Dropped relative to
ReEvo: temp-sandbox cloning (sandboxes are persistent, primed elsewhere)
and the argv/stdout-line protocol (replaced by --flags + JSON file).
Anti-plagiarism enforcement is intentionally omitted entirely (not handled
on either side of the py3.10/py3.12 split): ReEvo's check used a similarity
threshold of 1.01, which no achievable similarity score can cross, making
it a permanent no-op in the original code too -- dropping it here is not a
behavior change.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Bridge interpreter's bin dir. meson.build shells out to a bare `python`,
# so it must resolve to this env's python (which has meson/ninja/pybind11).
RLFT_BIN = "/home/zguo/miniconda/envs/rlft/bin"

VRP_FILE_SCALE = 1000
MODEL_COORD_SCALE = 100000

OPEN_PROBLEM_TYPES = {"OVRP", "OVRPTW"}
BACKHAUL_PROBLEM_TYPES = {"VRPB"}
ROUTE_LIMIT_PROBLEM_TYPES = {"VRPL"}
TW_PROBLEM_TYPES = {"VRPTW", "OVRPTW"}
SECTION_HEADERS = {
    "NODE_COORD_SECTION",
    "DEMAND_SECTION",
    "TIME_WINDOW_SECTION",
    "BACKHAUL_SECTION",
    "DEPOT_SECTION",
    "EOF",
}

CANDIDATE_RELATIVE_PATH = Path("pyvrp") / "cpp" / "crossover" / "selective_route_exchange.cpp"
INFEASIBLE_GAP_PERCENT = 100.0  # deviation from ReEvo's huge penalty; keeps scores softmax-friendly
DIAGNOSTIC_TAIL_CHARS = 4000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sandbox", required=True, help="Absolute path to sandbox pyvrp_rep root")
    parser.add_argument("--candidate", required=True, help="Path to candidate .cpp file")
    parser.add_argument("--variant", required=True, help="VRP variant, e.g. CVRP, OVRP, VRPTW, ...")
    parser.add_argument("--instances", required=True, help="Directory containing *.vrp instance files")
    parser.add_argument("--baselines", required=True, help="Directory containing *.sol baseline files")
    parser.add_argument("--num-instances", type=int, default=50)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", type=int, default=1, choices=[0, 1])
    parser.add_argument("--json-out", required=True, help="Path to write result JSON")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# Ported from ReEvo/problems/hgs_share/eval.py: parse_vrp_instance,
# build_manual_model, calculate_route_cost. Generic across all six variants
# via the TYPE header driving the *_PROBLEM_TYPES sets above.
# --------------------------------------------------------------------------


def parse_header(line: str) -> tuple[str | None, str | None]:
    if ":" not in line:
        return None, None
    key, value = line.split(":", 1)
    return key.strip(), value.strip().strip('"')


def default_depot_tw_end(problem_type: str) -> float | None:
    return 4.6 if problem_type in TW_PROBLEM_TYPES else None


def parse_vrp_instance(instance_path: Path) -> dict:
    headers: dict[str, str] = {}
    coords: dict[int, tuple[float, float]] = {}
    demands: dict[int, int] = {}
    time_windows: dict[int, tuple[float, float]] = {}
    backhauls: set[int] = set()
    section = None

    with open(instance_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            if line in SECTION_HEADERS:
                section = None if line == "EOF" else line
                continue

            if section is None:
                key, value = parse_header(line)
                if key is not None and value is not None:
                    headers[key] = value
                continue

            if section == "NODE_COORD_SECTION":
                idx, x_coord, y_coord = line.split()
                coords[int(idx)] = (
                    int(x_coord) / VRP_FILE_SCALE,
                    int(y_coord) / VRP_FILE_SCALE,
                )
            elif section == "DEMAND_SECTION":
                idx, demand = line.split()
                demands[int(idx)] = int(demand)
            elif section == "TIME_WINDOW_SECTION":
                idx, start, end = line.split()
                time_windows[int(idx)] = (
                    int(start) / VRP_FILE_SCALE,
                    int(end) / VRP_FILE_SCALE,
                )
            elif section == "BACKHAUL_SECTION":
                for token in line.split():
                    if token == "-1":
                        break
                    backhauls.add(int(token))

    dimension = int(headers["DIMENSION"])
    problem_type = headers["TYPE"]
    depot_xy = coords[1]
    client_xy = [coords[idx] for idx in range(2, dimension + 1)]
    client_demands = []
    for idx in range(2, dimension + 1):
        demand = demands.get(idx, 0)
        if idx in backhauls:
            demand = -abs(demand)
        client_demands.append(demand)

    depot_tw_end = None
    tw_start = None
    tw_end = None
    if time_windows:
        depot_tw_end = time_windows.get(1, (0.0, default_depot_tw_end(problem_type)))[1]
        tw_start = [time_windows[idx][0] for idx in range(2, dimension + 1)]
        tw_end = [time_windows[idx][1] for idx in range(2, dimension + 1)]

    service_time = None
    if "SERVICE_TIME" in headers:
        service = int(headers["SERVICE_TIME"]) / VRP_FILE_SCALE
        service_time = [service] * len(client_xy)

    return {
        "problem_type": problem_type,
        "depot_xy": depot_xy,
        "client_xy": client_xy,
        "client_demands": client_demands,
        "capacity": int(headers["CAPACITY"]),
        "route_limit": int(headers["DISTANCE"]) / VRP_FILE_SCALE if "DISTANCE" in headers else None,
        "service_time": service_time,
        "tw_start": tw_start,
        "tw_end": tw_end,
        "depot_tw_end": depot_tw_end,
    }


def scale_coord(value: float) -> int:
    return int(round(value * MODEL_COORD_SCALE))


def calculate_route_cost(fields: dict, routes) -> float:
    problem_type = fields["problem_type"]
    coords = [fields["depot_xy"], *fields["client_xy"]]
    num_depots = 2 if problem_type in OPEN_PROBLEM_TYPES else 1
    total_cost = 0.0

    for route in routes:
        # PyVRP stores route visits as location indices. Convert back to the
        # compact [depot, clients...] coordinate list used by generated data.
        visits = [visit - num_depots + 1 for visit in route.visits()]
        sequence = [0, *visits, 0]
        for frm, to in zip(sequence[:-1], sequence[1:]):
            if problem_type in OPEN_PROBLEM_TYPES and to == 0:
                continue

            x1, y1 = coords[frm]
            x2, y2 = coords[to]
            total_cost += math.hypot(x1 - x2, y1 - y2)

    return total_cost


def build_manual_model(fields: dict):
    from pyvrp import Model
    from pyvrp.constants import MAX_VALUE

    problem_type = fields["problem_type"]
    model = Model()
    depot_kwargs = {}
    depot_tw_end = fields["depot_tw_end"] or default_depot_tw_end(problem_type)
    if depot_tw_end is not None:
        depot_kwargs["tw_early"] = 0
        depot_kwargs["tw_late"] = scale_coord(depot_tw_end)

    depot_x, depot_y = fields["depot_xy"]
    start_depot = model.add_depot(
        scale_coord(depot_x),
        scale_coord(depot_y),
        name="start_depot",
        **depot_kwargs,
    )
    end_depot = start_depot
    if problem_type in OPEN_PROBLEM_TYPES:
        end_depot = model.add_depot(
            scale_coord(depot_x),
            scale_coord(depot_y),
            name="end_depot",
            **depot_kwargs,
        )

    clients = []
    linehaul_clients = []
    backhaul_clients = []
    for idx, ((x_coord, y_coord), raw_demand) in enumerate(
        zip(fields["client_xy"], fields["client_demands"]),
        start=1,
    ):
        client_kwargs = {
            "x": scale_coord(x_coord),
            "y": scale_coord(y_coord),
            "name": f"c{idx}",
        }
        if raw_demand >= 0:
            client_kwargs["delivery"] = raw_demand
            client_kwargs["pickup"] = 0
        else:
            client_kwargs["delivery"] = 0
            client_kwargs["pickup"] = abs(raw_demand)

        if problem_type in TW_PROBLEM_TYPES:
            service_time = fields["service_time"] or [0.0] * len(fields["client_xy"])
            client_kwargs["service_duration"] = scale_coord(service_time[idx - 1])
            client_kwargs["tw_early"] = scale_coord(fields["tw_start"][idx - 1])
            client_kwargs["tw_late"] = scale_coord(fields["tw_end"][idx - 1])

        client = model.add_client(**client_kwargs)
        clients.append(client)
        if raw_demand >= 0:
            linehaul_clients.append(client)
        else:
            backhaul_clients.append(client)

    locations = [start_depot, *clients]
    if problem_type in OPEN_PROBLEM_TYPES:
        locations.append(end_depot)

    for frm in locations:
        for to in locations:
            if frm == to:
                model.add_edge(frm, to, distance=0, duration=0)
                continue
            if problem_type in OPEN_PROBLEM_TYPES and to == end_depot:
                model.add_edge(frm, to, distance=0, duration=0)
                continue
            if problem_type in BACKHAUL_PROBLEM_TYPES and frm in backhaul_clients and to in linehaul_clients:
                model.add_edge(frm, to, distance=MAX_VALUE, duration=MAX_VALUE)
                continue

            dist = int(round(math.hypot(frm.x - to.x, frm.y - to.y)))
            model.add_edge(frm, to, distance=dist, duration=dist)

    vehicle_kwargs = {
        "num_available": len(clients),
        "capacity": fields["capacity"],
        "start_depot": start_depot,
        "end_depot": end_depot,
    }
    if depot_tw_end is not None:
        vehicle_kwargs["tw_early"] = 0
        vehicle_kwargs["tw_late"] = scale_coord(depot_tw_end)
    if problem_type in ROUTE_LIMIT_PROBLEM_TYPES and fields["route_limit"] is not None:
        vehicle_kwargs["max_distance"] = scale_coord(fields["route_limit"])

    model.add_vehicle_type(**vehicle_kwargs)
    return model


def parse_baseline_cost(sol_path: Path) -> float:
    with open(sol_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if line.startswith("Cost "):
                return float(line.split()[1])
    raise ValueError(f"Failed to parse `Cost` line from `{sol_path}`.")


# --------------------------------------------------------------------------
# Compile step
# --------------------------------------------------------------------------


def compile_candidate(sandbox: Path, candidate_path: Path) -> tuple[bool, str]:
    """Writes the candidate source into the sandbox and rebuilds in place.

    Returns ``(ok, stderr_tail)``. ``stderr_tail`` is the last
    ``DIAGNOSTIC_TAIL_CHARS`` characters of combined stdout+stderr from
    whichever meson step failed, or ``""`` on success.
    """
    target = sandbox / CANDIDATE_RELATIVE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(candidate_path.read_text(encoding="utf-8"), encoding="utf-8")

    env = os.environ.copy()
    env["PATH"] = f"{RLFT_BIN}:{env.get('PATH', '')}"

    build_dir = sandbox / "build"
    for step_args in (
        ["meson", "compile", "-C", str(build_dir)],
        ["meson", "install", "-C", str(build_dir)],
    ):
        proc = subprocess.run(
            step_args,
            cwd=str(sandbox),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if proc.returncode != 0:
            return False, proc.stdout[-DIAGNOSTIC_TAIL_CHARS:]

    return True, ""


# --------------------------------------------------------------------------
# Solve step
# --------------------------------------------------------------------------


def ensure_local_pyvrp_import(sandbox: Path) -> None:
    """Imports `pyvrp` and verifies it resolved to this sandbox.

    The rlft interpreter's site-packages carries a `pyvrp.pth` pointing at
    an unrelated external checkout; sys.path.insert(0, sandbox) makes our
    copy shadow it, but we verify explicitly rather than trust that
    silently (a wrong-location import would produce bogus scores, not an
    error).
    """
    sandbox = sandbox.resolve()
    if str(sandbox) not in sys.path:
        sys.path.insert(0, str(sandbox))

    import pyvrp

    imported_from = Path(pyvrp.__file__).resolve()
    if sandbox not in imported_from.parents:
        raise ImportError(
            "Imported `pyvrp` from an unexpected location. "
            f"Expected under {sandbox}, got {imported_from}."
        )


def solve_instance(
    instance_path: Path,
    baseline_obj: float,
    seed: int,
    max_iterations: int,
) -> dict:
    from pyvrp.stop import MaxIterations

    fields = parse_vrp_instance(instance_path)
    model = build_manual_model(fields)
    result = model.solve(stop=MaxIterations(max_iterations), seed=seed, display=False)

    obj = calculate_route_cost(fields, result.best.routes())
    feasible = bool(result.is_feasible()) and math.isfinite(obj)
    gap_percent = (obj / baseline_obj - 1.0) * 100.0 if feasible else INFEASIBLE_GAP_PERCENT

    return {
        "name": instance_path.stem,
        "obj": obj,
        "baseline": baseline_obj,
        "gap_percent": gap_percent,
        "feasible": feasible,
    }


def default_result() -> dict:
    return {
        "ok": False,
        "stage": "compile",
        "compile_stderr_tail": "",
        "n_instances": 0,
        "n_feasible": 0,
        "mean_gap_percent": 0.0,
        "per_instance": [],
        "wall_seconds": 0.0,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def main(argv: list[str] | None = None) -> int:
    start = time.perf_counter()
    args = parse_args(argv)

    sandbox = Path(args.sandbox).resolve()
    candidate_path = Path(args.candidate).resolve()
    instances_dir = Path(args.instances).resolve()
    baselines_dir = Path(args.baselines).resolve()
    json_out = Path(args.json_out)

    result = default_result()

    try:
        if args.compile:
            ok, stderr_tail = compile_candidate(sandbox, candidate_path)
            if not ok:
                result["stage"] = "compile"
                result["compile_stderr_tail"] = stderr_tail
                write_json(json_out, {**result, "wall_seconds": time.perf_counter() - start})
                return 0

        result["stage"] = "solve"
        ensure_local_pyvrp_import(sandbox)

        instance_paths = sorted(instances_dir.glob("*.vrp"))[: args.num_instances]
        if not instance_paths or len(instance_paths) < args.num_instances:
            result["stage"] = "solve"
            result["compile_stderr_tail"] = (
                f"hgs: expected {args.num_instances} instance(s) under {instances_dir}, "
                f"found only {len(instance_paths)} loadable .vrp instance(s)."
            )
            write_json(json_out, {**result, "wall_seconds": time.perf_counter() - start})
            return 0

        missing_baselines = [
            instance_path.name for instance_path in instance_paths
            if not (baselines_dir / f"{instance_path.stem}.sol").is_file()
        ]
        if missing_baselines:
            result["stage"] = "solve"
            result["compile_stderr_tail"] = (
                f"hgs: missing baseline .sol under {baselines_dir} for instance(s): "
                + ", ".join(missing_baselines)
            )
            write_json(json_out, {**result, "wall_seconds": time.perf_counter() - start})
            return 0

        per_instance = []
        for instance_path in instance_paths:
            baseline_path = baselines_dir / f"{instance_path.stem}.sol"
            baseline_obj = parse_baseline_cost(baseline_path)
            per_instance.append(
                solve_instance(instance_path, baseline_obj, args.seed, args.max_iterations)
            )

        n_instances = len(per_instance)
        n_feasible = sum(1 for item in per_instance if item["feasible"])
        mean_gap_percent = (
            sum(item["gap_percent"] for item in per_instance) / n_instances if n_instances else 0.0
        )

        result.update(
            ok=True,
            stage="done",
            compile_stderr_tail="",
            n_instances=n_instances,
            n_feasible=n_feasible,
            mean_gap_percent=mean_gap_percent,
            per_instance=per_instance,
        )
    except Exception:
        result["ok"] = False
        result["compile_stderr_tail"] = traceback.format_exc()[-DIAGNOSTIC_TAIL_CHARS:]

    result["wall_seconds"] = time.perf_counter() - start
    write_json(json_out, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
