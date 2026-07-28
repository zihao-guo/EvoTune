import argparse
import os
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

LOCAL_PYVRP_REP = Path(__file__).resolve().parents[2] / "pyvrp_rep"
if str(LOCAL_PYVRP_REP) not in sys.path:
    sys.path.insert(0, str(LOCAL_PYVRP_REP))

from pyvrp import Model
from pyvrp.constants import MAX_VALUE
from pyvrp.stop import MaxIterations
from utils import get_config_value, load_data_config

OPEN_PROBLEMS = {
    "OVRP",
    "OVRPTW",
    "OVRPB",
    "OVRPL",
    "OVRPBL",
    "OVRPBTW",
    "OVRPLTW",
    "OVRPBLTW",
}

BACKHAUL_PROBLEMS = {
    "VRPB",
    "OVRPB",
    "VRPBL",
    "VRPBTW",
    "VRPBLTW",
    "OVRPBL",
    "OVRPBTW",
    "OVRPBLTW",
}

ROUTE_LIMIT_PROBLEMS = {
    "VRPL",
    "OVRPL",
    "VRPBL",
    "OVRPBL",
    "VRPLTW",
    "OVRPLTW",
    "VRPBLTW",
    "OVRPBLTW",
}

TW_PROBLEMS = {
    "VRPTW",
    "OVRPTW",
    "VRPBTW",
    "OVRPBTW",
    "VRPLTW",
    "OVRPLTW",
    "VRPBLTW",
    "OVRPBLTW",
}

SUPPORTED = {
    "CVRP",
    "OVRP",
    "VRPB",
    "VRPL",
    "VRPTW",
    "OVRPTW",
    "OVRPB",
    "OVRPL",
    "VRPBL",
    "VRPBTW",
    "VRPLTW",
    "OVRPBL",
    "OVRPBTW",
    "OVRPLTW",
    "VRPBLTW",
    "OVRPBLTW",
}

SECTION_HEADERS = {
    "NODE_COORD_SECTION",
    "DEMAND_SECTION",
    "TIME_WINDOW_SECTION",
    "BACKHAUL_SECTION",
    "DEPOT_SECTION",
    "EOF",
}


def iter_instances(base_dir: Path, problem_filter: str | None = None):
    if problem_filter:
        return sorted((base_dir / problem_filter).glob("*/*.vrp"))
    return sorted(base_dir.glob("*/*/*.vrp"))


def infer_problem(instance_path: Path) -> str:
    return instance_path.parents[1].name


def default_output_path(opt_dir: Path, instance_path: Path) -> Path:
    return opt_dir / instance_path.parents[1].name / instance_path.parent.name / "pyvrp" / f"{instance_path.stem}.sol"


def parse_header(line: str):
    if ":" not in line:
        return None, None
    key, value = line.split(":", 1)
    return key.strip(), value.strip().strip('"')


def get_default_depot_end(problem: str):
    return 4.6 if problem in TW_PROBLEMS else None


def parse_vrp_instance(path: Path, file_scale: int):
    headers = {}
    coords = {}
    demands = {}
    time_windows = {}
    backhauls = set()
    section = None

    with open(path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue

            if line in SECTION_HEADERS:
                section = None if line == "EOF" else line
                continue

            if section is None:
                key, value = parse_header(line)
                if key is not None:
                    headers[key] = value
                continue

            if section == "NODE_COORD_SECTION":
                idx, x, y = line.split()
                coords[int(idx)] = (int(x) / file_scale, int(y) / file_scale)
            elif section == "DEMAND_SECTION":
                idx, demand = line.split()
                demands[int(idx)] = int(demand)
            elif section == "TIME_WINDOW_SECTION":
                idx, start, end = line.split()
                time_windows[int(idx)] = (int(start) / file_scale, int(end) / file_scale)
            elif section == "BACKHAUL_SECTION":
                for token in line.split():
                    if token == "-1":
                        break
                    backhauls.add(int(token))
            elif section == "DEPOT_SECTION" and line == "-1":
                section = None

    dimension = int(headers["DIMENSION"])
    problem = headers["TYPE"]
    depot_xy = [list(coords[1])]
    node_xy = [list(coords[idx]) for idx in range(2, dimension + 1)]
    node_demand = []

    for idx in range(2, dimension + 1):
        demand = demands.get(idx, 0)
        if idx in backhauls:
            demand = -abs(demand)
        node_demand.append(demand)

    fields = {
        "depot_xy": depot_xy,
        "node_xy": node_xy,
        "node_demand": node_demand,
        "capacity": int(headers["CAPACITY"]),
        "route_limit": int(headers["DISTANCE"]) / file_scale if "DISTANCE" in headers else None,
        "service_time": None,
        "tw_start": None,
        "tw_end": None,
        "depot_tw_end": None,
    }

    if "SERVICE_TIME" in headers:
        service = int(headers["SERVICE_TIME"]) / file_scale
        fields["service_time"] = [service] * len(node_xy)

    if time_windows:
        fields["depot_tw_end"] = time_windows.get(1, (0.0, get_default_depot_end(problem)))[1]
        fields["tw_start"] = [time_windows[idx][0] for idx in range(2, dimension + 1)]
        fields["tw_end"] = [time_windows[idx][1] for idx in range(2, dimension + 1)]

    return problem, fields


def get_depot_end(problem: str, fields):
    return fields.get("depot_tw_end") or get_default_depot_end(problem)


def scale_xy(xy, coord_scale):
    return int(round(xy[0] * coord_scale)), int(round(xy[1] * coord_scale))


def flatten_routes(routes):
    flat = []
    for idx, route in enumerate(routes):
        if idx:
            flat.append(0)
        flat.extend(route)
    return flat


def split_flat_route(route):
    routes = []
    current = []
    for node in route:
        if node == 0:
            if current:
                routes.append(current)
                current = []
            continue
        current.append(node)
    if current:
        routes.append(current)
    return routes


def calc_route_cost(problem: str, depot_xy, node_xy, route):
    coords = [depot_xy[0]] + node_xy
    sequence = [0] + route + [0]
    total = 0.0

    for frm, to in zip(sequence[:-1], sequence[1:]):
        if problem in OPEN_PROBLEMS and to == 0:
            continue
        x1, y1 = coords[frm]
        x2, y2 = coords[to]
        total += math.hypot(x1 - x2, y1 - y2)

    return total


def build_model(problem: str, fields, coord_scale: int):
    model = Model()
    depot_kwargs = {}
    depot_end = get_depot_end(problem, fields)
    if depot_end is not None:
        depot_kwargs["tw_early"] = 0
        depot_kwargs["tw_late"] = int(round(depot_end * coord_scale))

    start_depot = model.add_depot(*scale_xy(fields["depot_xy"][0], coord_scale), name="start_depot", **depot_kwargs)
    end_depot = start_depot
    if problem in OPEN_PROBLEMS:
        end_depot = model.add_depot(*scale_xy(fields["depot_xy"][0], coord_scale), name="end_depot", **depot_kwargs)

    clients = []
    linehaul_clients = []
    backhaul_clients = []
    for idx, (xy, raw_demand) in enumerate(zip(fields["node_xy"], fields["node_demand"]), start=1):
        x_coord, y_coord = scale_xy(xy, coord_scale)
        demand = int(round(raw_demand))
        client_kwargs = {
            "x": x_coord,
            "y": y_coord,
            "name": f"c{idx}",
        }

        if demand >= 0:
            client_kwargs["delivery"] = demand
            client_kwargs["pickup"] = 0
        else:
            client_kwargs["delivery"] = 0
            client_kwargs["pickup"] = abs(demand)

        if problem in TW_PROBLEMS:
            client_kwargs["service_duration"] = int(round(fields["service_time"][idx - 1] * coord_scale))
            client_kwargs["tw_early"] = int(round(fields["tw_start"][idx - 1] * coord_scale))
            client_kwargs["tw_late"] = int(round(fields["tw_end"][idx - 1] * coord_scale))

        client = model.add_client(**client_kwargs)
        clients.append(client)
        if demand >= 0:
            linehaul_clients.append(client)
        else:
            backhaul_clients.append(client)

    locations = [start_depot] + clients
    if problem in OPEN_PROBLEMS:
        locations.append(end_depot)

    for frm in locations:
        for to in locations:
            if frm == to:
                model.add_edge(frm, to, distance=0, duration=0)
                continue

            if problem in OPEN_PROBLEMS and to == end_depot:
                model.add_edge(frm, to, distance=0, duration=0)
                continue

            if problem in BACKHAUL_PROBLEMS:
                # Allow pure backhaul routes that start from the depot, but
                # still forbid switching back to linehauls after a backhaul.
                backhaul_to_linehaul = frm in backhaul_clients and to in linehaul_clients
                if backhaul_to_linehaul:
                    model.add_edge(frm, to, distance=MAX_VALUE, duration=MAX_VALUE)
                    continue

            dist = int(round(math.hypot(frm.x - to.x, frm.y - to.y)))
            model.add_edge(frm, to, distance=dist, duration=dist)

    vehicle_kwargs = {
        "num_available": len(clients),
        "capacity": int(round(fields["capacity"])),
        "start_depot": start_depot,
        "end_depot": end_depot,
    }
    if depot_end is not None:
        vehicle_kwargs["tw_early"] = 0
        vehicle_kwargs["tw_late"] = int(round(depot_end * coord_scale))
    if problem in ROUTE_LIMIT_PROBLEMS and fields["route_limit"] is not None:
        vehicle_kwargs["max_distance"] = int(round(fields["route_limit"] * coord_scale))

    model.add_vehicle_type(**vehicle_kwargs)
    return model


def solve_one(problem: str, fields, coord_scale: int, max_iterations: int, seed: int):
    model = build_model(problem, fields, coord_scale)
    started = time.time()
    result = model.solve(stop=MaxIterations(max_iterations), seed=seed, display=False)
    runtime_s = time.time() - started
    if not result.is_feasible():
        raise RuntimeError(f"PyVRP returned an infeasible solution for {problem}.")

    depot_shift = 1 if problem in OPEN_PROBLEMS else 0
    routes = [[visit - depot_shift for visit in route.visits()] for route in result.best.routes()]
    flat_route = flatten_routes(routes)
    cost = calc_route_cost(problem, fields["depot_xy"], fields["node_xy"], flat_route)
    return cost, flat_route, runtime_s


def format_cost(cost: float) -> str:
    rounded = round(cost)
    if abs(cost - rounded) < 1e-9:
        return str(int(rounded))
    return f"{cost:.6f}".rstrip("0").rstrip(".")


def write_solution(path: Path, route, cost: float, runtime_s: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for idx, visits in enumerate(split_flat_route(route), start=1):
            fh.write(f"Route #{idx}: {' '.join(map(str, visits))}\n")
        fh.write(f"Cost {format_cost(cost)}\n")
        fh.write(f"Time {runtime_s:.6f}s\n")


def worker(task):
    idx, instance_str, coord_scale, file_scale, max_iterations, seed = task
    instance = Path(instance_str)
    try:
        problem, fields = parse_vrp_instance(instance, file_scale)
        cost, route, runtime_s = solve_one(problem, fields, coord_scale, max_iterations, seed)
        return idx, instance_str, cost, route, runtime_s, None
    except Exception as exc:  # Continue other instances even if one fails.
        return idx, instance_str, None, None, None, str(exc)


def parse_args():
    cfg = load_data_config()
    data_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Solve generated .vrp instances with pyvrp_rep and write .sol files.")
    parser.add_argument("--generated_dir", type=Path, default=data_dir / get_config_value(cfg, "paths", "generated_dir", "generated"))
    parser.add_argument("--opt_dir", type=Path, default=data_dir / get_config_value(cfg, "paths", "opt_dir", "opt"))
    parser.add_argument("--dataset", type=Path, default=None, help="Optional single .vrp instance path.")
    parser.add_argument("--problem", type=str, default=None, help="Optional problem folder filter, e.g. CVRP.")
    parser.add_argument("--num_instances", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 1) // 4))
    parser.add_argument("--coord_scale", type=int, default=get_config_value(cfg, "scaling", "coord_scale", 100000))
    parser.add_argument("--file_scale", type=int, default=get_config_value(cfg, "scaling", "file_scale", 1000))
    parser.add_argument("--max_iterations", type=int, default=get_config_value(cfg, "pyvrp", "max_iterations", 1000))
    parser.add_argument("--seed", type=int, default=get_config_value(cfg, "pyvrp", "seed", 0))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    instance_paths = [args.dataset.resolve()] if args.dataset else [path.resolve() for path in iter_instances(args.generated_dir, args.problem)]
    if args.num_instances is not None:
        instance_paths = instance_paths[:args.num_instances]

    tasks = []
    for idx, instance_path in enumerate(instance_paths):
        problem = infer_problem(instance_path)
        if problem not in SUPPORTED:
            print(f">> Skip {instance_path}: unsupported by pyvrp_rep baseline.")
            continue
        output = default_output_path(args.opt_dir, instance_path)
        if output.exists():
            print(f">> Skip existing {output}")
            continue
        tasks.append((idx, str(instance_path), args.coord_scale, args.file_scale, args.max_iterations, args.seed))

    if not tasks:
        print(">> Nothing to solve.")
        raise SystemExit(0)

    print("# ---------------")
    print(f"exact worker = {args.num_workers}")
    print("# ---------------")

    started = time.time()
    failures = []
    with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=get_context("spawn")) as executor:
        futures = {executor.submit(worker, task): task for task in tasks}
        for done_idx, future in enumerate(as_completed(futures), start=1):
            _, instance_str, cost, route, runtime_s, error = future.result()
            instance_path = Path(instance_str)
            if error is not None:
                failures.append((instance_str, error))
                print(f"[pyvrp][ERROR] {done_idx}/{len(tasks)} {instance_path} -> {error}")
                continue

            output = default_output_path(args.opt_dir, instance_path)
            write_solution(output, route, cost, runtime_s)
            print(f"[pyvrp] {done_idx}/{len(tasks)} {instance_path} runtime={runtime_s:.2f}s total_elapsed={time.time() - started:.2f}s -> {output}")

    if failures:
        print("# ---------------")
        print(f"pyvrp failures = {len(failures)}")
        for instance_str, error in failures:
            print(f"[pyvrp][FAILED] {instance_str} :: {error}")
