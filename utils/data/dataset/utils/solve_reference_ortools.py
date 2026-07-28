import argparse
import os
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
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
        return sorted((base_dir / problem_filter).glob('*/*.vrp'))
    return sorted(base_dir.glob('*/*/*.vrp'))


def infer_problem(instance_path: Path) -> str:
    return instance_path.parents[1].name


def default_output_path(opt_dir: Path, instance_path: Path) -> Path:
    return opt_dir / instance_path.parents[1].name / instance_path.parent.name / "ortools" / f"{instance_path.stem}.sol"


def parse_header(line: str):
    if ':' not in line:
        return None, None
    key, value = line.split(':', 1)
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

    with open(path, 'r', encoding='utf-8') as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            if line in SECTION_HEADERS:
                section = None if line == 'EOF' else line
                continue
            if section is None:
                key, value = parse_header(line)
                if key is not None:
                    headers[key] = value
                continue
            if section == 'NODE_COORD_SECTION':
                idx, x, y = line.split()
                coords[int(idx)] = (int(x) / file_scale, int(y) / file_scale)
            elif section == 'DEMAND_SECTION':
                idx, demand = line.split()
                demands[int(idx)] = int(demand)
            elif section == 'TIME_WINDOW_SECTION':
                idx, start, end = line.split()
                time_windows[int(idx)] = (int(start) / file_scale, int(end) / file_scale)
            elif section == 'BACKHAUL_SECTION':
                for token in line.split():
                    if token == '-1':
                        break
                    backhauls.add(int(token))

    dimension = int(headers['DIMENSION'])
    problem = headers['TYPE']
    depot_xy = [list(coords[1])]
    node_xy = [list(coords[idx]) for idx in range(2, dimension + 1)]
    node_demand = []
    for idx in range(2, dimension + 1):
        demand = demands.get(idx, 0)
        if idx in backhauls:
            demand = -demand
        node_demand.append(demand)

    fields = {
        'depot_xy': depot_xy,
        'node_xy': node_xy,
        'node_demand': node_demand,
        'capacity': int(headers['CAPACITY']) if 'CAPACITY' in headers else None,
        'route_limit': int(headers['DISTANCE']) / file_scale if 'DISTANCE' in headers else None,
        'service_time': None,
        'tw_start': None,
        'tw_end': None,
        'depot_tw_end': None,
    }

    if 'SERVICE_TIME' in headers:
        service = int(headers['SERVICE_TIME']) / file_scale
        fields['service_time'] = [service] * len(node_xy)

    if time_windows:
        fields['depot_tw_end'] = time_windows.get(1, (0.0, get_default_depot_end(problem)))[1]
        fields['tw_start'] = [time_windows[idx][0] for idx in range(2, dimension + 1)]
        fields['tw_end'] = [time_windows[idx][1] for idx in range(2, dimension + 1)]

    return problem, fields


def get_depot_end(problem: str, fields):
    return fields.get('depot_tw_end') or get_default_depot_end(problem)


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


def build_data(problem: str, fields, coord_scale: int):
    depot = scale_xy(fields['depot_xy'][0], coord_scale)
    nodes = [scale_xy(xy, coord_scale) for xy in fields['node_xy']]
    locations = [depot] + nodes
    dummy_depot = None
    if problem in OPEN_PROBLEMS:
        dummy_depot = len(locations)
        locations.append(depot)

    size = len(locations)
    distance_matrix = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if problem in OPEN_PROBLEMS and j == dummy_depot:
                distance_matrix[i][j] = 0
            else:
                distance_matrix[i][j] = int(round(math.hypot(locations[i][0] - locations[j][0], locations[i][1] - locations[j][1])))

    data = {
        'problem': problem,
        'fields': fields,
        'distance_matrix': distance_matrix,
        'demands': [0] + [int(round(d)) for d in fields['node_demand']],
        'vehicle_capacity': int(round(fields['capacity'])),
        'num_vehicles': len(fields['node_xy']),
        'depot': 0,
        'dummy_depot': dummy_depot,
        'service_time': None,
        'time_windows': None,
        'route_limit': None,
        'coord_scale': coord_scale,
    }
    if dummy_depot is not None:
        data['demands'].append(0)
    if problem in TW_PROBLEMS:
        depot_end = int(round(get_depot_end(problem, fields) * coord_scale))
        service = [int(round(v * coord_scale)) for v in fields['service_time']]
        tw_start = [int(round(v * coord_scale)) for v in fields['tw_start']]
        tw_end = [int(round(v * coord_scale)) for v in fields['tw_end']]
        data['service_time'] = service
        data['time_windows'] = [(0, depot_end)] + list(zip(tw_start, tw_end))
        if dummy_depot is not None:
            data['time_windows'].append((0, depot_end))
    if problem in ROUTE_LIMIT_PROBLEMS and fields['route_limit'] is not None:
        data['route_limit'] = int(round(fields['route_limit'] * coord_scale))
    return data


def solve_one(problem: str, fields, coord_scale: int, time_limit_s: int):
    started = time.time()
    data = build_data(problem, fields, coord_scale)
    starts = [data['depot']] * data['num_vehicles']
    ends = [data['dummy_depot'] if data['dummy_depot'] is not None else data['depot']] * data['num_vehicles']
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], starts, ends)
    routing = pywrapcp.RoutingModel(manager)

    if problem in BACKHAUL_PROBLEMS:
        linehaul_nodes = [
            node
            for node, demand in enumerate(data['demands'])
            if node not in {data['depot'], data['dummy_depot']} and demand > 0
        ]
        backhaul_nodes = [
            node
            for node, demand in enumerate(data['demands'])
            if node not in {data['depot'], data['dummy_depot']} and demand < 0
        ]
        for from_node in backhaul_nodes:
            from_index = manager.NodeToIndex(from_node)
            for to_node in linehaul_nodes:
                routing.NextVar(from_index).RemoveValue(manager.NodeToIndex(to_node))

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_index,
        0,
        [data['vehicle_capacity']] * data['num_vehicles'],
        problem not in BACKHAUL_PROBLEMS,
        'Capacity',
    )

    if problem in ROUTE_LIMIT_PROBLEMS and data['route_limit'] is not None:
        routing.AddDimension(transit_index, 0, data['route_limit'], True, 'Distance')

    if problem in TW_PROBLEMS:
        time_horizon = int(round(get_depot_end(problem, fields) * data['coord_scale']))

        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            if data['dummy_depot'] is not None and to_node == data['dummy_depot']:
                return 0
            service = 0 if from_node == data['depot'] else data['service_time'][from_node - 1]
            return service + data['distance_matrix'][from_node][to_node]

        time_index = routing.RegisterTransitCallback(time_callback)
        routing.AddDimension(time_index, time_horizon, time_horizon, False, 'Time')
        time_dim = routing.GetDimensionOrDie('Time')
        for loc_idx, window in enumerate(data['time_windows']):
            if loc_idx == data['depot'] or loc_idx == data['dummy_depot']:
                continue
            index = manager.NodeToIndex(loc_idx)
            time_dim.CumulVar(index).SetRange(window[0], window[1])
        for vehicle_id in range(data['num_vehicles']):
            start_idx = routing.Start(vehicle_id)
            time_dim.CumulVar(start_idx).SetRange(0, 0)
            if data['dummy_depot'] is not None:
                end_idx = routing.End(vehicle_id)
                time_dim.CumulVar(end_idx).SetRange(0, time_horizon)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(int(time_limit_s))
    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        raise RuntimeError('OR-Tools did not return a feasible solution.')

    routes = []
    for vehicle_id in range(data['num_vehicles']):
        if not routing.IsVehicleUsed(solution, vehicle_id):
            continue
        visits = []
        index = routing.Start(vehicle_id)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node not in {data['depot'], data['dummy_depot']}:
                visits.append(node)
            index = solution.Value(routing.NextVar(index))
        if visits:
            routes.append(visits)

    flat_route = flatten_routes(routes)
    cost = calc_route_cost(problem, fields['depot_xy'], fields['node_xy'], flat_route)
    return cost, flat_route, time.time() - started


def format_cost(cost: float) -> str:
    rounded = round(cost)
    if abs(cost - rounded) < 1e-9:
        return str(int(rounded))
    return f"{cost:.6f}".rstrip("0").rstrip(".")


def write_solution(path: Path, route, cost: float, runtime_s: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        for idx, visits in enumerate(split_flat_route(route), start=1):
            fh.write(f"Route #{idx}: {' '.join(map(str, visits))}\n")
        fh.write(f"Cost {format_cost(cost)}\n")
        fh.write(f"Time {runtime_s:.6f}s\n")


def worker(task):
    idx, instance_str, coord_scale, file_scale, time_limit_s = task
    instance = Path(instance_str)
    try:
        problem, fields = parse_vrp_instance(instance, file_scale)
        cost, route, runtime_s = solve_one(problem, fields, coord_scale, time_limit_s)
        return idx, instance_str, cost, route, runtime_s, None
    except Exception as exc:  # Continue other instances even if one fails.
        return idx, instance_str, None, None, None, str(exc)


def parse_args():
    cfg = load_data_config()
    data_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Solve generated .vrp instances with OR-Tools and write .sol files.")
    parser.add_argument("--generated_dir", type=Path, default=data_dir / get_config_value(cfg, "paths", "generated_dir", "generated"))
    parser.add_argument("--opt_dir", type=Path, default=data_dir / get_config_value(cfg, "paths", "opt_dir", "opt"))
    parser.add_argument("--dataset", type=Path, default=None, help="Optional single .vrp instance path.")
    parser.add_argument("--problem", type=str, default=None, help="Optional problem folder filter, e.g. CVRP.")
    parser.add_argument("--num_instances", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 1) // 4))
    parser.add_argument("--coord_scale", type=int, default=get_config_value(cfg, "scaling", "coord_scale", 100000))
    parser.add_argument("--file_scale", type=int, default=get_config_value(cfg, "scaling", "file_scale", 1000))
    parser.add_argument("--time_limit", type=int, default=get_config_value(cfg, "ortools", "time_limit", 600))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    instance_paths = [args.dataset.resolve()] if args.dataset else [path.resolve() for path in iter_instances(args.generated_dir, args.problem)]
    if args.num_instances is not None:
        instance_paths = instance_paths[:args.num_instances]

    tasks = []
    for idx, instance_path in enumerate(instance_paths):
        output = default_output_path(args.opt_dir, instance_path)
        if output.exists():
            print(f">> Skip existing {output}")
            continue
        tasks.append((idx, str(instance_path), args.coord_scale, args.file_scale, args.time_limit))

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
                print(f"[ortools][ERROR] {done_idx}/{len(tasks)} {instance_path} -> {error}")
                continue

            output = default_output_path(args.opt_dir, instance_path)
            write_solution(output, route, cost, runtime_s)
            print(f"[ortools] {done_idx}/{len(tasks)} {instance_path} runtime={runtime_s:.2f}s total_elapsed={time.time() - started:.2f}s -> {output}")

    if failures:
        print("# ---------------")
        print(f"ortools failures = {len(failures)}")
        for instance_str, error in failures:
            print(f"[ortools][FAILED] {instance_str} :: {error}")
