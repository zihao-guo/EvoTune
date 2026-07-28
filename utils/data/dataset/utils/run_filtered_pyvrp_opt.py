import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

from solve_reference_pyvrp import (
    default_output_path,
    get_config_value,
    iter_instances,
    load_data_config,
    worker,
    write_solution,
)


def main():
    cfg = load_data_config()
    data_dir = Path(__file__).resolve().parents[1]
    generated_dir = data_dir / get_config_value(cfg, "paths", "generated_dir", "generated")
    opt_dir = data_dir / get_config_value(cfg, "paths", "opt_dir", "opt")
    coord_scale = get_config_value(cfg, "scaling", "coord_scale", 100000)
    file_scale = get_config_value(cfg, "scaling", "file_scale", 1000)
    max_iterations = get_config_value(cfg, "pyvrp", "max_iterations", 1000)
    seed = get_config_value(cfg, "pyvrp", "seed", 0)
    num_workers = max(1, (os.cpu_count() or 1) // 4)
    allowed_sizes = {"501", "1001"}

    instance_paths = [path.resolve() for path in iter_instances(generated_dir) if path.parent.name in allowed_sizes]
    tasks = []
    for idx, instance_path in enumerate(instance_paths):
        output = default_output_path(opt_dir, instance_path)
        if output.exists():
            continue
        tasks.append((idx, str(instance_path), coord_scale, file_scale, max_iterations, seed))

    print(f"[start] workers={num_workers} pending={len(tasks)}", flush=True)
    started = time.time()
    failures = []

    with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context("spawn")) as executor:
        futures = {executor.submit(worker, task): task for task in tasks}
        for done_idx, future in enumerate(as_completed(futures), start=1):
            _, instance_str, cost, route, runtime_s, error = future.result()
            instance_path = Path(instance_str)
            if error is not None:
                failures.append((instance_str, error))
                print(f"[error] {done_idx}/{len(tasks)} {instance_path} :: {error}", flush=True)
                continue

            output = default_output_path(opt_dir, instance_path)
            write_solution(output, route, cost, runtime_s)
            print(
                f"[ok] {done_idx}/{len(tasks)} {instance_path} "
                f"runtime={runtime_s:.2f}s elapsed={time.time() - started:.2f}s -> {output}",
                flush=True,
            )

    print(f"[done] failures={len(failures)} elapsed={time.time() - started:.2f}s", flush=True)
    for instance_str, error in failures:
        print(f"[failed] {instance_str} :: {error}", flush=True)


if __name__ == "__main__":
    main()
