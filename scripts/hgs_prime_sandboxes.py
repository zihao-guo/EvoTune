"""CLI to pre-warm HGS/Crex per-worker sandboxes (Task 4 of the plan).

Each slot gets a persistent copy of ``pyvrp_rep`` at
``<repo>/scratch/hgs/<VARIANT>/w<slot>/pyvrp_rep`` with a full meson build.
Priming is idempotent (a slot that's already built is a fast no-op) and
flock-guarded per slot, so slots can be primed in parallel.

Usage::

    PYTHONPATH=src /home/zguo/miniconda/envs/evotune/bin/python \\
        scripts/hgs_prime_sandboxes.py --variant CVRP --slots 0 1 2
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _add_src_to_path(repo_root: Path) -> None:
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, help="VRP variant, e.g. CVRP")
    parser.add_argument("--slots", type=int, nargs="+", required=True, help="Worker slot indices to prime")
    parser.add_argument(
        "--repo-root", default=None,
        help="Override repo root used for `import packing` (defaults to this script's parent dir)",
    )
    parser.add_argument(
        "--sandbox-root", default="scratch/hgs",
        help="Repo-root-relative sandbox root (matches configs/task/*.yaml's task.sandbox_root)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)

    if args.repo_root:
        repo_root = Path(args.repo_root).resolve()
    else:
        repo_root = Path(__file__).resolve().parents[1]
    if not (repo_root / "pyvrp_rep").is_dir():
        raise SystemExit(f"--repo-root {repo_root} does not contain pyvrp_rep; pass --repo-root explicitly.")

    _add_src_to_path(repo_root)
    from packing.evaluate.hgs.hgs_common import ensure_slot  # noqa: E402 (path set up above)

    # A plain dict satisfies hgs_common._cfg_get just like a DictConfig would;
    # this script runs standalone, without Hydra.
    task_cfg = {"sandbox_root": args.sandbox_root}

    failures: list[int] = []
    with ThreadPoolExecutor(max_workers=len(args.slots)) as pool:
        futures = {pool.submit(ensure_slot, task_cfg, args.variant, slot): slot for slot in args.slots}
        for future in as_completed(futures):
            slot = futures[future]
            try:
                sandbox = future.result()
                logging.info(f"[hgs-prime] slot={slot} ready at {sandbox}")
            except Exception:
                logging.exception(f"[hgs-prime] slot={slot} FAILED to prime")
                failures.append(slot)

    if failures:
        logging.error(f"[hgs-prime] failed slots: {sorted(failures)}")
        return 1

    logging.info(f"[hgs-prime] all slots primed: {sorted(args.slots)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
