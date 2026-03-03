#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from pprint import pprint
from typing import Any


DEFAULT_PICKLE_PATH = Path(
    "data/dataset/optimal_objs_dict_val_drop_0.2_townsizes_100_200_perturbations_20_20_iterlimits_16_8.pkl"
)


def summarize(obj: Any) -> str:
    if isinstance(obj, dict):
        keys = list(obj.keys())
        preview = keys[:5]
        return f"dict(len={len(obj)}, sample_keys={preview})"
    if isinstance(obj, list):
        return f"list(len={len(obj)})"
    if isinstance(obj, tuple):
        return f"tuple(len={len(obj)})"
    if isinstance(obj, set):
        return f"set(len={len(obj)})"
    return type(obj).__name__


def normalize(obj: Any) -> Any:
    # Convert numpy scalars or arrays to plain Python objects if available.
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except (ValueError, TypeError):
            pass
    if hasattr(obj, "tolist") and callable(obj.tolist):
        try:
            return obj.tolist()
        except (ValueError, TypeError):
            pass
    if isinstance(obj, dict):
        return {normalize(k): normalize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(normalize(v) for v in obj)
    if isinstance(obj, set):
        return {normalize(v) for v in obj}
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and print a pickle file.")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_PICKLE_PATH),
        help="Path to the pickle file. Defaults to the target dataset pickle.",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")

    with path.open("rb") as f:
        obj = pickle.load(f)

    print(f"Path: {path}")
    print(f"Top-level type: {type(obj).__name__}")
    print(f"Summary: {summarize(obj)}")
    print("Contents:")
    pprint(normalize(obj), sort_dicts=False)


if __name__ == "__main__":
    main()
