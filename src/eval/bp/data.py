from __future__ import annotations

import numpy as np

from packing.evaluate.bin_packing.bin_datasets import datasets


DEFAULT_DATASET = "OR3_val"


def available_datasets() -> list[str]:
    return sorted(datasets.keys())


def l1_bound(items: np.ndarray | list[int], capacity: int) -> float:
    return float(np.ceil(np.sum(items) / capacity))


def l1_bound_dataset(instances: dict) -> float:
    return float(
        np.mean([l1_bound(instance["items"], instance["capacity"]) for instance in instances.values()])
    )


def load_dataset(name: str) -> dict:
    if name not in datasets:
        raise KeyError(f"Unknown dataset {name!r}. Available: {available_datasets()}")
    return datasets[name]
