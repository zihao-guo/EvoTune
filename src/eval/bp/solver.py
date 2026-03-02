from __future__ import annotations

from typing import Callable

import numpy as np


PriorityFn = Callable[[float, np.ndarray], np.ndarray]


def get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
    return np.nonzero((bins - item) >= 0)[0]


def pack_instance(
    items: list[int] | np.ndarray,
    capacity: int,
    priority_fn: PriorityFn,
) -> tuple[int, list[list[int]]]:
    num_items = len(items)
    bins = np.full(num_items, capacity, dtype=np.float64)
    packing = [[] for _ in range(num_items)]

    for item in items:
        valid_bin_indices = get_valid_bin_indices(item, bins)
        priorities = priority_fn(float(item), bins[valid_bin_indices])
        priorities = np.asarray(priorities, dtype=np.float64)
        if priorities.shape[0] != valid_bin_indices.shape[0]:
            raise ValueError(
                f"priority() returned {priorities.shape[0]} scores for {valid_bin_indices.shape[0]} valid bins"
            )
        if priorities.size == 0 or not np.all(np.isfinite(priorities)):
            raise ValueError("priority() must return a non-empty finite array")

        best_bin = valid_bin_indices[int(np.argmax(priorities))]
        bins[best_bin] -= item
        packing[best_bin].append(int(item))

    used_bins = int(np.sum(bins != capacity))
    packing = [bin_items for bin_items in packing if bin_items]
    return used_bins, packing
