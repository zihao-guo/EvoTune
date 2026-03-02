from __future__ import annotations

import numpy as np


def best_fit(item: float, bins: np.ndarray) -> np.ndarray:
    """The human-designed best-fit heuristic used to initialize BP search."""
    return -(bins - item)


__all__ = ["best_fit"]
