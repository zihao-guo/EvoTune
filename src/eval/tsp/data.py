from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
from scipy.spatial import distance_matrix


PERTURBATION_MOVES_MAP = {
    100: 20,
    200: 20,
}

ITER_LIMIT_MAP = {
    100: 16,
    200: 8,
}


@dataclass(frozen=True)
class TSPInstance:
    positions: np.ndarray
    adjacency: np.ndarray
    distmat: np.ndarray

    @property
    def n(self) -> int:
        return int(self.positions.shape[0])


def _build_distance_matrix(positions: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    n = positions.shape[0]
    distmat = distance_matrix(positions, positions) + np.eye(n) * 1e-5
    distmat = distmat * adjacency + (1.0 - adjacency) * 10.0
    return distmat


def load_instances(
    data_dir: Path,
    split: str,
    problem_size: int,
    drop_rate: float = 0.0,
) -> list[TSPInstance]:
    positions = np.load(data_dir / f"{split}{problem_size}_positions.npy")
    adjacency = np.load(data_dir / f"{split}{problem_size}_adjacency_{drop_rate}.npy")
    return [
        TSPInstance(pos, adj, _build_distance_matrix(pos, adj))
        for pos, adj in zip(positions, adjacency)
    ]


def load_optimal_objectives(
    data_dir: Path,
    split: str,
    drop_rate: float,
    problem_sizes: list[int],
) -> dict[int, float]:
    sizes_str = "_".join(str(size) for size in problem_sizes)
    perturbations_str = "_".join(str(PERTURBATION_MOVES_MAP[size]) for size in problem_sizes)
    iterlimits_str = "_".join(str(ITER_LIMIT_MAP[size]) for size in problem_sizes)
    path = data_dir / (
        f"optimal_objs_dict_{split}_drop_{drop_rate}_townsizes_{sizes_str}"
        f"_perturbations_{perturbations_str}_iterlimits_{iterlimits_str}.pkl"
    )
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    return {int(key): float(value) for key, value in raw.items()}


def calculate_cost(distmat: np.ndarray, tour: np.ndarray) -> float:
    return float(distmat[tour, np.roll(tour, 1)].sum())
