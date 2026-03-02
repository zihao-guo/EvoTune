from __future__ import annotations

from typing import Callable

import numba as nb
import numpy as np


FloatArray = np.ndarray
HeuristicFn = Callable[[FloatArray], FloatArray]
NUMBA_CACHE = True


@nb.njit(nb.float32(nb.float32[:, :], nb.uint16[:], nb.uint16), nogil=True, cache=NUMBA_CACHE)
def _two_opt_once(distmat, tour, fixed_i=0):
    n = tour.shape[0]
    p = q = 0
    delta = 0.0
    if fixed_i == 0:
        outer_range = range(1, n - 1)
    else:
        outer_range = range(fixed_i, fixed_i + 1)

    for i in outer_range:
        for j in range(i + 1, n):
            node_i, node_j = tour[i], tour[j]
            node_prev, node_next = tour[i - 1], tour[(j + 1) % n]
            if node_prev == node_j or node_next == node_i:
                continue
            change = (
                distmat[node_prev, node_j]
                + distmat[node_i, node_next]
                - distmat[node_prev, node_i]
                - distmat[node_j, node_next]
            )
            if change < delta:
                p, q, delta = i, j, change
    if delta < -1e-6:
        tour[p : q + 1] = np.flip(tour[p : q + 1])
        return delta
    return 0.0


@nb.njit(nb.float32(nb.float32[:, :], nb.uint16[:], nb.uint16), nogil=True, cache=NUMBA_CACHE)
def _relocate_once(distmat, tour, fixed_i=0):
    n = distmat.shape[0]
    delta = 0.0
    p = q = 0
    if fixed_i == 0:
        outer_range = range(1, n)
    else:
        outer_range = range(fixed_i, fixed_i + 1)

    for i in outer_range:
        node = tour[i]
        prev_node = tour[i - 1]
        next_node = tour[(i + 1) % n]
        for j in range(n):
            if j == i or j == i - 1:
                continue
            prev_insert = tour[j]
            next_insert = tour[(j + 1) % n]
            cost = (
                -distmat[prev_node, node]
                - distmat[node, next_node]
                - distmat[prev_insert, next_insert]
                + distmat[prev_insert, node]
                + distmat[node, next_insert]
                + distmat[prev_node, next_node]
            )
            if cost < delta:
                delta, p, q = cost, i, j
    if delta >= 0.0:
        return 0.0
    if p < q:
        tour[p : q + 1] = np.roll(tour[p : q + 1], -1)
    else:
        tour[q : p + 1] = np.roll(tour[q : p + 1], 1)
    return delta


@nb.njit(nb.float32(nb.float32[:, :], nb.uint16[:]), nogil=True, cache=NUMBA_CACHE)
def _calculate_cost(distmat, tour):
    cost = distmat[tour[-1], tour[0]]
    for i in range(len(tour) - 1):
        cost += distmat[tour[i], tour[i + 1]]
    return cost


@nb.njit(nb.float32(nb.float32[:, :], nb.uint16[:], nb.uint16, nb.uint16), nogil=True, cache=NUMBA_CACHE)
def _local_search(distmat, cur_tour, fixed_i=0, count=1000):
    sum_delta = 0.0
    delta = -1.0
    while delta < 0.0 and count > 0:
        delta = 0.0
        delta += _two_opt_once(distmat, cur_tour, fixed_i)
        delta += _relocate_once(distmat, cur_tour, fixed_i)
        count -= 1
        sum_delta += delta
    return sum_delta


@nb.njit(
    nb.void(nb.float32[:, :], nb.float32[:, :], nb.float32[:, :], nb.uint16[:], nb.float32, nb.uint32),
    nogil=True,
    cache=NUMBA_CACHE,
)
def _perturbation(distmat, guide, penalty, cur_tour, k, perturbation_moves=30):
    moves = 0
    n = distmat.shape[0]
    while moves < perturbation_moves:
        max_util = 0.0
        max_util_idx = 0
        for i in range(n - 1):
            j = i + 1
            u, v = cur_tour[i], cur_tour[j]
            util = guide[u, v] / (1.0 + penalty[u, v])
            if util > max_util:
                max_util_idx, max_util = i, util

        penalty[cur_tour[max_util_idx], cur_tour[max_util_idx + 1]] += 1.0
        edge_weight_guided = distmat + k * penalty

        for fixed_i in (max_util_idx, max_util_idx + 1):
            if fixed_i == 0 or fixed_i + 1 == n:
                continue
            delta = _local_search(edge_weight_guided, cur_tour, fixed_i, 1)
            if delta < 0.0:
                moves += 1


@nb.njit(nb.uint16[:](nb.float32[:, :], nb.uint16), nogil=True, cache=NUMBA_CACHE)
def _init_nearest_neighbor(distmat, start):
    n = distmat.shape[0]
    tour = np.zeros(n, dtype=np.uint16)
    visited = np.zeros(n, dtype=np.bool_)
    visited[start] = True
    tour[0] = start
    for i in range(1, n):
        min_dist = np.inf
        min_idx = -1
        for j in range(n):
            if not visited[j] and distmat[tour[i - 1], j] < min_dist:
                min_dist = distmat[tour[i - 1], j]
                min_idx = j
        tour[i] = min_idx
        visited[min_idx] = True
    return tour


@nb.njit(nb.uint16[:](nb.float32[:, :], nb.float32[:, :], nb.uint16, nb.int32, nb.uint16), nogil=True, cache=NUMBA_CACHE)
def _guided_local_search(distmat, guide, start, perturbation_moves=30, iter_limit=1000):
    penalty = np.zeros_like(distmat)
    best_tour = _init_nearest_neighbor(distmat, start)
    _local_search(distmat, best_tour, 0, 1000)
    best_cost = _calculate_cost(distmat, best_tour)
    k = 0.1 * best_cost / distmat.shape[0]
    cur_tour = best_tour.copy()

    for _ in range(iter_limit):
        _perturbation(distmat, guide, penalty, cur_tour, k, perturbation_moves)
        _local_search(distmat, cur_tour, 0, 1000)
        cur_cost = _calculate_cost(distmat, cur_tour)
        if cur_cost < best_cost:
            best_tour = cur_tour.copy()
            best_cost = cur_cost
    return best_tour


def build_guide_matrix(distmat: FloatArray, heuristic_fn: HeuristicFn) -> FloatArray:
    guide = heuristic_fn(distmat.copy())
    if guide.shape != distmat.shape:
        raise ValueError(
            f"heuristics() returned shape {guide.shape}, expected {distmat.shape}"
        )
    return np.asarray(guide, dtype=np.float32)


def guided_local_search(
    distmat: FloatArray,
    guide: FloatArray,
    perturbation_moves: int,
    iter_limit: int,
    start: int = 0,
) -> np.ndarray:
    return _guided_local_search(
        distmat=np.asarray(distmat, dtype=np.float32),
        guide=np.asarray(guide, dtype=np.float32),
        start=np.uint16(start),
        perturbation_moves=np.int32(perturbation_moves),
        iter_limit=np.uint16(iter_limit),
    )


def solve_tsp(
    distmat: FloatArray,
    heuristic_fn: HeuristicFn,
    perturbation_moves: int,
    iter_limit: int,
    start: int = 0,
) -> tuple[np.ndarray, float, FloatArray]:
    guide = build_guide_matrix(distmat, heuristic_fn)
    tour = guided_local_search(distmat, guide, perturbation_moves, iter_limit, start=start)
    cost = float(_calculate_cost(np.asarray(distmat, dtype=np.float32), tour))
    return tour, cost, guide


def warm_up_numba() -> None:
    dummy = np.ones((8, 8), dtype=np.float32)
    np.fill_diagonal(dummy, 1e-5)
    guide = dummy.copy()
    guided_local_search(dummy, guide, perturbation_moves=2, iter_limit=1)
