from __future__ import annotations

import numpy as np


def all_equal(
    current_grid: np.ndarray,
    blocks: np.ndarray,
    action_mask: np.ndarray,
) -> np.ndarray:
    num_blocks = blocks.shape[0]
    rotated_blocks = np.array([[np.rot90(block, k=r) for r in range(4, 0, -1)] for block in blocks])
    padded_grid = np.pad(current_grid, 1, mode="constant", constant_values=0)
    q_values = np.full(action_mask.shape, -np.inf)

    for block_idx in range(num_blocks):
        for rotation in range(4):
            block = rotated_blocks[block_idx, rotation]
            block_rows, block_cols = block.shape
            np.lib.stride_tricks.sliding_window_view(padded_grid, (block_rows, block_cols))
            q_values[block_idx, rotation, ...] = 1

    q_values[~action_mask] = -np.inf
    return q_values


def heuristic_flatpack(
    current_grid: np.ndarray,
    blocks: np.ndarray,
    action_mask: np.ndarray,
) -> np.ndarray:
    num_blocks = blocks.shape[0]
    rotated_blocks = np.array([[np.rot90(block, k=r) for r in range(4)] for block in blocks])
    block_sizes = np.sum(rotated_blocks[:, 0] > 0, axis=(1, 2))
    padded_grid = np.pad(current_grid, 1, mode="constant", constant_values=0)
    q_values = np.full(action_mask.shape, -np.inf)

    for block_idx in range(num_blocks):
        for rotation in range(4):
            block = rotated_blocks[block_idx, rotation]
            block_rows, block_cols = block.shape
            sub_grids = np.lib.stride_tricks.sliding_window_view(padded_grid, (block_rows, block_cols))
            adjacency_scores = np.sum((sub_grids > 0) & (block > 0), axis=(-2, -1))
            isolation_penalties = np.where(adjacency_scores > 0, 0, -block_sizes[block_idx])
            q_values[block_idx, rotation, ...] = (
                block_sizes[block_idx] + adjacency_scores[:-2, :-2] + isolation_penalties[:-2, :-2]
            )

    q_values[~action_mask] = -np.inf
    return q_values


__all__ = ["all_equal", "heuristic_flatpack"]
