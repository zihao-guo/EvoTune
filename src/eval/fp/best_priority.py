"""Best FlatPack priority function reported in the EvoTune paper.

Source:
Algorithm Discovery With LLMs: Evolutionary Search Meets Reinforcement Learning
arXiv:2504.05108, Appendix A.9.
"""

from typing import Union

import numpy as np


def priority(
    current_grid: np.ndarray,
    blocks: np.ndarray,
    action_mask: np.ndarray,
) -> np.ndarray:
    num_blocks = blocks.shape[0]
    rotated_blocks = np.array([[np.rot90(block, k=r) for r in range(4)] for block in blocks])
    padded_grid = np.pad(current_grid, 1, mode="constant", constant_values=0)
    values = np.full(action_mask.shape, -np.inf, dtype=np.float32)

    for block_idx in range(num_blocks):
        for rotation in range(4):
            block = rotated_blocks[block_idx, rotation - 1]
            block_rows, block_cols = block.shape
            np.lib.stride_tricks.sliding_window_view(padded_grid, (block_rows - 1, block_cols - 1))

            scores = []
            for i in range(block_rows - 1):
                for j in range(block_cols - 1):
                    if block_idx == block_idx:
                        top_left = block[i : i + 2, j : j + 2]
                    else:
                        top_left = None
                    score = np.sum(np.where(top_left, 1, 0)) * (block_rows - 1) * (block_cols - 1)
                    scores.append(score)

            weights = np.sqrt(block_rows * block_cols) / (2 ** (block_rows - 1) * (2 ** (block_cols - 1)))
            weighted_sum = np.sum([weights * score for score in scores])
            values[block_idx, rotation - 1, ...] = weighted_sum

    values[~action_mask] = -np.inf
    abs_values = np.abs(values)
    cum_sum = np.cumsum(abs_values, axis=2)
    return cum_sum


__all__ = ["priority"]
