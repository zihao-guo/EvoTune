"""Best TSP heuristic discovered by EvoTune with Granite 3.1 2B Instruct.

Source:
Algorithm Discovery With LLMs: Evolutionary Search Meets Reinforcement Learning
arXiv:2504.05108, Appendix A.9.
"""

import numpy as np


def heuristics(distance_matrix):
    average_distance = np.mean(distance_matrix, axis=1)
    distance_ranking = np.argsort(distance_matrix, axis=1)
    closest_mean_distance = np.mean(
        distance_matrix[np.arange(distance_matrix.shape[0])[:, None], distance_ranking[:, 1:5]],
        axis=1,
    )
    indicators = distance_matrix / average_distance[:, np.newaxis]
    normalized_distance_ranking = np.argsort(distance_ranking, axis=1) / distance_matrix.shape[0]
    indicators += normalized_distance_ranking
    indicators += closest_mean_distance[:, np.newaxis] / np.sum(distance_matrix, axis=1)[:, np.newaxis]
    indicators += distance_matrix / np.sum(distance_matrix, axis=1)[:, np.newaxis]
    indicators += np.where(
        distance_matrix > np.mean(distance_matrix),
        distance_matrix / np.mean(distance_matrix),
        0,
    )
    return indicators


__all__ = ["heuristics"]
