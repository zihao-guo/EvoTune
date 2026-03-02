"""Best Bin Packing priority function reported in the EvoTune paper.

Source:
Algorithm Discovery With LLMs: Evolutionary Search Meets Reinforcement Learning
arXiv:2504.05108, Appendix A.9.
"""

import numpy as np


def priority(
    item: float,
    bins: np.ndarray,
    decay_rate: float = 1.2,
    load_balance_weight: float = 0.5,
    balance_threshold: float = 0.05,
    max_balance_bonus: float = 7.0,
    urgency_inflation_rate: float = 1.3,
    innovation_factor: float = 1.5,
    dynamic_state_weight: float = 0.25,
    time_weight: float = 0.1,
    real_time_optimization_step: float = 0.01,
    history_decay_rate: float = 0.95,
    urgency_trend_weight: float = 0.2,
    bin_state_adaptation_rate: float = 0.05,
    capacity_sensitivity_factor: float = 1.1,
    exploration_factor: float = 0.05,
    exploration_decay: float = 0.99,
    temporal_diversity_weight: float = 0.07,
) -> np.ndarray:
    ideal_capacity = np.mean(bins)
    balance_factor = np.where(
        np.abs(bins - ideal_capacity) <= balance_threshold,
        1,
        1 / (1 + np.abs(bins - ideal_capacity) / balance_threshold),
    )

    urgency_bonus = np.where(bins - item >= balance_threshold, urgency_inflation_rate, 1)
    time_influence = np.sin(np.arange(len(bins)) * real_time_optimization_step)
    adaptive_decay = (
        -(np.abs(bins - item) * decay_rate ** (np.abs(bins - item) * urgency_bonus * time_influence))
        * balance_factor
    )

    load_balance_score = np.clip(np.std(bins) / np.mean(bins) * load_balance_weight, 0, 1)
    exploration_bonus = np.clip(
        1 - np.exp(-np.sum(bins - item) / np.sum(bins) * exploration_factor),
        0,
        1,
    )
    capacity_sensitivity = np.power(np.max(bins) / np.min(bins), capacity_sensitivity_factor)
    temporal_diversity = np.exp(-np.arange(len(bins)) / np.max(bins) * temporal_diversity_weight)
    bin_state_impact = np.exp(-np.var(bins) * dynamic_state_weight) * temporal_diversity

    priority_scores = adaptive_decay * capacity_sensitivity
    priority_scores += load_balance_score * max_balance_bonus
    priority_scores += bin_state_impact
    priority_scores += exploration_bonus * exploration_factor
    priority_scores = (
        np.clip(priority_scores, 0, 1)
        * (1 + np.log1p(np.sum(bins - item)))
        * innovation_factor
        * time_weight
    )
    return priority_scores


__all__ = ["priority"]
