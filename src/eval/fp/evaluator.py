from __future__ import annotations

import os
from typing import Callable

import jax
import numpy as np
from jumanji.environments import FlatPack

from data import PreloadedGenerator


PriorityFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def evaluate_flatpack_instance(env: FlatPack, random_key: jax.Array, priority_fn: PriorityFn) -> float:
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    state, timestep = reset_fn(random_key)
    obs = timestep.observation
    done = False

    while not done:
        q_values = priority_fn(
            np.asarray(obs.grid),
            np.asarray(obs.blocks),
            np.asarray(obs.action_mask),
        )
        q_values = np.asarray(q_values, dtype=np.float32)
        if q_values.shape != obs.action_mask.shape:
            raise ValueError(f"priority() returned shape {q_values.shape}, expected {obs.action_mask.shape}")

        masked_q_values = np.where(np.asarray(obs.action_mask), q_values, -np.inf)
        best_action_idx = np.unravel_index(np.argmax(masked_q_values), masked_q_values.shape)
        action = np.array(best_action_idx, dtype=np.int32)

        state, timestep = step_fn(state, action)
        obs = timestep.observation
        done = bool(timestep.last())

    grid = np.asarray(state.grid)
    proportion_cells_occupied = 1.0 - np.mean(grid == 0)
    return float(proportion_cells_occupied)


def evaluate_dataset(instance_file: str, priority_fn: PriorityFn, limit: int | None = None) -> list[float]:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    generator = PreloadedGenerator(instance_file=instance_file)
    key = jax.random.PRNGKey(0)
    scores = []
    num_instances = generator.num_instances if limit is None else min(generator.num_instances, limit)

    for _ in range(num_instances):
        cur_key, key = jax.random.split(key, num=2)
        env = FlatPack(generator=generator)
        score = evaluate_flatpack_instance(env, cur_key, priority_fn)
        scores.append(score)
    return scores
