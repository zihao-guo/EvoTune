from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import chex
import jax.numpy as jnp
from jumanji.environments.packing.flat_pack.generator import InstanceGenerator
from jumanji.environments.packing.flat_pack.types import State


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TEST_SET = ROOT / "data" / "flat_pack" / "test_flatpack_dynamic_0_seed.json"


@dataclass(frozen=True)
class FlatPackMetadata:
    num_instances: int
    grid_sizes: list[list[int]]
    upper_bound_optimal_scores: list[int]


class PreloadedGenerator(InstanceGenerator):
    def __init__(self, instance_file: str | Path) -> None:
        with open(instance_file, "r") as handle:
            payload = json.load(handle)
        self.instances = payload["instances"]
        self.num_instances = len(self.instances)
        self.cur_idx = 0

        super().__init__(
            num_row_blocks=self.instances[self.cur_idx]["num_row_blocks"],
            num_col_blocks=self.instances[self.cur_idx]["num_col_blocks"],
        )

        self.num_row_blocks = self.instances[self.cur_idx]["num_row_blocks"]
        self.num_col_blocks = self.instances[self.cur_idx]["num_col_blocks"]

    def __call__(self, key: chex.PRNGKey) -> State:
        del key
        instance = self.instances[self.cur_idx]
        self.cur_idx = (self.cur_idx + 1) % self.num_instances
        self.num_row_blocks = self.instances[self.cur_idx]["num_row_blocks"]
        self.num_col_blocks = self.instances[self.cur_idx]["num_col_blocks"]

        return State(
            blocks=jnp.array(instance["state"]["blocks"], jnp.int32),
            num_blocks=instance["state"]["num_blocks"],
            action_mask=jnp.array(instance["state"]["action_mask"], bool),
            grid=jnp.array(instance["state"]["grid"], jnp.int32),
            step_count=instance["state"]["step_count"],
            key=jnp.array(instance["state"]["key"], jnp.uint32),
            placed_blocks=jnp.array(instance["state"]["placed_blocks"], bool),
        )


def load_metadata(instance_file: str | Path) -> FlatPackMetadata:
    with open(instance_file, "r") as handle:
        payload = json.load(handle)
    return FlatPackMetadata(
        num_instances=len(payload["instances"]),
        grid_sizes=payload["grid_sizes"],
        upper_bound_optimal_scores=[inst["upper_bound_optimal_score"] for inst in payload["instances"]],
    )
