import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


DATA_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = DATA_DIR / "config.yaml"


def load_data_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def get_config_value(config: dict[str, Any], section: str, key: str, default: Any) -> Any:
    return config.get(section, {}).get(key, default)


def seed_everything(seed=2023):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def get_env(problem):
    from envs import (
        CVRPEnv,
        OVRPEnv,
        VRPBEnv,
        VRPLEnv,
        VRPTWEnv,
        OVRPTWEnv,
        OVRPBEnv,
        OVRPLEnv,
        VRPBLEnv,
        VRPBTWEnv,
        VRPLTWEnv,
        OVRPBLEnv,
        OVRPBTWEnv,
        OVRPLTWEnv,
        VRPBLTWEnv,
        OVRPBLTWEnv,
    )

    training_problems = ["CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW"]
    all_problems = {
        "CVRP": CVRPEnv,
        "OVRP": OVRPEnv,
        "VRPB": VRPBEnv,
        "VRPL": VRPLEnv,
        "VRPTW": VRPTWEnv,
        "OVRPTW": OVRPTWEnv,
        "OVRPB": OVRPBEnv,
        "OVRPL": OVRPLEnv,
        "VRPBL": VRPBLEnv,
        "VRPBTW": VRPBTWEnv,
        "VRPLTW": VRPLTWEnv,
        "OVRPBL": OVRPBLEnv,
        "OVRPBTW": OVRPBTWEnv,
        "OVRPLTW": OVRPLTWEnv,
        "VRPBLTW": VRPBLTWEnv,
        "OVRPBLTW": OVRPBLTWEnv,
    }

    if problem == "Train_ALL":
        return [all_problems[name] for name in training_problems]
    if problem == "ALL":
        return list(all_problems.values())
    return [all_problems[problem]]
