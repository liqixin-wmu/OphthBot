from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def filter_trainable_parameters(model):
    return filter(lambda p: p.requires_grad, model.parameters())


def join_if_relative(root: str, maybe_path: str) -> str:
    if os.path.isabs(maybe_path):
        return maybe_path
    return os.path.join(root, maybe_path)
