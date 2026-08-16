"""Shared seeds for fair training comparisons across model families."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SharedTrainingSeeds:
    data_split: int = 2026
    dataloader_shuffle: int = 2026
    model_init_dropout: int = 2026


COMMON_SEEDS = SharedTrainingSeeds()


def create_torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def seed_data_loader_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def shared_seed_payload() -> dict[str, int]:
    return asdict(COMMON_SEEDS)
