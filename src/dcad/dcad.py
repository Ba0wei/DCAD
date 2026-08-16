"""Core DCAD configuration, masking, and diffusion objective."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


@dataclass(frozen=True)
class DCADConfig:
    """Paper-level model and optimization parameters for DCAD."""

    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    batch_size: int = 32
    epochs: int = 500
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    t_min: float = 0.0
    t_max: float = 0.6
    seed: int = 2026


def sample_continuous_time(config: DCADConfig, rng: random.Random) -> float:
    """Sample the continuous diffusion time used as the target mask ratio."""
    if not 0.0 <= config.t_min < config.t_max <= 1.0:
        raise ValueError("Expected 0 <= t_min < t_max <= 1.")
    return config.t_min + rng.random() * (config.t_max - config.t_min)


def uniform_mask_positions(length: int, mask_ratio: float, rng: random.Random) -> list[int]:
    """Sample DCAD's uniformly distributed masked event positions."""
    if length < 1:
        return []
    count = min(length, max(1, round(length * mask_ratio)))
    return rng.sample(range(length), k=count)


def apply_mask(
    token_ids: list[int], mask_token_id: int, positions: list[int]
) -> tuple[list[int], list[int], float]:
    """Create corrupted inputs and masked-token reconstruction targets."""
    selected = set(positions)
    inputs = [mask_token_id if i in selected else token for i, token in enumerate(token_ids)]
    labels = [token if i in selected else IGNORE_INDEX for i, token in enumerate(token_ids)]
    ratio = len(selected) / len(token_ids) if token_ids else 0.0
    return inputs, labels, ratio


def mdlm_loss_weight(mask_ratios: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Continuous-time MDLM weighting: d sigma / (exp(sigma) - 1)."""
    ratios = mask_ratios.clamp(min=eps, max=1.0 - eps)
    sigma = -torch.log1p(-ratios)
    d_sigma = 1.0 / (1.0 - ratios)
    return d_sigma / torch.expm1(sigma)


def diffusion_reconstruction_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask_ratios: torch.Tensor,
) -> torch.Tensor:
    """Compute DCAD's weighted masked-token reconstruction objective."""
    token_loss = F.cross_entropy(
        logits.transpose(1, 2), labels, ignore_index=IGNORE_INDEX, reduction="none"
    )
    valid = labels.ne(IGNORE_INDEX)
    per_trace = (token_loss * valid).sum(1) / valid.sum(1).clamp_min(1)
    return (per_trace * mdlm_loss_weight(mask_ratios)).mean()


def train_dcad_batch(
    model,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    mask_ratios: torch.Tensor,
) -> float:
    """Run the core forward/backward optimization step for one DCAD batch."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        mask_ratios=mask_ratios,
    )
    loss = diffusion_reconstruction_loss(logits, labels, mask_ratios)
    loss.backward()
    optimizer.step()
    return float(loss.detach())
