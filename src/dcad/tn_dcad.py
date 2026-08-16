"""Trace-aware noising core for TN-DCAD."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .dcad import DCADConfig, IGNORE_INDEX, mdlm_loss_weight

CostLevel = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class TNDCADConfig(DCADConfig):
    t_min: float = 0.05
    t_max: float = 0.5
    low_cost_quantile: float = 0.25
    high_cost_quantile: float = 0.75
    trace_weight_beta: float = 0.5
    low_start_at_zero: bool = True


def assign_cost_levels(costs: Sequence[float], config: TNDCADConfig) -> list[CostLevel]:
    """Divide trace alignment costs into low, medium, and high groups."""
    low = float(np.quantile(costs, config.low_cost_quantile))
    high = float(np.quantile(costs, config.high_cost_quantile))
    return ["low" if cost <= low else "high" if cost >= high else "medium" for cost in costs]


def trace_noising_intervals(config: TNDCADConfig) -> dict[CostLevel, list[tuple[float, float]]]:
    """Construct the cost-conditioned continuous-time intervals."""
    low_start = 0.0 if config.low_start_at_zero else config.t_min
    return {
        "low": [(low_start, config.t_max)],
        "medium": [(config.t_min, config.t_max / 2), (config.t_max / 2, config.t_max)],
        "high": [
            (config.t_min, config.t_max / 3),
            (config.t_max / 3, 2 * config.t_max / 3),
            (2 * config.t_max / 3, config.t_max),
        ],
    }


def sample_trace_time(level: CostLevel, config: TNDCADConfig, rng: random.Random) -> float:
    intervals = trace_noising_intervals(config)[level]
    start, end = rng.choice(intervals)
    return start + rng.random() * (end - start)


def trace_contribution_weights(levels: Sequence[CostLevel], beta: float) -> list[float]:
    """Smooth unequal numbers of noising views contributed by each trace group."""
    views = {"low": 1, "medium": 2, "high": 3}
    raw = [views[level] ** (-beta) for level in levels]
    mean = sum(raw) / max(len(raw), 1)
    return [weight / mean for weight in raw]


def tn_dcad_reconstruction_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask_ratios: torch.Tensor,
    trace_weights: torch.Tensor,
) -> torch.Tensor:
    """Apply diffusion and trace-contribution weights to masked-token loss."""
    token_loss = F.cross_entropy(
        logits.transpose(1, 2), labels, ignore_index=IGNORE_INDEX, reduction="none"
    )
    valid = labels.ne(IGNORE_INDEX)
    per_trace = (token_loss * valid).sum(1) / valid.sum(1).clamp_min(1)
    return (per_trace * mdlm_loss_weight(mask_ratios) * trace_weights).mean()


def train_tn_dcad_batch(
    model,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    mask_ratios: torch.Tensor,
    trace_weights: torch.Tensor,
) -> float:
    """Run the trace-weighted TN-DCAD optimization step for one batch."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids=input_ids, attention_mask=attention_mask, mask_ratios=mask_ratios)
    loss = tn_dcad_reconstruction_loss(logits, labels, mask_ratios, trace_weights)
    loss.backward()
    optimizer.step()
    return float(loss.detach())
