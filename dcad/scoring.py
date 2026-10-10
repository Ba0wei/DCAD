"""Core anomaly-scoring procedure shared by the DCAD variants."""

from __future__ import annotations

import random

import numpy as np
import torch


@torch.no_grad()
def score_trace(
    model,
    token_ids: list[int],
    mask_token_id: int,
    t_max: float = 0.6,
    num_time_samples: int = 5,
    num_mask_samples: int = 5,
    seed: int = 2026,
) -> np.ndarray:
    """Average masked-token NLL over continuous times and corruption samples."""
    if not token_ids:
        return np.empty(0, dtype=np.float32)
    rng = random.Random(seed)
    device = next(model.parameters()).device
    length = len(token_ids)
    inputs, positions, targets, ratios = [], [], [], []
    for _ in range(num_time_samples):
        sampled_time = rng.random() * t_max
        masked_count = min(length, max(1, round(length * sampled_time)))
        for _ in range(num_mask_samples):
            for target_position in range(length):
                row = list(token_ids)
                candidates = [position for position in range(length) if position != target_position]
                extras = rng.sample(candidates, k=min(len(candidates), masked_count - 1))
                for position in [target_position, *extras]:
                    row[position] = mask_token_id
                inputs.append(row)
                positions.append(target_position)
                targets.append(token_ids[target_position])
                ratios.append(masked_count / length)
    input_tensor = torch.tensor(inputs, dtype=torch.long, device=device)
    logits = model(
        input_ids=input_tensor,
        attention_mask=torch.ones_like(input_tensor),
        mask_ratios=torch.tensor(ratios, dtype=torch.float32, device=device),
    )
    rows = torch.arange(len(inputs), device=device)
    probabilities = torch.softmax(logits[rows, torch.tensor(positions, device=device)], dim=-1)
    nll = -torch.log(probabilities[rows, torch.tensor(targets, device=device)].clamp_min(1e-12))
    return nll.reshape(num_time_samples * num_mask_samples, length).mean(0).cpu().numpy().astype(np.float32)
