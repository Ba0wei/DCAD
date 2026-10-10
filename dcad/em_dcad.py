"""Event-aware masking core for EM-DCAD."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from .dcad import DCADConfig, apply_mask

START_CONTEXT = -1
END_CONTEXT = -2


@dataclass(frozen=True)
class EMDCADConfig(DCADConfig):
    adaptive_weight: float = 0.1
    adaptive_schedule_gamma: float = 3.0
    rarity_temperature: float = 1.0
    dfg_smoothing_alpha: float = 1.0


class TransitionRarityMasker:
    """Bias masking toward events surrounded by rare DFG transitions."""

    def __init__(self, sequences: Sequence[Sequence[int]], config: EMDCADConfig):
        self.config = config
        self.transition_counts: Counter[tuple[int, int]] = Counter()
        self.context_counts: Counter[int] = Counter()
        for sequence in sequences:
            extended = [START_CONTEXT, *map(int, sequence), END_CONTEXT]
            for left, right in zip(extended[:-1], extended[1:]):
                self.transition_counts[left, right] += 1
                self.context_counts[left] += 1

    def transition_rarity(self, left: int, right: int) -> float:
        alpha = self.config.dfg_smoothing_alpha
        vocabulary_size = max(len(self.context_counts), 1)
        probability = (self.transition_counts[left, right] + alpha) / (
            self.context_counts[left] + alpha * vocabulary_size
        )
        return -math.log(max(probability, 1e-12))

    def position_scores(self, sequence: Sequence[int]) -> list[float]:
        scores = []
        for index, token in enumerate(sequence):
            left = START_CONTEXT if index == 0 else int(sequence[index - 1])
            right = END_CONTEXT if index == len(sequence) - 1 else int(sequence[index + 1])
            scores.append(0.5 * (self.transition_rarity(left, int(token)) + self.transition_rarity(int(token), right)))
        return scores

    def sampling_probabilities(self, sequence: Sequence[int], mask_ratio: float) -> list[float]:
        progress = (mask_ratio - self.config.t_min) / max(self.config.t_max - self.config.t_min, 1e-8)
        progress = min(max(progress, 0.0), 1.0)
        adaptive = self.config.adaptive_weight * progress ** self.config.adaptive_schedule_gamma
        scaled = [score / self.config.rarity_temperature for score in self.position_scores(sequence)]
        maximum = max(scaled)
        exponentials = [math.exp(score - maximum) for score in scaled]
        total = sum(exponentials)
        uniform = 1.0 / len(sequence)
        return [(1.0 - adaptive) * uniform + adaptive * value / total for value in exponentials]

    def sample_positions(self, sequence: Sequence[int], mask_ratio: float, rng: random.Random) -> list[int]:
        count = min(len(sequence), max(1, round(len(sequence) * mask_ratio)))
        indices = list(range(len(sequence)))
        weights = self.sampling_probabilities(sequence, mask_ratio)
        selected = []
        for _ in range(count):
            choice = rng.choices(range(len(indices)), weights=weights, k=1)[0]
            selected.append(indices.pop(choice))
            weights.pop(choice)
        return selected


def prepare_em_dcad_example(
    token_ids: list[int],
    mask_token_id: int,
    mask_ratio: float,
    masker: TransitionRarityMasker,
    rng: random.Random,
) -> tuple[list[int], list[int], float]:
    """Create one event-aware corrupted training example for EM-DCAD."""
    positions = masker.sample_positions(token_ids, mask_ratio, rng)
    return apply_mask(token_ids, mask_token_id, positions)
