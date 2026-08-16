"""Variant-level train/validation splitting utilities."""

from __future__ import annotations

import random
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Sequence

from torch.utils.data import Dataset, Subset


@dataclass(frozen=True)
class VariantSplit:
    train_dataset: Dataset
    validation_dataset: Dataset | None
    train_indices: list[int]
    validation_indices: list[int]
    train_variant_count: int
    validation_variant_count: int


def _group_indices_by_variant(sequences: Sequence[Sequence[int]]) -> list[list[int]]:
    variant_to_indices: OrderedDict[tuple[int, ...], list[int]] = OrderedDict()
    for index, sequence in enumerate(sequences):
        variant = tuple(int(token_id) for token_id in sequence)
        variant_to_indices.setdefault(variant, []).append(index)
    return list(variant_to_indices.values())


def _count_tokens(sequences: Sequence[Sequence[int]], indices: Sequence[int]) -> Counter[int]:
    token_counts: Counter[int] = Counter()
    for index in indices:
        token_counts.update(int(token_id) for token_id in sequences[index])
    return token_counts


def _can_move_group_without_unknown_tokens(
    train_token_counts: Counter[int],
    group_token_counts: Counter[int],
) -> bool:
    return all(train_token_counts[token_id] > count for token_id, count in group_token_counts.items())


def split_dataset_by_variant(
    dataset: Dataset,
    sequences: Sequence[Sequence[int]],
    validation_ratio: float,
    seed: int,
) -> VariantSplit:
    if len(dataset) != len(sequences):
        raise ValueError(
            "Dataset length must match the number of sequences used for variant splitting. "
            f"Got len(dataset)={len(dataset)} and len(sequences)={len(sequences)}."
        )
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in [0, 1).")

    all_indices = list(range(len(sequences)))
    variant_groups = _group_indices_by_variant(sequences)
    if validation_ratio == 0.0 or len(sequences) < 2 or len(variant_groups) < 2:
        return VariantSplit(
            train_dataset=dataset,
            validation_dataset=None,
            train_indices=all_indices,
            validation_indices=[],
            train_variant_count=len(variant_groups),
            validation_variant_count=0,
        )

    rng = random.Random(seed)
    shuffled_groups = list(variant_groups)
    rng.shuffle(shuffled_groups)

    total_cases = len(sequences)
    target_validation_cases = max(1, round(total_cases * validation_ratio))

    train_token_counts = _count_tokens(sequences, all_indices)
    validation_group_ids: set[int] = set()
    validation_case_count = 0

    for group in shuffled_groups:
        group_token_counts = _count_tokens(sequences, group)
        if not _can_move_group_without_unknown_tokens(train_token_counts, group_token_counts):
            continue

        current_gap = abs(target_validation_cases - validation_case_count)
        next_case_count = validation_case_count + len(group)
        next_gap = abs(target_validation_cases - next_case_count)
        should_move = not validation_group_ids or next_gap <= current_gap
        if not should_move:
            continue

        train_token_counts.subtract(group_token_counts)
        validation_group_ids.add(id(group))
        validation_case_count = next_case_count

    train_groups = [group for group in shuffled_groups if id(group) not in validation_group_ids]
    validation_groups = [group for group in shuffled_groups if id(group) in validation_group_ids]
    train_indices = sorted(index for group in train_groups for index in group)
    validation_indices = sorted(index for group in validation_groups for index in group)

    if not validation_indices:
        return VariantSplit(
            train_dataset=dataset,
            validation_dataset=None,
            train_indices=all_indices,
            validation_indices=[],
            train_variant_count=len(variant_groups),
            validation_variant_count=0,
        )

    return VariantSplit(
        train_dataset=Subset(dataset, train_indices),
        validation_dataset=Subset(dataset, validation_indices),
        train_indices=train_indices,
        validation_indices=validation_indices,
        train_variant_count=len(train_groups),
        validation_variant_count=len(validation_groups),
    )
