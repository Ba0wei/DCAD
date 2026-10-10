#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tn-mdm.py

"""Train TN-MDM: Trace-aware Noising Masked Diffusion Model."""

from __future__ import annotations

import csv
import json
import os
import random
import sys
import argparse
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Optional, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dcad.model import DCADActivityModel as MDMActivityModel
from dcad.variant_split import split_dataset_by_variant


PAD_TOKEN = "[PAD]"
MASK_TOKEN = "[MASK]"
UNK_TOKEN = "[UNK]"
IGNORE_INDEX = -100
TRACE_LENGTHS_PATH = ROOT_DIR / "data" / "trace_lengths.json"
VALIDATION_MASK_SEED_OFFSET = 100000

DEFAULT_T_MIN = 0.05
DEFAULT_T_MAX = 0.5
TRACE_COST_LEVELS = ("low", "medium", "high")
TRACE_START_CHOICES = ("zero", "t_min")


def _trace_interval_start(mode: str, t_min: float) -> float:
    if mode == "zero":
        return 0.0
    if mode == "t_min":
        return float(t_min)
    raise ValueError(f"Unknown trace noising start mode: {mode!r}")


@dataclass
class TrainConfig:
    event_log: str = "BPIC17"
    # Data
    csv_path: str = "data/processed/splits/"+event_log+"_variant_train.csv"
    case_id_col: str = "case_id"
    activity_col: str = "name"
    max_len: Optional[int] = None
    alignment_cost_path: Optional[str] = 'outputs/alignment_costs/'+event_log+'_train_alignment_cost.csv'
    alignment_case_id_col: str = "case_id"
    alignment_cost_col: str = "alignment_cost"
    low_cost_quantile: float = 0.25
    high_cost_quantile: float = 0.75

    # Saving
    save_path: str = "outputs/model_train/"+event_log+"/TN-MDM_test"

    # Training
    batch_size: int = 32
    epochs: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 2026
    num_workers: int = 0
    validation_ratio: float = 0.1
    mask_seed: int = 2026
    validation_mask_seed: Optional[int] = None
    trace_weight_beta: float = 0.5
    t_min: float = DEFAULT_T_MIN
    t_max: float = DEFAULT_T_MAX
    low_start: str = "zero"
    medium_start: str = "t_min"
    high_start: str = "t_min"
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 1e-4

    # Model
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1


class ActivityCaseSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: Sequence[Sequence[int]],
        cost_levels: Optional[Sequence[str]] = None,
    ):
        self.sequences = [list(sequence) for sequence in sequences]
        self.cost_levels = list(cost_levels) if cost_levels is not None else ["low"] * len(sequences)

        if not self.sequences or len(self.cost_levels) != len(self.sequences):
            raise ValueError("Invalid or empty dataset.")
        invalid_levels = sorted(set(self.cost_levels) - set(TRACE_COST_LEVELS))
        if invalid_levels:
            raise ValueError(f"Unknown cost levels: {invalid_levels}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "sequence": self.sequences[index],
            "cost_level": self.cost_levels[index],
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_trace_noising_intervals(config: TrainConfig) -> dict[str, list[tuple[float, float]]]:
    low_start = _trace_interval_start(config.low_start, config.t_min)
    medium_start = _trace_interval_start(config.medium_start, config.t_min)
    high_start = _trace_interval_start(config.high_start, config.t_min)

    intervals = {
        "low": [(low_start, config.t_max)],
        "medium": [(medium_start, 0.5 * config.t_max), (0.5 * config.t_max, config.t_max)],
        "high": [
            (high_start, config.t_max / 3.0),
            (config.t_max / 3.0, 2.0 * config.t_max / 3.0),
            (2.0 * config.t_max / 3.0, config.t_max),
        ],
    }

    for level, level_intervals in intervals.items():
        for start, end in level_intervals:
            if not (0.0 <= start < end <= 1.0):
                raise ValueError(
                    f"Invalid noising interval for {level}: ({start}, {end}). "
                    "Check t_min, t_max, and start-mode settings."
                )
    return intervals


def ensure_dir(path: str | Path) -> None:
    os.makedirs(path, exist_ok=True)


def read_activity_cases_from_csv(
    csv_path: str,
    case_id_col: str,
    activity_col: str,
) -> list[tuple[str, list[str]]]:
    case_to_sequence: OrderedDict[str, list[str]] = OrderedDict()
    skipped_rows = 0

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        fieldnames = set(reader.fieldnames)
        missing = {case_id_col, activity_col} - fieldnames
        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}. Available columns: {reader.fieldnames}"
            )

        for row in reader:
            case_id = row.get(case_id_col)
            activity = row.get(activity_col)
            if case_id is None:
                skipped_rows += 1
                continue

            case_id = str(case_id).strip()
            activity = "" if activity is None else str(activity).strip()
            if case_id == "" or activity == "":
                skipped_rows += 1
                continue

            if case_id not in case_to_sequence:
                case_to_sequence[case_id] = []
            case_to_sequence[case_id].append(activity)

    cases = [(case_id, sequence) for case_id, sequence in case_to_sequence.items() if sequence]
    if not cases:
        raise ValueError("No non-empty case sequences were loaded from the CSV.")
    if skipped_rows > 0:
        print(f"Warning: skipped {skipped_rows} rows due to empty/missing case_id or activity.")
    return cases


def truncate_activity_cases(
    cases: Sequence[tuple[str, Sequence[str]]],
    max_len: Optional[int],
) -> list[tuple[str, list[str]]]:
    truncated_cases = []
    for case_id, sequence in cases:
        sequence_list = list(sequence)
        if max_len is not None:
            sequence_list = sequence_list[:max_len]
        if sequence_list:
            truncated_cases.append((case_id, sequence_list))
    return truncated_cases


def build_vocab(sequences: Sequence[Sequence[str]]) -> tuple[dict[str, int], dict[int, str]]:
    token_to_id = {
        PAD_TOKEN: 0,
        MASK_TOKEN: 1,
        UNK_TOKEN: 2,
    }
    for sequence in sequences:
        for activity in sequence:
            if activity not in token_to_id:
                token_to_id[activity] = len(token_to_id)
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    return token_to_id, id_to_token


def encode_sequences(sequences: Sequence[Sequence[str]], token_to_id: dict[str, int]) -> list[list[int]]:
    unk_id = token_to_id[UNK_TOKEN]
    return [[token_to_id.get(activity, unk_id) for activity in sequence] for sequence in sequences]


def _repo_relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR))
    except ValueError:
        return str(path.resolve())


def infer_log_key_from_train_csv(csv_path: str | Path) -> str:
    stem = Path(csv_path).name
    if stem.endswith(".csv"):
        stem = stem[:-4]
    for suffix in ("_variant_train", "_random_train", "_train"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def load_trace_length_index(path: Path = TRACE_LENGTHS_PATH) -> dict[str, dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Trace length index not found: {path}. "
            "Restore the released `data/trace_lengths.json` file."
        )

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    logs = payload.get("logs") if isinstance(payload, dict) else None
    if not isinstance(logs, dict):
        raise ValueError(f"Invalid trace length index: {path}")

    return {
        str(log_key): stats
        for log_key, stats in logs.items()
        if isinstance(stats, dict)
    }


def load_trace_lengths_for_train_csv(csv_path: str | Path) -> dict[str, object]:
    log_key = infer_log_key_from_train_csv(csv_path)
    index = load_trace_length_index()
    stats = index.get(log_key)
    if not isinstance(stats, dict):
        raise KeyError(
            f"Trace length index has no entry for {log_key!r}. "
            "Check the dataset entry in `data/trace_lengths.json`."
        )
    required = ("train_csv_max_trace_length", "custom_test_max_trace_length")
    missing = [key for key in required if key not in stats]
    if missing:
        raise KeyError(
            f"Trace length index entry {log_key!r} is missing {missing}. "
            "Check the dataset entry in `data/trace_lengths.json`."
        )
    return stats


def read_alignment_costs(path: str, case_id_col: str, cost_col: str) -> dict[str, float]:
    cost_by_case: dict[str, float] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Alignment cost CSV has no header: {path}")

        fieldnames = set(reader.fieldnames)
        missing = {case_id_col, cost_col} - fieldnames
        if missing:
            raise ValueError(
                f"Missing required alignment cost columns: {sorted(missing)}. "
                f"Available columns: {reader.fieldnames}"
            )

        for row_number, row in enumerate(reader, start=2):
            case_id = row.get(case_id_col)
            if case_id is None:
                continue

            case_id = str(case_id).strip()
            if case_id == "":
                continue

            raw_cost = row.get(cost_col)
            try:
                cost_by_case[case_id] = float(raw_cost)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid alignment cost at row {row_number} for case_id={case_id!r}: {raw_cost!r}"
                ) from exc

    return cost_by_case


def assign_cost_levels(
    target_case_ids: Sequence[str],
    cost_by_case: dict[str, float],
    low_quantile: float,
    high_quantile: float,
) -> tuple[list[str], dict[str, object]]:
    ranked_cases = [
        (case_id, cost_by_case[case_id], index)
        for index, case_id in enumerate(target_case_ids)
        if case_id in cost_by_case
    ]
    if not ranked_cases:
        raise ValueError("alignment_cost_path was provided, but none of the target case_ids had a cost.")

    ranked_cases.sort(key=lambda item: (item[1], item[2]))
    low_end = int(len(ranked_cases) * low_quantile)
    high_start = int(len(ranked_cases) * high_quantile)
    ranked_level_by_case: dict[str, str] = {}
    for rank, (case_id, _, _) in enumerate(ranked_cases):
        if rank < low_end:
            ranked_level_by_case[case_id] = "low"
        elif rank < high_start:
            ranked_level_by_case[case_id] = "medium"
        else:
            ranked_level_by_case[case_id] = "high"

    cost_levels = []
    missing_count = 0
    for case_id in target_case_ids:
        level = ranked_level_by_case.get(case_id)
        if level is None:
            cost_levels.append("low")
            missing_count += 1
        else:
            cost_levels.append(level)

    level_counts = dict(Counter(cost_levels))
    for level in TRACE_COST_LEVELS:
        level_counts.setdefault(level, 0)

    print(
        "Alignment cost levels: "
        f"low={level_counts['low']}, "
        f"medium={level_counts['medium']}, "
        f"high={level_counts['high']}, "
        f"missing_cost={missing_count}, "
        "threshold_source=target_cases_rank, "
        f"low_rank_fraction={low_quantile:.6f}, "
        f"high_rank_fraction={high_quantile:.6f}"
    )

    return cost_levels, {
        "alignment_cost_mode": "enabled_rank_percentile_levels",
        "alignment_cost_thresholds": {
            "low_rank_fraction": float(low_quantile),
            "high_rank_fraction": float(high_quantile),
        },
        "alignment_cost_threshold_source": "target_cases_rank",
        "alignment_cost_level_counts": level_counts,
        "alignment_cost_missing_count": missing_count,
    }


def mask_sequence(
    token_ids: Sequence[int],
    mask_token_id: int,
    mask_ratio: float = 0.15,
    ignore_index: int = IGNORE_INDEX,
    rng: Optional[random.Random] = None,
) -> tuple[list[int], list[int], float]:
    rng = rng if rng is not None else random
    seq_len = len(token_ids)
    if seq_len == 0:
        return [], [], 0.0

    num_to_mask = max(1, int(round(seq_len * mask_ratio)))
    num_to_mask = min(num_to_mask, seq_len)
    actual_mask_ratio = num_to_mask / seq_len
    mask_positions = set(rng.sample(range(seq_len), k=num_to_mask))

    input_ids = []
    labels = []
    for index, token_id in enumerate(token_ids):
        if index in mask_positions:
            input_ids.append(mask_token_id)
            labels.append(int(token_id))
        else:
            input_ids.append(int(token_id))
            labels.append(ignore_index)

    return input_ids, labels, actual_mask_ratio


def validate_config(config: TrainConfig) -> None:
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    if config.max_len is not None and config.max_len < 1:
        raise ValueError("max_len must be positive when set.")
    if not (0.0 <= config.validation_ratio < 1.0):
        raise ValueError("validation_ratio must be in [0, 1).")
    if config.early_stopping_patience < 1 or config.early_stopping_min_delta < 0.0:
        raise ValueError("Invalid early stopping settings.")
    if not (0.0 <= config.low_cost_quantile <= config.high_cost_quantile <= 1.0):
        raise ValueError("Invalid alignment cost quantiles.")
    if config.trace_weight_beta < 0.0:
        raise ValueError("trace_weight_beta must be non-negative.")
    if not (0.0 <= config.t_min < config.t_max <= 1.0):
        raise ValueError("t_min and t_max must satisfy 0 <= t_min < t_max <= 1.")
    for name, value in (
        ("low_start", config.low_start),
        ("medium_start", config.medium_start),
        ("high_start", config.high_start),
    ):
        if value not in TRACE_START_CHOICES:
            raise ValueError(f"{name} must be one of {TRACE_START_CHOICES}; got {value!r}.")


def compute_mdlm_loss_weight(mask_ratios: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mask_ratios = mask_ratios.clamp(min=eps, max=1.0 - eps)
    sigma = -torch.log1p(-mask_ratios)
    dsigma = 1.0 / (1.0 - mask_ratios)
    return dsigma / torch.expm1(sigma)


def build_trace_aware_noising_collate_fn(
    pad_token_id: int,
    mask_token_id: int,
    trace_weight_beta: float,
    trace_noising_intervals: dict[str, list[tuple[float, float]]],
    max_len: Optional[int] = None,
    mask_seed: Optional[int] = None,
):
    rng_by_worker: dict[int, random.Random] = {}

    def get_rng() -> random.Random:
        if mask_seed is None:
            return random
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        if worker_id not in rng_by_worker:
            rng_by_worker[worker_id] = random.Random(int(mask_seed) + worker_id)
        return rng_by_worker[worker_id]

    def collate_fn(batch: Sequence[dict[str, object]]) -> Dict[str, object]:
        masked_inputs = []
        masked_labels = []
        batch_mask_ratios = []
        batch_trace_weights = []
        rng = get_rng()

        for item in batch:
            seq = list(item["sequence"])
            if max_len is not None:
                seq = seq[:max_len]

            cost_level = str(item["cost_level"])
            if cost_level not in trace_noising_intervals:
                raise ValueError(f"Unknown cost level: {cost_level}")

            intervals = trace_noising_intervals[cost_level]
            views_per_trace = len(intervals)
            per_view_trace_weight = views_per_trace ** (trace_weight_beta - 1.0)

            for t_low, t_high in intervals:
                sampled_mask_ratio = rng.uniform(t_low, t_high)
                input_ids, labels, actual_mask_ratio = mask_sequence(
                    token_ids=seq,
                    mask_token_id=mask_token_id,
                    mask_ratio=sampled_mask_ratio,
                    ignore_index=IGNORE_INDEX,
                    rng=rng,
                )

                masked_inputs.append(input_ids)
                masked_labels.append(labels)
                batch_mask_ratios.append(actual_mask_ratio)
                batch_trace_weights.append(per_view_trace_weight)

        batch_max_len = max(len(seq) for seq in masked_inputs)

        padded_inputs = []
        padded_labels = []
        attention_masks = []

        for input_ids, labels in zip(masked_inputs, masked_labels):
            pad_len = batch_max_len - len(input_ids)
            padded_inputs.append(input_ids + [pad_token_id] * pad_len)
            padded_labels.append(labels + [IGNORE_INDEX] * pad_len)
            attention_masks.append([1] * len(input_ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "mask_ratios": torch.tensor(batch_mask_ratios, dtype=torch.float32),
            "trace_weights": torch.tensor(batch_trace_weights, dtype=torch.float32),
        }

    return collate_fn


def build_fixed_validation_samples(
    validation_dataset: Dataset,
    mask_token_id: int,
    mask_seed: int,
    trace_weight_beta: float,
    trace_noising_intervals: dict[str, list[tuple[float, float]]],
    max_len: Optional[int] = None,
) -> list[dict[str, object]]:
    fixed_samples = []

    for local_index in range(len(validation_dataset)):
        item = validation_dataset[local_index]
        seq = list(item["sequence"])
        if max_len is not None:
            seq = seq[:max_len]

        cost_level = str(item["cost_level"])
        if cost_level not in trace_noising_intervals:
            raise ValueError(f"Unknown cost level: {cost_level}")

        intervals = trace_noising_intervals[cost_level]
        views_per_trace = len(intervals)
        per_view_trace_weight = views_per_trace ** (trace_weight_beta - 1.0)

        for interval_index, (t_low, t_high) in enumerate(intervals):
            rng = random.Random(int(mask_seed) + local_index * 1000 + interval_index)
            sampled_mask_ratio = rng.uniform(t_low, t_high)
            input_ids, labels, actual_mask_ratio = mask_sequence(
                token_ids=seq,
                mask_token_id=mask_token_id,
                mask_ratio=sampled_mask_ratio,
                ignore_index=IGNORE_INDEX,
                rng=rng,
            )
            fixed_samples.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                    "mask_ratio": actual_mask_ratio,
                    "trace_weight": per_view_trace_weight,
                }
            )

    return fixed_samples


def build_fixed_validation_collate_fn(pad_token_id: int):
    def collate_fn(batch: Sequence[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        masked_inputs = [list(sample["input_ids"]) for sample in batch]
        masked_labels = [list(sample["labels"]) for sample in batch]
        batch_mask_ratios = [float(sample["mask_ratio"]) for sample in batch]
        batch_trace_weights = [float(sample.get("trace_weight", 1.0)) for sample in batch]

        batch_max_len = max(len(seq) for seq in masked_inputs)

        padded_inputs = []
        padded_labels = []
        attention_masks = []

        for input_ids, labels in zip(masked_inputs, masked_labels):
            pad_len = batch_max_len - len(input_ids)
            padded_inputs.append(input_ids + [pad_token_id] * pad_len)
            padded_labels.append(labels + [IGNORE_INDEX] * pad_len)
            attention_masks.append([1] * len(input_ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "mask_ratios": torch.tensor(batch_mask_ratios, dtype=torch.float32),
            "trace_weights": torch.tensor(batch_trace_weights, dtype=torch.float32),
        }

    return collate_fn


def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict[str, object]:
    is_training = optimizer is not None
    model.train(is_training)

    total_weighted_loss_sum = 0.0
    total_loss_sum = 0.0
    total_masked_tokens = 0
    total_trace_weighted_masked_tokens = 0.0
    total_mask_ratio = 0.0
    total_samples = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        mask_ratios = batch["mask_ratios"].to(device)
        trace_weights = batch.get("trace_weights")
        if trace_weights is None:
            trace_weights = torch.ones_like(mask_ratios)
        else:
            trace_weights = trace_weights.to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mask_ratios=mask_ratios,
            )

            batch_size, seq_len, vocab_size = logits.size()
            ce_flat = F.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=IGNORE_INDEX,
                reduction="none",
            )
            ce = ce_flat.view(batch_size, seq_len)
            valid_mask = labels != IGNORE_INDEX
            valid_mask_float = valid_mask.float()

            weights = compute_mdlm_loss_weight(mask_ratios).view(batch_size, 1)
            trace_weights_view = trace_weights.view(batch_size, 1)
            masked_ce = ce * valid_mask_float
            weighted_masked_ce = masked_ce * weights * trace_weights_view
            weighted_valid_mask = valid_mask_float * trace_weights_view
            loss = weighted_masked_ce.sum() / weighted_valid_mask.sum().clamp_min(1.0)

        masked_count = int(valid_mask.sum().item())
        if masked_count == 0:
            continue

        if is_training:
            loss.backward()
            optimizer.step()

        total_weighted_loss_sum += float(weighted_masked_ce.sum().item())
        total_loss_sum += float(masked_ce.sum().item())
        total_masked_tokens += masked_count
        total_trace_weighted_masked_tokens += float(weighted_valid_mask.sum().item())
        total_mask_ratio += float(mask_ratios.detach().sum().item())
        total_samples += int(mask_ratios.numel())

    if total_masked_tokens == 0:
        mode = "training" if is_training else "validation"
        raise RuntimeError(f"No masked tokens were generated during {mode}.")

    return {
        "avg_weighted_masked_ce_loss": (
            total_weighted_loss_sum / max(total_trace_weighted_masked_tokens, 1.0)
        ),
        "avg_masked_ce_loss": total_loss_sum / total_masked_tokens,
        "total_masked_tokens": total_masked_tokens,
        "avg_mask_ratio": total_mask_ratio / max(total_samples, 1),
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, object]:
    return _run_epoch(
        model=model,
        dataloader=dataloader,
        device=device,
        optimizer=optimizer,
    )


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    return _run_epoch(model=model, dataloader=dataloader, device=device)


def copy_state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def save_artifacts(
    save_dir: str,
    model: nn.Module,
    token_to_id: Dict[str, int],
    id_to_token: Dict[int, str],
    config: TrainConfig,
    max_len_source: str,
    max_len_details: dict[str, object],
    best_epoch: int,
    best_metrics: dict[str, object],
    early_stopping_monitor: str,
    best_monitor_loss: float,
    stopped_epoch: int,
    alignment_cost_info: dict[str, object],
    trace_noising_intervals: dict[str, list[tuple[float, float]]],
) -> None:
    ensure_dir(save_dir)

    model_path = os.path.join(save_dir, "model.pt")
    vocab_path = os.path.join(save_dir, "vocab.json")
    config_path = os.path.join(save_dir, "config.json")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        model_path,
    )

    vocab_payload = {
        "token_to_id": token_to_id,
        "id_to_token": {str(k): v for k, v in id_to_token.items()},
    }
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_payload, f, ensure_ascii=False, indent=2)

    config_payload = asdict(config)
    config_payload["configured_max_len"] = config_payload["max_len"]
    config_payload["max_len"] = int(model.max_len)
    config_payload["max_len_source"] = max_len_source
    config_payload["max_len_details"] = max_len_details
    config_payload.update(
        {
            "model_type": "TN-MDM",
            "loss_type": "mdlm_continuous_t_weighted_masked_ce",
            "noise_schedule": "trace_aware_alignment_cost_intervals",
            "t_min": float(config.t_min),
            "t_max": float(config.t_max),
            "trace_noising_start_modes": {
                "low": config.low_start,
                "medium": config.medium_start,
                "high": config.high_start,
            },
            "trace_noising_intervals": trace_noising_intervals,
            "loss_weight": "dsigma_over_expm1_sigma",
            "trace_contribution_smoothing": "views_per_trace_beta",
            "best_epoch": int(best_epoch),
            "best_metrics": best_metrics,
            "early_stopping_monitor": early_stopping_monitor,
            "best_monitor_loss": (
                float(best_monitor_loss) if np.isfinite(best_monitor_loss) else None
            ),
            "stopped_epoch": int(stopped_epoch),
        }
    )
    config_payload.update(alignment_cost_info)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, ensure_ascii=False, indent=2)

    print(f"Saved model to: {model_path}")
    print(f"Saved vocab to: {vocab_path}")
    print(f"Saved config to: {config_path}")


def train(config: TrainConfig) -> None:
    validate_config(config)
    validation_mask_seed = (
        int(config.validation_mask_seed)
        if config.validation_mask_seed is not None
        else int(config.seed) + VALIDATION_MASK_SEED_OFFSET
    )
    config = replace(
        config,
        mask_seed=int(config.mask_seed),
        validation_mask_seed=validation_mask_seed,
    )
    trace_noising_intervals = build_trace_noising_intervals(config)
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    raw_cases = read_activity_cases_from_csv(
        csv_path=config.csv_path,
        case_id_col=config.case_id_col,
        activity_col=config.activity_col,
    )
    raw_cases = truncate_activity_cases(raw_cases, config.max_len)
    if not raw_cases:
        raise ValueError("No non-empty case sequences remain after truncation.")

    case_ids = [case_id for case_id, _ in raw_cases]
    raw_sequences = [sequence for _, sequence in raw_cases]

    token_to_id, id_to_token = build_vocab(raw_sequences)
    encoded_sequences = encode_sequences(raw_sequences, token_to_id)

    pad_token_id = token_to_id[PAD_TOKEN]
    mask_token_id = token_to_id[MASK_TOKEN]

    trace_length_stats = load_trace_lengths_for_train_csv(config.csv_path)
    train_csv_max_len = int(trace_length_stats["train_csv_max_trace_length"])
    custom_test_max_len = int(trace_length_stats["custom_test_max_trace_length"])
    train_csv_boundary_max_len = int(
        trace_length_stats.get("train_csv_max_boundary_input_length", train_csv_max_len)
    )
    custom_test_boundary_max_len = int(
        trace_length_stats.get("custom_test_max_boundary_input_length", custom_test_max_len)
    )
    max_len_details: dict[str, object] = {
        "rule": "configured",
        "train_csv_path": trace_length_stats.get(
            "train_csv_path",
            _repo_relative_path(Path(config.csv_path)),
        ),
        "train_csv_max_trace_length": train_csv_max_len,
        "train_csv_max_boundary_input_length": train_csv_boundary_max_len,
        "custom_test_dataset": trace_length_stats.get("custom_test_dataset"),
        "custom_test_path": trace_length_stats.get("custom_test_path"),
        "custom_test_max_trace_length": custom_test_max_len,
        "custom_test_max_boundary_input_length": custom_test_boundary_max_len,
    }
    dataset_max_len = max(len(seq) for seq in encoded_sequences)
    if config.max_len is not None:
        model_max_len = config.max_len
        max_len_source = "configured"
    else:
        model_max_len = max(train_csv_boundary_max_len, custom_test_boundary_max_len)
        max_len_source = "trace_lengths"
        max_len_details["rule"] = (
            "max(train_csv_max_boundary_input_length, custom_test_max_boundary_input_length)"
        )

    print(f"Number of cases: {len(encoded_sequences)}")
    print(f"Vocabulary size: {len(token_to_id)}")
    print(f"Max sequence length in dataset: {dataset_max_len}")
    print(f"Max sequence length in train CSV: {train_csv_max_len}")
    print(f"Max boundary input length in train CSV: {train_csv_boundary_max_len}")
    print(f"Max sequence length in custom test: {custom_test_max_len}")
    print(f"Max boundary input length in custom test: {custom_test_boundary_max_len}")
    print(f"Model max_len: {model_max_len}")
    print(f"Model max_len source: {max_len_source}")
    print(f"TN-MDM trace-aware noising intervals: {trace_noising_intervals}")
    print(
        "TN-MDM noising start modes: "
        f"low={config.low_start}, "
        f"medium={config.medium_start}, "
        f"high={config.high_start}, "
        f"t_min={config.t_min}, "
        f"t_max={config.t_max}"
    )
    print(
        "Trace contribution smoothing: "
        f"trace_weight_beta={config.trace_weight_beta}"
    )
    print(
        "Early stopping: "
        f"enabled={config.early_stopping_enabled}, "
        f"patience={config.early_stopping_patience}, "
        f"min_delta={config.early_stopping_min_delta}"
    )

    split_dataset = ActivityCaseSequenceDataset(sequences=encoded_sequences)
    split = split_dataset_by_variant(
        dataset=split_dataset,
        sequences=encoded_sequences,
        validation_ratio=config.validation_ratio,
        seed=config.seed,
    )

    if config.alignment_cost_path is None:
        cost_levels = ["low"] * len(case_ids)
        alignment_cost_info: dict[str, object] = {
            "alignment_cost_mode": "disabled_all_low",
            "alignment_cost_thresholds": {
                "q_low": None,
                "q_high": None,
            },
            "alignment_cost_level_counts": {
                "low": len(case_ids),
                "medium": 0,
                "high": 0,
            },
            "alignment_cost_missing_count": 0,
        }
        print("Alignment cost mode: disabled_all_low; all traces use low-cost noising.")
    else:
        cost_by_case = read_alignment_costs(
            path=config.alignment_cost_path,
            case_id_col=config.alignment_case_id_col,
            cost_col=config.alignment_cost_col,
        )
        cost_levels, alignment_cost_info = assign_cost_levels(
            target_case_ids=case_ids,
            cost_by_case=cost_by_case,
            low_quantile=config.low_cost_quantile,
            high_quantile=config.high_cost_quantile,
        )

    dataset = ActivityCaseSequenceDataset(
        sequences=encoded_sequences,
        cost_levels=cost_levels,
    )
    split = replace(
        split,
        train_dataset=Subset(dataset, split.train_indices),
        validation_dataset=(
            Subset(dataset, split.validation_indices)
            if split.validation_indices
            else None
        ),
    )

    print(f"Train cases: {len(split.train_dataset)}")
    if split.validation_dataset is not None:
        print(f"Validation cases: {len(split.validation_dataset)}")
        print(
            "Validation masking: "
            f"fixed per trace, validation_mask_seed={config.validation_mask_seed}"
        )
        print(
            "Variant split: "
            f"train variants={split.train_variant_count}, "
            f"validation variants={split.validation_variant_count}"
        )
    else:
        print(
            "Variant validation split disabled or unavailable; "
            "early stopping will monitor training loss."
        )

    collate_fn = build_trace_aware_noising_collate_fn(
        pad_token_id=pad_token_id,
        mask_token_id=mask_token_id,
        trace_weight_beta=config.trace_weight_beta,
        trace_noising_intervals=trace_noising_intervals,
        max_len=config.max_len,
        mask_seed=config.mask_seed,
    )

    dataloader = DataLoader(
        split.train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    val_dataloader = None
    if split.validation_dataset is not None:
        fixed_validation_samples = build_fixed_validation_samples(
            validation_dataset=split.validation_dataset,
            mask_token_id=mask_token_id,
            mask_seed=int(config.validation_mask_seed),
            trace_weight_beta=config.trace_weight_beta,
            trace_noising_intervals=trace_noising_intervals,
            max_len=config.max_len,
        )
        val_collate_fn = build_fixed_validation_collate_fn(pad_token_id=pad_token_id)
        val_dataloader = DataLoader(
            fixed_validation_samples,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=val_collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    model = MDMActivityModel(
        vocab_size=len(token_to_id),
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_len=model_max_len,
        pad_token_id=pad_token_id,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    early_stopping_monitor = (
        "validation_avg_weighted_masked_ce_loss"
        if val_dataloader is not None
        else "training_avg_weighted_masked_ce_loss"
    )
    best_monitor_loss = float("inf")
    best_epoch = 0
    best_metrics: dict[str, object] = {}
    best_state_dict = copy_state_dict_to_cpu(model)
    epochs_without_improvement = 0
    stopped_epoch = config.epochs

    for epoch in range(1, config.epochs + 1):
        metrics = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = None
        if val_dataloader is not None:
            val_metrics = evaluate_one_epoch(
                model=model,
                dataloader=val_dataloader,
                device=device,
            )

        monitor_metrics = val_metrics if val_metrics is not None else metrics
        current_monitor_loss = float(monitor_metrics["avg_weighted_masked_ce_loss"])
        improved = current_monitor_loss < best_monitor_loss - config.early_stopping_min_delta
        if improved:
            best_monitor_loss = current_monitor_loss
            best_epoch = epoch
            best_metrics = dict(monitor_metrics)
            best_metrics["monitor"] = early_stopping_monitor
            best_state_dict = copy_state_dict_to_cpu(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        log_message = (
            f"Epoch [{epoch}/{config.epochs}] - "
            f"avg weighted masked CE loss: {metrics['avg_weighted_masked_ce_loss']:.6f} | "
            f"avg masked CE loss: {metrics['avg_masked_ce_loss']:.6f} | "
            f"total masked tokens: {metrics['total_masked_tokens']} | "
            f"avg mask ratio: {metrics['avg_mask_ratio']:.6f}"
        )
        if val_metrics is not None:
            log_message += (
                f" | val weighted masked CE loss: "
                f"{val_metrics['avg_weighted_masked_ce_loss']:.6f} | "
                f"val masked CE loss: {val_metrics['avg_masked_ce_loss']:.6f} | "
                f"best epoch: {best_epoch} | "
                f"epochs without improvement: {epochs_without_improvement}"
            )
        else:
            log_message += (
                " | early stopping monitor: train weighted masked CE loss | "
                f"best epoch: {best_epoch} | "
                f"epochs without improvement: {epochs_without_improvement}"
            )
        print(log_message)

        if (
            config.early_stopping_enabled
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            stopped_epoch = epoch
            print(
                "Early stopping triggered: "
                f"no improvement for {config.early_stopping_patience} epochs. "
                "Best epoch: "
                f"{best_epoch}, best {early_stopping_monitor}: {best_monitor_loss:.6f}"
            )
            break

    if best_epoch > 0:
        model.load_state_dict(best_state_dict)
    save_artifacts(
        save_dir=config.save_path,
        model=model,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        config=config,
        max_len_source=max_len_source,
        max_len_details=max_len_details,
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        early_stopping_monitor=early_stopping_monitor,
        best_monitor_loss=best_monitor_loss,
        stopped_epoch=stopped_epoch,
        alignment_cost_info=alignment_cost_info,
        trace_noising_intervals=trace_noising_intervals,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TN-MDM.")
    parser.add_argument(
        "--event-log",
        default=None,
        help=f"Event log name used for default paths. Default: {TrainConfig.event_log}.",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help=f"Training CSV path. Default: {TrainConfig.csv_path}.",
    )
    parser.add_argument(
        "--alignment-cost-path",
        default=None,
        help=(
            "Trace alignment cost CSV path. Use 'none' to disable. "
            f"Default: {TrainConfig.alignment_cost_path}."
        ),
    )
    parser.add_argument(
        "--mask-seed",
        type=int,
        default=None,
        help=(
            "Seed for TN-MDM training noising. "
            f"Default: {TrainConfig.mask_seed}."
        ),
    )
    parser.add_argument(
        "--validation-mask-seed",
        type=int,
        default=None,
        help=(
            "Seed for fixed validation masking. "
            f"Default: seed + {VALIDATION_MASK_SEED_OFFSET}."
        ),
    )
    parser.add_argument(
        "--trace-weight-beta",
        type=float,
        default=None,
        help=(
            "Trace contribution smoothing beta for TN-MDM view weighting. "
            f"Default: {TrainConfig.trace_weight_beta}."
        ),
    )
    parser.add_argument(
        "--t-min",
        type=float,
        default=None,
        help=f"Minimum non-zero noising ratio used by configurable TN intervals. Default: {TrainConfig.t_min}.",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=None,
        help=f"Maximum noising ratio for TN intervals. Default: {TrainConfig.t_max}.",
    )
    parser.add_argument(
        "--low-start",
        choices=TRACE_START_CHOICES,
        default=None,
        help=f"Start mode for low-cost trace interval. Default: {TrainConfig.low_start}.",
    )
    parser.add_argument(
        "--medium-start",
        choices=TRACE_START_CHOICES,
        default=None,
        help=f"Start mode for first medium-cost trace interval. Default: {TrainConfig.medium_start}.",
    )
    parser.add_argument(
        "--high-start",
        choices=TRACE_START_CHOICES,
        default=None,
        help=f"Start mode for first high-cost trace interval. Default: {TrainConfig.high_start}.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Maximum training epochs. Default: {TrainConfig.epochs}.",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help=f"Directory for model.pt, vocab.json, and config.json. Default: {TrainConfig.save_path}.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    config = TrainConfig()
    if args.event_log is not None:
        config.event_log = args.event_log
        config.csv_path = "data/processed/splits/" + config.event_log + "_variant_train.csv"
        config.alignment_cost_path = (
            "outputs/alignment_costs/" + config.event_log + "_train_alignment_cost.csv"
        )
        config.save_path = "outputs/model_train/" + config.event_log + "/TN-MDM"
    if args.csv_path is not None:
        config.csv_path = args.csv_path
    if args.alignment_cost_path is not None:
        normalized = args.alignment_cost_path.strip().lower()
        config.alignment_cost_path = None if normalized in {"none", "null"} else args.alignment_cost_path
    if args.mask_seed is not None:
        config.mask_seed = args.mask_seed
    if args.validation_mask_seed is not None:
        config.validation_mask_seed = args.validation_mask_seed
    if args.trace_weight_beta is not None:
        config.trace_weight_beta = args.trace_weight_beta
    if args.t_min is not None:
        config.t_min = args.t_min
    if args.t_max is not None:
        config.t_max = args.t_max
    if args.low_start is not None:
        config.low_start = args.low_start
    if args.medium_start is not None:
        config.medium_start = args.medium_start
    if args.high_start is not None:
        config.high_start = args.high_start
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.save_path is not None:
        config.save_path = args.save_path
    config.csv_path = str(ROOT_DIR / config.csv_path)
    config.save_path = str(ROOT_DIR / config.save_path)
    if config.alignment_cost_path is not None:
        config.alignment_cost_path = str(ROOT_DIR / config.alignment_cost_path)
    return config


def main() -> None:
    train(config_from_args(parse_args()))


if __name__ == "__main__":
    main()
