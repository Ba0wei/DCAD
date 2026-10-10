#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train DCAD and EM-DCAD with the experiment implementation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
torch.use_deterministic_algorithms(True)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dcad.model import DCADActivityModel as MDMActivityModel
from dcad.variant_split import split_dataset_by_variant


PAD_TOKEN = "[PAD]"
MASK_TOKEN = "[MASK]"
UNK_TOKEN = "[UNK]"
IGNORE_INDEX = -100

START_CONTEXT = -1
END_CONTEXT = -2
VALIDATION_MASK_SEED_OFFSET = 100000
TRACE_LENGTHS_PATH = ROOT_DIR / "data" / "trace_lengths.json"


@dataclass
class TrainConfig:
    event_log: str = "BPIC12"
    csv_path: str = "data/processed/splits/" + event_log + "_variant_train.csv"
    case_id_col: str = "case_id"
    activity_col: str = "name"
    max_len: Optional[int] = None

    batch_size: int = 32
    epochs: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 2026
    num_workers: int = 0
    t_min: float = 0
    t_max: float = 0.6
    validation_ratio: float = 0.1
    mask_seed: int = 2026
    validation_mask_seed: Optional[int] = None
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 1e-4

    # adaptive_weight=0 gives the plain MDM masking baseline.
    adaptive_weight: float = 0.1
    # adaptive_weight: float = 0
    adaptive_schedule_gamma: float = 3
    rarity_temperature: float = 1
    dfg_smoothing_alpha: float = 1.0

    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1

    save_path: str = "outputs/model_train/" + event_log + "/EM-MDM"


class ActivitySequenceDataset(Dataset):
    def __init__(self, sequences: Sequence[Sequence[int]]):
        self.sequences = [list(sequence) for sequence in sequences if len(sequence) > 0]
        if not self.sequences:
            raise ValueError("Dataset is empty after filtering empty sequences.")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> list[int]:
        return self.sequences[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> None:
    os.makedirs(path, exist_ok=True)


def read_activity_sequences_from_csv(
    csv_path: str,
    case_id_col: str,
    activity_col: str,
) -> list[list[str]]:
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

    sequences = [sequence for sequence in case_to_sequence.values() if sequence]
    if not sequences:
        raise ValueError("No non-empty case sequences were loaded from the CSV.")
    if skipped_rows > 0:
        print(f"Warning: skipped {skipped_rows} rows due to empty/missing case_id or activity.")
    return sequences


def truncate_sequences(sequences: Sequence[Sequence[str]], max_len: Optional[int]) -> list[list[str]]:
    if max_len is None:
        return [list(sequence) for sequence in sequences if sequence]
    return [list(sequence[:max_len]) for sequence in sequences if sequence[:max_len]]


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


def validate_config(config: TrainConfig) -> None:
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    if config.max_len is not None and config.max_len < 1:
        raise ValueError("max_len must be positive when set.")
    if not (
        0.0 <= config.t_min < config.t_max <= 1.0
        and 0.0 <= config.validation_ratio < 1.0
    ):
        raise ValueError("Invalid mask or validation ratio.")
    if not (0.0 <= config.adaptive_weight <= 1.0):
        raise ValueError("adaptive_weight must be in [0, 1].")
    if min(
        config.early_stopping_patience,
        config.adaptive_schedule_gamma,
        config.rarity_temperature,
        config.dfg_smoothing_alpha,
    ) <= 0 or config.early_stopping_min_delta < 0.0:
        raise ValueError("Invalid adaptive or early stopping settings.")


class TransitionRarityMasker:
    """Weighted position sampler based on local directly-follows rarity."""

    def __init__(
        self,
        sequences: Sequence[Sequence[int]],
        adaptive_weight: float,
        temperature: float,
        smoothing_alpha: float,
        t_min: float,
        t_max: float,
        adaptive_schedule_gamma: float,
    ):
        self.adaptive_weight = float(adaptive_weight)
        self.temperature = float(temperature)
        self.smoothing_alpha = float(smoothing_alpha)
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.adaptive_schedule_gamma = float(adaptive_schedule_gamma)
        self.edge_counts: Counter[tuple[int, int]] = Counter()
        self.out_counts: Counter[int] = Counter()
        self.target_context_tokens = {END_CONTEXT}
        self._fit(sequences)

    def _fit(self, sequences: Sequence[Sequence[int]]) -> None:
        for seq in sequences:
            tokens = [START_CONTEXT] + [int(token_id) for token_id in seq] + [END_CONTEXT]
            self.target_context_tokens.update(int(token_id) for token_id in seq)
            for source, target in zip(tokens, tokens[1:]):
                self.edge_counts[(source, target)] += 1
                self.out_counts[source] += 1

    def _edge_surprisal(self, source: int, target: int) -> float:
        alpha = self.smoothing_alpha
        denominator = self.out_counts[source] + alpha * len(self.target_context_tokens)
        probability = (self.edge_counts[(source, target)] + alpha) / denominator
        return -math.log(probability)

    def _position_scores(self, seq: Sequence[int]) -> list[float]:
        scores = []
        for index, token_id in enumerate(seq):
            left = START_CONTEXT if index == 0 else int(seq[index - 1])
            right = END_CONTEXT if index == len(seq) - 1 else int(seq[index + 1])
            score = 0.5 * (
                self._edge_surprisal(left, int(token_id))
                + self._edge_surprisal(int(token_id), right)
            )
            scores.append(score)
        return scores

    def _sampling_probabilities(self, seq: Sequence[int], mask_ratio: float) -> list[float]:
        if self.adaptive_weight == 0.0:
            return [1.0 / len(seq)] * len(seq)

        ratio = (float(mask_ratio) - self.t_min) / max(self.t_max - self.t_min, 1e-8)
        ratio = min(max(ratio, 0.0), 1.0)
        effective_adaptive_weight = self.adaptive_weight * (
            ratio ** self.adaptive_schedule_gamma
        )
        effective_adaptive_weight = min(
            max(effective_adaptive_weight, 0.0),
            self.adaptive_weight,
        )

        scores = self._position_scores(seq)
        scaled_scores = [score / self.temperature for score in scores]
        max_score = max(scaled_scores)
        exp_scores = [math.exp(score - max_score) for score in scaled_scores]
        exp_total = sum(exp_scores)
        adaptive_probs = [score / exp_total for score in exp_scores]
        uniform_prob = 1.0 / len(seq)

        return [
            (1.0 - effective_adaptive_weight) * uniform_prob
            + effective_adaptive_weight * adaptive_prob
            for adaptive_prob in adaptive_probs
        ]

    @staticmethod
    def _weighted_sample_without_replacement(
        probabilities: Sequence[float],
        k: int,
        rng: Optional[random.Random] = None,
    ) -> list[int]:
        rng = rng if rng is not None else random
        available_indices = list(range(len(probabilities)))
        available_weights = [float(value) for value in probabilities]
        selected_indices = []

        for _ in range(k):
            total_weight = sum(available_weights)
            threshold = rng.random() * total_weight
            cumulative = 0.0
            chosen_offset = len(available_indices) - 1
            for offset, weight in enumerate(available_weights):
                cumulative += weight
                if cumulative >= threshold:
                    chosen_offset = offset
                    break

            selected_indices.append(available_indices.pop(chosen_offset))
            available_weights.pop(chosen_offset)

        return selected_indices

    def sample_positions(
        self,
        seq: Sequence[int],
        mask_ratio: float,
        rng: Optional[random.Random] = None,
    ) -> list[int]:
        rng = rng if rng is not None else random
        seq_len = len(seq)
        num_to_mask = max(1, int(round(seq_len * mask_ratio)))
        num_to_mask = min(num_to_mask, seq_len)

        if self.adaptive_weight == 0.0:
            return rng.sample(range(seq_len), k=num_to_mask)

        probabilities = self._sampling_probabilities(seq, mask_ratio=mask_ratio)
        return self._weighted_sample_without_replacement(
            probabilities,
            num_to_mask,
            rng=rng,
        )


def mask_sequence_at_indices(
    token_ids: Sequence[int],
    mask_token_id: int,
    indices: Sequence[int],
    ignore_index: int = IGNORE_INDEX,
) -> tuple[list[int], list[int], float]:
    seq_len = len(token_ids)
    if seq_len == 0:
        return [], [], 0.0

    selected = {int(index) for index in indices}
    actual_mask_ratio = len(selected) / seq_len
    input_ids = []
    labels = []

    for index, token_id in enumerate(token_ids):
        if index in selected:
            input_ids.append(mask_token_id)
            labels.append(int(token_id))
        else:
            input_ids.append(int(token_id))
            labels.append(ignore_index)

    return input_ids, labels, actual_mask_ratio


def compute_mdlm_loss_weight(mask_ratios: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mask_ratios = mask_ratios.clamp(min=eps, max=1.0 - eps)
    sigma = -torch.log1p(-mask_ratios)
    dsigma = 1.0 / (1.0 - mask_ratios)
    return dsigma / torch.expm1(sigma)


def build_adaptive_continuous_t_collate_fn(
    pad_token_id: int,
    mask_token_id: int,
    t_min: float,
    t_max: float,
    masker: TransitionRarityMasker,
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

    def collate_fn(batch: Sequence[Sequence[int]]) -> Dict[str, torch.Tensor]:
        masked_inputs = []
        masked_labels = []
        batch_mask_ratios = []
        rng = get_rng()

        for seq in batch:
            seq = list(seq)
            if max_len is not None:
                seq = seq[:max_len]

            sampled_ratio = t_min + rng.random() * (t_max - t_min)
            selected = masker.sample_positions(seq, sampled_ratio, rng=rng)
            input_ids, labels, actual_mask_ratio = mask_sequence_at_indices(
                token_ids=seq,
                mask_token_id=mask_token_id,
                indices=selected,
                ignore_index=IGNORE_INDEX,
            )

            masked_inputs.append(input_ids)
            masked_labels.append(labels)
            batch_mask_ratios.append(actual_mask_ratio)

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
        }

    return collate_fn


def build_fixed_validation_samples(
    validation_dataset: Dataset,
    mask_token_id: int,
    t_min: float,
    t_max: float,
    masker: TransitionRarityMasker,
    validation_mask_seed: int,
    max_len: Optional[int] = None,
) -> list[dict[str, object]]:
    fixed_samples = []

    for local_index in range(len(validation_dataset)):
        seq = list(validation_dataset[local_index])
        if max_len is not None:
            seq = seq[:max_len]

        rng = random.Random(int(validation_mask_seed) + local_index)
        sampled_ratio = t_min + rng.random() * (t_max - t_min)
        selected = masker.sample_positions(seq, sampled_ratio, rng=rng)
        input_ids, labels, actual_mask_ratio = mask_sequence_at_indices(
            token_ids=seq,
            mask_token_id=mask_token_id,
            indices=selected,
            ignore_index=IGNORE_INDEX,
        )

        fixed_samples.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "mask_ratio": actual_mask_ratio,
            }
        )

    return fixed_samples


def build_fixed_validation_collate_fn(pad_token_id: int):
    def collate_fn(batch: Sequence[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        masked_inputs = [list(sample["input_ids"]) for sample in batch]
        masked_labels = [list(sample["labels"]) for sample in batch]
        batch_mask_ratios = [float(sample["mask_ratio"]) for sample in batch]

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
    total_mask_ratio = 0.0
    total_samples = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        mask_ratios = batch["mask_ratios"].to(device)

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
            masked_ce = ce * valid_mask_float
            weighted_masked_ce = masked_ce * weights
            loss = weighted_masked_ce.sum() / valid_mask_float.sum().clamp_min(1.0)

        masked_count = int(valid_mask.sum().item())
        if masked_count == 0:
            continue

        if is_training:
            loss.backward()
            optimizer.step()

        total_weighted_loss_sum += float(weighted_masked_ce.sum().item())
        total_loss_sum += float(masked_ce.sum().item())
        total_masked_tokens += masked_count
        total_mask_ratio += float(mask_ratios.detach().sum().item())
        total_samples += int(mask_ratios.numel())

    if total_masked_tokens == 0:
        mode = "training" if is_training else "validation"
        raise RuntimeError(f"No masked tokens were generated during {mode}.")

    return {
        "avg_weighted_masked_ce_loss": total_weighted_loss_sum / total_masked_tokens,
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
            "model_type": "Adaptive-MDM",
            "loss_type": "mdlm_continuous_t_weighted_masked_ce",
            "loss_weight": "dsigma_over_expm1_sigma",
            "masking_strategy": "noise_level_aware_dfg_transition_rarity",
            "best_epoch": int(best_epoch),
            "best_metrics": best_metrics,
            "early_stopping_monitor": early_stopping_monitor,
            "best_monitor_loss": (
                float(best_monitor_loss) if math.isfinite(best_monitor_loss) else None
            ),
            "stopped_epoch": int(stopped_epoch),
        }
    )
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, ensure_ascii=False, indent=2)

    print(f"Saved model to: {model_path}")
    print(f"Saved vocab to: {vocab_path}")
    print(f"Saved config to: {config_path}")


def optional_positive_int(value: str) -> Optional[int]:
    normalized = value.strip().lower()
    if normalized in {"none", "null"}:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "max_len must be a positive integer or 'none'."
        ) from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("max_len must be positive when set.")
    return parsed


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
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    raw_sequences = read_activity_sequences_from_csv(
        csv_path=config.csv_path,
        case_id_col=config.case_id_col,
        activity_col=config.activity_col,
    )
    raw_sequences = truncate_sequences(raw_sequences, config.max_len)

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
        "train_csv_path": trace_length_stats.get("train_csv_path", _repo_relative_path(Path(config.csv_path))),
        "train_csv_max_trace_length": train_csv_max_len,
        "train_csv_max_boundary_input_length": train_csv_boundary_max_len,
        "custom_test_dataset": trace_length_stats.get("custom_test_dataset"),
        "custom_test_path": trace_length_stats.get("custom_test_path"),
        "custom_test_max_trace_length": custom_test_max_len,
        "custom_test_max_boundary_input_length": custom_test_boundary_max_len,
    }
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
    print(f"Max sequence length in train CSV: {train_csv_max_len}")
    print(f"Max boundary input length in train CSV: {train_csv_boundary_max_len}")
    if max_len_details["custom_test_max_trace_length"] is not None:
        print(
            "Max sequence length in custom test: "
            f"{max_len_details['custom_test_max_trace_length']}"
        )
        print(
            "Max boundary input length in custom test: "
            f"{max_len_details['custom_test_max_boundary_input_length']}"
        )
    print(f"Model max_len: {model_max_len}")
    print(f"Model max_len source: {max_len_source}")
    print(
        "Continuous t: "
        f"mask_ratio sampled uniformly from [{config.t_min}, {config.t_max}]"
    )
    print(
        "Adaptive masking: "
        "strategy=noise_level_aware_dfg_transition_rarity, "
        f"adaptive_weight={config.adaptive_weight}, "
        f"gamma={config.adaptive_schedule_gamma}, "
        f"temperature={config.rarity_temperature}, "
        f"smoothing_alpha={config.dfg_smoothing_alpha}"
    )
    print(
        "Early stopping: "
        f"enabled={config.early_stopping_enabled}, "
        f"patience={config.early_stopping_patience}, "
        f"min_delta={config.early_stopping_min_delta}"
    )

    dataset = ActivitySequenceDataset(encoded_sequences)
    split = split_dataset_by_variant(
        dataset=dataset,
        sequences=encoded_sequences,
        validation_ratio=config.validation_ratio,
        seed=config.seed,
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

    train_sequences = [encoded_sequences[index] for index in split.train_indices]
    masker = TransitionRarityMasker(
        sequences=train_sequences,
        adaptive_weight=config.adaptive_weight,
        temperature=config.rarity_temperature,
        smoothing_alpha=config.dfg_smoothing_alpha,
        t_min=config.t_min,
        t_max=config.t_max,
        adaptive_schedule_gamma=config.adaptive_schedule_gamma,
    )

    collate_fn = build_adaptive_continuous_t_collate_fn(
        pad_token_id=pad_token_id,
        mask_token_id=mask_token_id,
        t_min=config.t_min,
        t_max=config.t_max,
        masker=masker,
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
            t_min=config.t_min,
            t_max=config.t_max,
            masker=masker,
            validation_mask_seed=int(config.validation_mask_seed),
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
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Adaptive-MDM with noise-level-aware DFG transition-rarity masking."
    )
    parser.add_argument(
        "--event-log",
        default=None,
        help=f"Event log name used for default train CSV paths. Default: {TrainConfig.event_log}.",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help=f"Training CSV path. Default: {TrainConfig.csv_path}.",
    )
    parser.add_argument(
        "--t-min",
        type=float,
        default=None,
        help=f"Minimum continuous mask ratio. Default: {TrainConfig.t_min}.",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=None,
        help=f"Maximum continuous mask ratio. Default: {TrainConfig.t_max}.",
    )
    parser.add_argument(
        "--adaptive-weight",
        type=float,
        default=None,
        help=f"Adaptive rarity/uniform mixture weight. Default: {TrainConfig.adaptive_weight}.",
    )
    parser.add_argument(
        "--rarity-temperature",
        type=float,
        default=None,
        help=f"Softmax temperature for transition-rarity scores. Default: {TrainConfig.rarity_temperature}.",
    )
    parser.add_argument(
        "--adaptive-schedule-gamma",
        type=float,
        default=None,
        help=(
            "Gamma for noise-level-aware adaptive masking. effective_aw = "
            "adaptive_weight * ((mask_ratio - t_min) / (t_max - t_min)) ** gamma. "
            f"Default: {TrainConfig.adaptive_schedule_gamma}."
        ),
    )
    parser.add_argument(
        "--max-len",
        type=optional_positive_int,
        default=None,
        help=(
            "Maximum activity sequence length. Use 'none' to infer it from "
            f"the matching custom_test log. Default: {TrainConfig.max_len}."
        ),
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help=f"Directory for model.pt, vocab.json, and config.json. Default: {TrainConfig.save_path}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Random seed for model initialization, split, and dataloader shuffle. Default: {TrainConfig.seed}.",
    )
    parser.add_argument(
        "--mask-seed",
        type=int,
        default=None,
        help=f"Seed for training-time masking. Default: {TrainConfig.mask_seed}.",
    )
    parser.add_argument(
        "--validation-mask-seed",
        type=int,
        default=None,
        help=f"Seed for fixed validation masking. Default: seed + {VALIDATION_MASK_SEED_OFFSET}.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Maximum training epochs. Default: {TrainConfig.epochs}.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    config = TrainConfig()
    if args.event_log is not None:
        config.event_log = args.event_log
        config.csv_path = (
            "data/processed/splits/"
            + config.event_log
            + "_variant_train.csv"
        )
        config.save_path = (
            "outputs/model_train/"
            + config.event_log
            + "/MDM"
        )
    if args.csv_path is not None:
        config.csv_path = args.csv_path
    if args.t_min is not None:
        config.t_min = args.t_min
    if args.t_max is not None:
        config.t_max = args.t_max
    if args.adaptive_weight is not None:
        config.adaptive_weight = args.adaptive_weight
    if args.rarity_temperature is not None:
        config.rarity_temperature = args.rarity_temperature
    if args.adaptive_schedule_gamma is not None:
        config.adaptive_schedule_gamma = args.adaptive_schedule_gamma
    if args.max_len is not None:
        config.max_len = args.max_len
    if args.save_path is not None:
        config.save_path = args.save_path
    if args.seed is not None:
        config.seed = args.seed
    if args.mask_seed is not None:
        config.mask_seed = args.mask_seed
    if args.validation_mask_seed is not None:
        config.validation_mask_seed = args.validation_mask_seed
    if args.epochs is not None:
        config.epochs = args.epochs
    config.csv_path = str(ROOT_DIR / config.csv_path)
    config.save_path = str(ROOT_DIR / config.save_path)
    return config


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    train(config)


if __name__ == "__main__":
    main()
