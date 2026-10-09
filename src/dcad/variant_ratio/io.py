"""Data readers matching the experiment preprocessing and evaluation protocol."""
from __future__ import annotations
import csv
import gzip
import json
from collections import OrderedDict
from pathlib import Path
from typing import Sequence, Optional
from torch.utils.data import Dataset
PAD_TOKEN, MASK_TOKEN, UNK_TOKEN = "[PAD]", "[MASK]", "[UNK]"

class ActivitySequenceDataset(Dataset):
    def __init__(self, sequences: Sequence[Sequence[int]]):
        self.sequences = [list(sequence) for sequence in sequences if len(sequence) > 0]
        if not self.sequences:
            raise ValueError("Dataset is empty after filtering empty sequences.")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> list[int]:
        return self.sequences[index]


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


def _open_json_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def load_test_traces(dataset_arg: str) -> list[list[str]]:
    dataset_path = Path(dataset_arg).expanduser().resolve()

    with _open_json_maybe_gzip(dataset_path) as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Dataset file must contain a JSON object: {dataset_path}")

    if "cases" in payload:
        case_key = "cases"
    elif "traces" in payload:
        case_key = "traces"
    else:
        raise ValueError(f"Dataset file does not contain 'cases' or 'traces': {dataset_path}")

    traces: list[list[str]] = []
    for case_index, case in enumerate(payload[case_key]):
        events = case.get("events")
        if not isinstance(events, list):
            raise ValueError(
                f"Case at index {case_index} in {dataset_path} does not contain a valid 'events' list."
            )

        trace: list[str] = []
        for event_index, event in enumerate(events):
            name = event.get("name")
            if name is None:
                raise ValueError(
                    f"Event at case index {case_index}, event index {event_index} in {dataset_path} "
                    "does not contain a 'name' field."
                )
            trace.append(str(name))

        traces.append(trace)

    return traces
