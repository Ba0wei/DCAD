#!/usr/bin/env python3
"""Build a merged custom test set from normal CSV traces and anomalous JSON traces.

This script no longer performs "match test_csv to full json.gz and extract a
subset". That old subset-matching task has been retired.

The current task is:
1. read a normal test CSV and aggregate it into normal traces in CSV order;
2. read the full anomalous json.gz and extract all anomalous traces;
3. merge the two groups into one json.gz that remains compatible with the
   existing Dataset/EventLog loader used by evaluate_scores.py.

The implementation intentionally reuses the full json.gz schema wherever
possible, so the merged output looks like the source dataset with a replaced
case list rather than a brand-new ad-hoc format.

Example:
python build_custom_testset.py \
  --normal-test-csv BPIC17_variant_test.csv \
  --full-json-gz BPIC17-0.10.json.gz \
  --case-id-col case_id \
  --activity-col concept:name \
  --output-json-gz eventlogs/BPIC17_custom_test.json.gz \
  --output-debug-csv BPIC17s_custom_test_debug.csv

python build_custom_testset.py \
  --normal-test-csv wide_variant_test.csv \
  --full-json-gz wide-0.10.json.gz \
  --case-id-col case_id \
  --activity-col name \
  --output-json-gz eventlogs/wide_custom_test.json.gz \
  --output-debug-csv wide_custom_test_debug.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent

# Centralized schema candidates for easy hand-tuning.
CASE_CONTAINER_CANDIDATES = ("cases", "traces", "logs")
EVENT_CONTAINER_CANDIDATES = ("events", "event_list", "activities")
CASE_ID_CANDIDATES = ("id", "case_id", "concept:name", "IDofConceptCase", "trace_id")
TRACE_LABEL_CANDIDATES = ("label", "labels", "trace_label", "trace_labels", "target", "targets")
EVENT_LABEL_CANDIDATES = ("event_labels", "event_label", "targets", "classes")
EVENT_ACTIVITY_CANDIDATES = ("name", "concept:name", "activity", "activityNameEN", "activityNameNL")
EVENT_TIMESTAMP_CANDIDATES = ("timestamp", "time:timestamp", "dateFinished", "start_time", "end_time")
EVENT_TIMESTAMP_END_CANDIDATES = ("timestamp_end", "dateStop", "end_time")
BOUNDARY_MARKERS = {"", "▶", "■", "START", "END"}
TRACE_LEVEL_TOP_KEYS = {"labels", "trace_labels", "case_labels", "classes"}
SAFE_TRACE_LEVEL_TOP_KEYS = {"labels", "trace_labels", "case_labels"}


@dataclass
class CsvCase:
    order: int
    case_id: str
    case_id_col: str
    rows: list[dict[str, str]]
    fieldnames: list[str]
    activity_col: str
    timestamp_col: str | None

    @property
    def non_boundary_rows(self) -> list[dict[str, str]]:
        return [row for row in self.rows if not is_boundary_row(row, self.activity_col)]

    @property
    def trace_length(self) -> int:
        return len(self.non_boundary_rows)


@dataclass
class SchemaHints:
    case_key: str
    sample_case: dict[str, Any] | None
    sample_event: dict[str, Any] | None
    trace_label_location: tuple[str, str]
    explicit_case_event_label_location: tuple[str, str] | None
    explicit_per_event_label_location: tuple[str, str] | None
    template_case_attribute_keys: list[str]
    template_event_keys: list[str]
    template_event_attribute_keys: list[str]
    global_trace_attribute_defaults: dict[str, Any]
    global_event_attribute_defaults: dict[str, Any]
    top_level_parallel_trace_keys: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a merged test json.gz from normal CSV traces and all anomalous traces in a full json.gz.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--normal-test-csv", required=True, help="Normal test CSV used to construct normal traces.")
    parser.add_argument("--full-json-gz", required=True, help="Full anomalous json.gz dataset.")
    parser.add_argument("--case-id-col", required=True, help="Case id column in the normal test CSV.")
    parser.add_argument("--activity-col", required=True, help="Activity column in the normal test CSV.")
    parser.add_argument(
        "--timestamp-col",
        default=None,
        help="Optional timestamp column in the normal test CSV.",
    )
    parser.add_argument("--output-json-gz", required=True, help="Output merged test json.gz path.")
    parser.add_argument(
        "--output-debug-csv",
        default=None,
        help="Optional debug CSV describing the merged dataset composition.",
    )
    return parser.parse_args()


def clean_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def read_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_gz(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=4, separators=(",", ": "))


def resolve_existing_input_path(path_arg: str, search_dirs: tuple[Path, ...]) -> Path:
    path = Path(path_arg).expanduser()
    candidates = []
    if not path.is_absolute():
        candidates.extend((path, ROOT_DIR / path))
        candidates.extend(search_dir / path for search_dir in search_dirs)
    else:
        candidates.append(path)

    seen: set[Path] = set()
    unique_candidates = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(resolved)
        if resolved.exists():
            return resolved

    searched = ", ".join(str(candidate) for candidate in unique_candidates)
    raise FileNotFoundError(f"Could not find input file '{path_arg}'. Searched: {searched}")


def is_boundary_marker(value: Any) -> bool:
    return clean_value(value) in BOUNDARY_MARKERS


def is_boundary_row(row: dict[str, str], activity_col: str) -> bool:
    activity_value = clean_value(row.get(activity_col))
    name_value = clean_value(row.get("name"))
    if activity_value and not is_boundary_marker(activity_value):
        return False
    if name_value and not is_boundary_marker(name_value):
        return False
    return True


def infer_case_container_key(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise TypeError(
            "The merged output is designed for dict-style event logs with top-level attributes and cases/traces."
        )

    for candidate in CASE_CONTAINER_CANDIDATES:
        value = payload.get(candidate)
        if isinstance(value, list):
            return candidate

    list_keys = []
    for key, value in payload.items():
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            if any(any(event_key in item for event_key in EVENT_CONTAINER_CANDIDATES) for item in value[:3]):
                list_keys.append(key)
    if len(list_keys) == 1:
        return list_keys[0]
    if len(list_keys) > 1:
        raise ValueError(
            f"Could not infer the case container key unambiguously. Candidate keys: {list_keys}. "
            "Please adjust CASE_CONTAINER_CANDIDATES."
        )
    raise ValueError("Could not infer the case container key from the json.gz payload.")


def get_cases_from_payload(payload: Any, case_key: str) -> list[dict[str, Any]]:
    value = payload.get(case_key)
    if not isinstance(value, list):
        raise ValueError(f"Case container '{case_key}' is missing or is not a list.")
    return value


def get_case_attributes(case: dict[str, Any]) -> dict[str, Any]:
    attributes = case.get("attributes")
    if isinstance(attributes, dict):
        return attributes
    return {}


def get_case_events(case: dict[str, Any]) -> list[dict[str, Any]]:
    for key in EVENT_CONTAINER_CANDIDATES:
        value = case.get(key)
        if isinstance(value, list):
            return value
    return []


def get_event_attributes(event: dict[str, Any]) -> dict[str, Any]:
    attributes = event.get("attributes")
    if isinstance(attributes, dict):
        return attributes
    return {}


def extract_case_field(case: dict[str, Any], preferred_field: str | None = None) -> str | None:
    attributes = get_case_attributes(case)
    candidates = []
    if preferred_field:
        candidates.append(preferred_field)
    candidates.extend(CASE_ID_CANDIDATES)
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate in case:
            value = clean_value(case.get(candidate))
            if value:
                return value
        if candidate in attributes:
            value = clean_value(attributes.get(candidate))
            if value:
                return value
    return None


def extract_event_field_with_fallback(
    event: dict[str, Any],
    preferred_field: str | None = None,
    fallbacks: tuple[str, ...] = (),
) -> str:
    attributes = get_event_attributes(event)
    candidates = []
    if preferred_field:
        candidates.append(preferred_field)
    candidates.extend(fallbacks)
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate in event:
            value = clean_value(event.get(candidate))
            if value:
                return value
        if candidate in attributes:
            value = clean_value(attributes.get(candidate))
            if value:
                return value
    return ""


def extract_csv_field_with_fallback(
    row: dict[str, str],
    preferred_field: str | None = None,
    fallbacks: tuple[str, ...] = (),
) -> str:
    candidates = []
    if preferred_field:
        candidates.append(preferred_field)
    candidates.extend(fallbacks)
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        value = clean_value(row.get(candidate))
        if value:
            return value
    return ""


def extract_declared_defaults(section: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    if not isinstance(section, dict):
        return defaults
    for key, metadata in section.items():
        if isinstance(metadata, dict) and "value" in metadata:
            defaults[key] = metadata["value"]
        else:
            defaults[key] = ""
    return defaults


def find_trace_label_location(case: dict[str, Any]) -> tuple[str, str]:
    case_attributes = get_case_attributes(case)
    for key in TRACE_LABEL_CANDIDATES:
        if key in case_attributes:
            return ("case_attributes", key)
    for key in TRACE_LABEL_CANDIDATES:
        if key in case:
            return ("case", key)
    return ("case_attributes", "label")


def find_explicit_case_event_label_location(cases: list[dict[str, Any]]) -> tuple[str, str] | None:
    for case in cases[:10]:
        events = get_case_events(case)
        if not events:
            continue
        case_attributes = get_case_attributes(case)
        for key in EVENT_LABEL_CANDIDATES:
            value = case.get(key)
            if isinstance(value, list) and len(value) == len(events):
                return ("case", key)
        for key in EVENT_LABEL_CANDIDATES:
            value = case_attributes.get(key)
            if isinstance(value, list) and len(value) == len(events):
                return ("case_attributes", key)
    return None


def find_explicit_per_event_label_location(cases: list[dict[str, Any]]) -> tuple[str, str] | None:
    for case in cases[:10]:
        events = get_case_events(case)
        for event in events[:10]:
            event_attributes = get_event_attributes(event)
            for key in EVENT_LABEL_CANDIDATES:
                if key in event:
                    return ("event", key)
            for key in EVENT_LABEL_CANDIDATES:
                if key in event_attributes:
                    return ("event_attributes", key)
    return None


def build_schema_hints(payload: dict[str, Any], case_key: str) -> SchemaHints:
    cases = get_cases_from_payload(payload, case_key)
    if not cases:
        raise ValueError("The full json.gz does not contain any cases.")

    sample_case = cases[0]
    sample_events = get_case_events(sample_case)
    sample_event = sample_events[0] if sample_events else None
    top_attributes = payload.get("attributes") if isinstance(payload, dict) else {}
    if not isinstance(top_attributes, dict):
        top_attributes = {}
    global_attributes = top_attributes.get("global_attributes")
    if not isinstance(global_attributes, dict):
        global_attributes = {}

    top_level_parallel_trace_keys = []
    for key, value in payload.items():
        if key == case_key:
            continue
        if key in TRACE_LEVEL_TOP_KEYS and isinstance(value, list) and len(value) == len(cases):
            top_level_parallel_trace_keys.append(key)

    return SchemaHints(
        case_key=case_key,
        sample_case=sample_case,
        sample_event=sample_event,
        trace_label_location=find_trace_label_location(sample_case),
        explicit_case_event_label_location=find_explicit_case_event_label_location(cases),
        explicit_per_event_label_location=find_explicit_per_event_label_location(cases),
        template_case_attribute_keys=list(get_case_attributes(sample_case).keys()),
        template_event_keys=list(sample_event.keys()) if isinstance(sample_event, dict) else ["name", "timestamp", "attributes"],
        template_event_attribute_keys=list(get_event_attributes(sample_event).keys()) if isinstance(sample_event, dict) else [],
        global_trace_attribute_defaults=extract_declared_defaults(global_attributes.get("trace")),
        global_event_attribute_defaults=extract_declared_defaults(global_attributes.get("event")),
        top_level_parallel_trace_keys=top_level_parallel_trace_keys,
    )


def discover_json_structure(payload: Any, case_key: str, hints: SchemaHints) -> None:
    print("=== JSON structure inspection ===")
    print(f"Top-level type: {type(payload).__name__}")
    if isinstance(payload, dict):
        print(f"Top-level keys: {list(payload.keys())}")
    else:
        print("Top-level keys: <not a dict>")

    cases = get_cases_from_payload(payload, case_key)
    print(f"Case container key: {case_key}")
    print(f"Number of cases: {len(cases)}")
    print(f"Trace label location: {hints.trace_label_location}")
    print(f"Explicit case-level event labels: {hints.explicit_case_event_label_location}")
    print(f"Explicit per-event labels: {hints.explicit_per_event_label_location}")
    print(f"Top-level trace arrays: {hints.top_level_parallel_trace_keys}")

    if not cases:
        print("No cases found in the full json.gz payload.")
        print("=== End of inspection ===")
        return

    sample_case = hints.sample_case or {}
    sample_case_attrs = get_case_attributes(sample_case)
    sample_events = get_case_events(sample_case)
    print(f"Sample case keys: {list(sample_case.keys())}")
    print(f"Sample case attribute keys: {list(sample_case_attrs.keys())}")
    print(f"Sample case id: {extract_case_field(sample_case)}")
    print(f"Sample case event count: {len(sample_events)}")

    if sample_events:
        sample_event = sample_events[0]
        sample_event_attrs = get_event_attributes(sample_event)
        print(f"Sample event keys: {list(sample_event.keys())}")
        print(f"Sample event attribute keys: {list(sample_event_attrs.keys())}")
        print(
            "Event activity field candidates found: "
            f"{[k for k in EVENT_ACTIVITY_CANDIDATES if k in sample_event or k in sample_event_attrs]}"
        )
        print(
            "Event timestamp field candidates found: "
            f"{[k for k in EVENT_TIMESTAMP_CANDIDATES if k in sample_event or k in sample_event_attrs]}"
        )
    print("=== End of inspection ===")


def extract_trace_label(case: dict[str, Any], hints: SchemaHints) -> Any:
    location, key = hints.trace_label_location
    if location == "case":
        return case.get(key)
    return get_case_attributes(case).get(key)


def set_trace_label(case: dict[str, Any], hints: SchemaHints, label: Any) -> None:
    location, key = hints.trace_label_location
    if location == "case":
        case[key] = label
        return
    case.setdefault("attributes", {})
    case["attributes"][key] = label


def is_trace_anomalous(label: Any) -> bool:
    """Central anomaly decision point.

    Default rule:
    - "normal", 0, "0", False, None => normal
    - dict-like labels or other non-zero / non-normal values => anomalous
    """
    if label is None:
        return False
    if isinstance(label, bool):
        return label
    if isinstance(label, (int, float)):
        return label != 0
    if isinstance(label, str):
        lowered = label.strip().lower()
        return lowered not in {"", "0", "normal"}
    if isinstance(label, dict):
        anomaly_name = clean_value(label.get("anomaly"))
        return anomaly_name.lower() not in {"", "0", "normal"}
    return True


def build_normal_trace_label() -> str:
    return "normal"


def build_normal_event_labels(length: int) -> list[int]:
    return [0] * length


def load_normal_csv_cases(
    csv_path: Path,
    case_id_col: str,
    activity_col: str,
    timestamp_col: str | None,
) -> list[CsvCase]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{csv_path} is missing a header row.")
        fieldnames = list(reader.fieldnames)
        missing_cols = [col for col in (case_id_col, activity_col) if col not in fieldnames]
        if timestamp_col and timestamp_col not in fieldnames:
            missing_cols.append(timestamp_col)
        if missing_cols:
            raise ValueError(f"{csv_path} is missing required columns: {missing_cols}")

        cases: list[CsvCase] = []
        current_case_id: str | None = None
        current_rows: list[dict[str, str]] = []
        closed_case_ids: set[str] = set()

        for row_number, row in enumerate(reader, start=2):
            case_id = clean_value(row.get(case_id_col))
            if not case_id:
                raise ValueError(f"Row {row_number} in {csv_path} has an empty '{case_id_col}'.")

            if current_case_id is None:
                current_case_id = case_id
            elif case_id != current_case_id:
                closed_case_ids.add(current_case_id)
                if case_id in closed_case_ids:
                    raise ValueError(
                        f"Case id '{case_id}' appears in multiple non-contiguous blocks in {csv_path}. "
                        "Please keep the normal CSV grouped by case."
                    )
                cases.append(
                    CsvCase(
                        order=len(cases),
                        case_id=current_case_id,
                        case_id_col=case_id_col,
                        rows=current_rows,
                        fieldnames=fieldnames,
                        activity_col=activity_col,
                        timestamp_col=timestamp_col,
                    )
                )
                current_case_id = case_id
                current_rows = []

            current_rows.append(row)

        if current_case_id is not None:
            cases.append(
                CsvCase(
                    order=len(cases),
                    case_id=current_case_id,
                    case_id_col=case_id_col,
                    rows=current_rows,
                    fieldnames=fieldnames,
                    activity_col=activity_col,
                    timestamp_col=timestamp_col,
                )
            )

    if not cases:
        raise ValueError(f"{csv_path} does not contain any cases.")
    return cases


def build_normal_case_attributes(case_id: str, hints: SchemaHints) -> dict[str, Any]:
    attributes = {"label": build_normal_trace_label()}
    for key in hints.template_case_attribute_keys:
        if key in TRACE_LABEL_CANDIDATES:
            continue
        if key in {"concept:name", "IDofConceptCase", "case_id", "trace_id"}:
            attributes[key] = case_id
    return attributes


def choose_event_name(row: dict[str, str], activity_col: str) -> str:
    name_value = clean_value(row.get("name"))
    if name_value and not is_boundary_marker(name_value):
        return name_value
    return extract_csv_field_with_fallback(row, activity_col, EVENT_ACTIVITY_CANDIDATES)


def choose_event_timestamp(row: dict[str, str], timestamp_col: str | None) -> str:
    return extract_csv_field_with_fallback(row, timestamp_col, EVENT_TIMESTAMP_CANDIDATES)


def choose_event_timestamp_end(row: dict[str, str]) -> str:
    return extract_csv_field_with_fallback(row, "timestamp_end", EVENT_TIMESTAMP_END_CANDIDATES)


def build_normal_event_attributes(
    row: dict[str, str],
    case_id: str,
    event_name: str,
    event_timestamp: str,
    hints: SchemaHints,
) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    template_keys = hints.template_event_attribute_keys or list(hints.global_event_attribute_defaults.keys())

    for key in template_keys:
        if hints.explicit_per_event_label_location == ("event_attributes", key):
            continue
        if key in row:
            attributes[key] = clean_value(row.get(key))
            continue
        if key == "concept:name":
            attributes[key] = extract_csv_field_with_fallback(row, "concept:name", (event_name,))
            continue
        if key == "time:timestamp":
            attributes[key] = event_timestamp
            continue
        if key in {"case_id", "trace_id", "IDofConceptCase"}:
            attributes[key] = case_id
            continue
        if key in hints.global_event_attribute_defaults:
            attributes[key] = hints.global_event_attribute_defaults[key]
            continue
        attributes[key] = ""

    return attributes


def maybe_attach_explicit_event_labels(case: dict[str, Any], hints: SchemaHints) -> None:
    events = get_case_events(case)
    zeros = build_normal_event_labels(len(events))

    if hints.explicit_case_event_label_location is not None:
        location, key = hints.explicit_case_event_label_location
        if location == "case":
            case[key] = zeros
        else:
            case.setdefault("attributes", {})
            case["attributes"][key] = zeros

    if hints.explicit_per_event_label_location is not None:
        location, key = hints.explicit_per_event_label_location
        for event in events:
            if location == "event":
                event[key] = 0
            else:
                event.setdefault("attributes", {})
                event["attributes"][key] = 0


def build_normal_case(csv_case: CsvCase, hints: SchemaHints) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    template_event_keys = hints.template_event_keys or ["name", "timestamp", "attributes"]
    include_timestamp_end = "timestamp_end" in template_event_keys

    for row in csv_case.non_boundary_rows:
        event_name = choose_event_name(row, csv_case.activity_col)
        if not event_name:
            raise ValueError(
                f"Could not determine event name for case '{csv_case.case_id}'. "
                "Please check --activity-col and CSV headers."
            )

        event_timestamp = choose_event_timestamp(row, csv_case.timestamp_col)
        event: dict[str, Any] = {
            "name": event_name,
            "timestamp": event_timestamp,
            "attributes": build_normal_event_attributes(row, csv_case.case_id, event_name, event_timestamp, hints),
        }
        if include_timestamp_end:
            event["timestamp_end"] = choose_event_timestamp_end(row)
        events.append(event)

    case = {
        "id": csv_case.case_id,
        "attributes": build_normal_case_attributes(csv_case.case_id, hints),
        hints.case_key[:-1] if False else "events": events,
    }
    maybe_attach_explicit_event_labels(case, hints)
    set_trace_label(case, hints, build_normal_trace_label())
    return case


def extract_anomalous_cases(payload: dict[str, Any], hints: SchemaHints) -> list[tuple[int, dict[str, Any]]]:
    anomalous_cases: list[tuple[int, dict[str, Any]]] = []
    for index, case in enumerate(get_cases_from_payload(payload, hints.case_key)):
        label = extract_trace_label(case, hints)
        if is_trace_anomalous(label):
            anomalous_cases.append((index, copy.deepcopy(case)))
    return anomalous_cases


def build_normal_top_level_label_entry(example_value: Any) -> Any:
    if isinstance(example_value, str):
        return build_normal_trace_label()
    if isinstance(example_value, (int, float, bool)) or example_value is None:
        return 0
    return build_normal_trace_label()


def build_merged_payload(
    payload: dict[str, Any],
    hints: SchemaHints,
    normal_cases: list[dict[str, Any]],
    anomalous_case_records: list[tuple[int, dict[str, Any]]],
) -> dict[str, Any]:
    merged = copy.deepcopy(payload)
    merged_cases = normal_cases + [case for _, case in anomalous_case_records]
    merged[hints.case_key] = merged_cases

    original_cases = get_cases_from_payload(payload, hints.case_key)
    original_case_count = len(original_cases)
    anomaly_indices = [index for index, _ in anomalous_case_records]

    for key in hints.top_level_parallel_trace_keys:
        value = merged.get(key)
        if not isinstance(value, list) or len(value) != original_case_count:
            continue
        if key in SAFE_TRACE_LEVEL_TOP_KEYS:
            example_value = value[0] if value else build_normal_trace_label()
            merged[key] = [build_normal_top_level_label_entry(example_value) for _ in normal_cases] + [
                copy.deepcopy(payload[key][index]) for index in anomaly_indices
            ]
        else:
            warnings.warn(
                f"Top-level key '{key}' matches the original case count but cannot be safely rebuilt. "
                "It was removed from the merged payload. Adjust TRACE_LEVEL_TOP_KEYS if needed.",
                RuntimeWarning,
            )
            del merged[key]

    return merged


def try_import_label_to_targets():
    try:
        from dcad.anomaly import label_to_targets  # pylint: disable=import-outside-toplevel

        return label_to_targets
    except Exception as exc:  # pragma: no cover - depends on runtime env
        warnings.warn(
            f"Could not import utils.anomaly.label_to_targets, derived event-label counting will be skipped: {exc}",
            RuntimeWarning,
        )
        return None


def infer_num_event_attributes(payload: dict[str, Any], hints: SchemaHints) -> int:
    top_attributes = payload.get("attributes")
    if isinstance(top_attributes, dict):
        global_attributes = top_attributes.get("global_attributes")
        if isinstance(global_attributes, dict):
            event_attrs = global_attributes.get("event")
            if isinstance(event_attrs, dict):
                return 1 + len(event_attrs)

    if hints.template_event_attribute_keys:
        return 1 + len(hints.template_event_attribute_keys)
    return 1


def label_value_is_anomalous(value: Any) -> bool:
    if isinstance(value, list):
        return any(label_value_is_anomalous(item) for item in value)
    if isinstance(value, dict):
        return True
    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered not in {"", "0", "normal"}
    if isinstance(value, (int, float, bool)):
        return value != 0
    return value is not None


def count_explicit_event_anomalies(case: dict[str, Any], hints: SchemaHints) -> int | None:
    events = get_case_events(case)

    if hints.explicit_case_event_label_location is not None:
        location, key = hints.explicit_case_event_label_location
        if location == "case":
            values = case.get(key)
        else:
            values = get_case_attributes(case).get(key)
        if isinstance(values, list):
            return sum(1 for value in values if label_value_is_anomalous(value))

    if hints.explicit_per_event_label_location is not None:
        location, key = hints.explicit_per_event_label_location
        count = 0
        for event in events:
            value = event.get(key) if location == "event" else get_event_attributes(event).get(key)
            if label_value_is_anomalous(value):
                count += 1
        return count

    return None


def derive_anomalous_event_count(
    case: dict[str, Any],
    hints: SchemaHints,
    label_to_targets: Any,
    num_event_attributes: int,
) -> int | None:
    explicit_count = count_explicit_event_anomalies(case, hints)
    if explicit_count is not None:
        return explicit_count

    if label_to_targets is None:
        return None

    label = extract_trace_label(case, hints)
    events = get_case_events(case)
    try:
        targets = label_to_targets(label, len(events) + 2, num_event_attributes)
    except Exception as exc:  # pragma: no cover - depends on external label implementation
        warnings.warn(
            f"Could not derive event-level targets for case '{extract_case_field(case) or ''}': {exc}",
            RuntimeWarning,
        )
        return None

    count = 0
    for row in targets[1 : len(events) + 1]:
        if len(row) > 0 and row[0] > 0:
            count += 1
    return count


def format_trace_label_for_debug(label: Any) -> str:
    if label == "normal":
        return "0"
    if label == 0:
        return "0"
    if isinstance(label, (dict, list)):
        return json.dumps(label, ensure_ascii=False, sort_keys=True)
    return clean_value(label)


def write_debug_csv(
    output_path: Path,
    normal_cases: list[dict[str, Any]],
    anomalous_cases: list[dict[str, Any]],
    hints: SchemaHints,
    payload: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_index",
        "source",
        "case_id",
        "trace_length",
        "trace_label",
        "anomalous_event_count",
    ]
    label_to_targets = try_import_label_to_targets()
    num_event_attributes = infer_num_event_attributes(payload, hints)

    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        sample_index = 0
        for case in normal_cases:
            writer.writerow(
                {
                    "sample_index": sample_index,
                    "source": "normal_csv",
                    "case_id": extract_case_field(case) or "",
                    "trace_length": len(get_case_events(case)),
                    "trace_label": "0",
                    "anomalous_event_count": 0,
                }
            )
            sample_index += 1

        for case in anomalous_cases:
            label = extract_trace_label(case, hints)
            anomalous_event_count = derive_anomalous_event_count(case, hints, label_to_targets, num_event_attributes)
            writer.writerow(
                {
                    "sample_index": sample_index,
                    "source": "anomalous_json",
                    "case_id": extract_case_field(case) or "",
                    "trace_length": len(get_case_events(case)),
                    "trace_label": format_trace_label_for_debug(label),
                    "anomalous_event_count": "" if anomalous_event_count is None else anomalous_event_count,
                }
            )
            sample_index += 1


def invalidate_output_cache(output_path: Path) -> None:
    try:
        from dcad.fs import EventLogFile  # pylint: disable=import-outside-toplevel

        cache_file = EventLogFile(output_path).cache_file
        if cache_file.exists():
            cache_file.unlink()
            print(f"Removed stale dataset cache: {cache_file}")
    except Exception as exc:  # pragma: no cover - best effort only
        warnings.warn(f"Could not invalidate dataset cache automatically: {exc}", RuntimeWarning)


def warn_if_no_explicit_event_labels(hints: SchemaHints) -> None:
    if hints.explicit_case_event_label_location is None and hints.explicit_per_event_label_location is None:
        warnings.warn(
            "The full json.gz does not expose explicit event-level label fields. "
            "Event-level targets will continue to rely on Dataset/utils.anomaly.label_to_targets fallback logic.",
            RuntimeWarning,
        )


def validate_case_shapes(normal_cases: list[dict[str, Any]], anomalous_cases: list[dict[str, Any]]) -> None:
    for case in normal_cases + anomalous_cases:
        if "id" not in case:
            raise ValueError("Every output case must contain an 'id' field.")
        if "attributes" not in case or not isinstance(case["attributes"], dict):
            raise ValueError("Every output case must contain an 'attributes' dict.")
        events = get_case_events(case)
        if not isinstance(events, list):
            raise ValueError("Every output case must contain an 'events' list.")
        for event in events:
            if "name" not in event:
                raise ValueError("Every output event must contain 'name'.")
            if "timestamp" not in event:
                raise ValueError("Every output event must contain 'timestamp'.")
            if "attributes" not in event or not isinstance(event["attributes"], dict):
                raise ValueError("Every output event must contain an 'attributes' dict.")


def print_summary(normal_count: int, anomaly_count: int, output_json_gz: Path) -> None:
    print("=== Summary ===")
    print(f"Normal traces         : {normal_count}")
    print(f"Anomalous traces      : {anomaly_count}")
    print(f"Total traces          : {normal_count + anomaly_count}")
    print(f"Output json.gz        : {output_json_gz}")
    print("")
    print("Example command:")
    print(
        "python build_custom_testset.py "
        "--normal-test-csv normal_test.csv "
        "--full-json-gz BPIC15_1-0.10.json.gz "
        "--case-id-col case_id "
        "--activity-col concept:name "
        "--output-json-gz BPIC15_1_custom_test.json.gz "
        "--output-debug-csv BPIC15_1_custom_test_debug.csv"
    )


def main() -> None:
    args = parse_args()

    normal_test_csv = resolve_existing_input_path(
        args.normal_test_csv,
        (
            ROOT_DIR / "csv_logs" / "splits",
            ROOT_DIR / "csv_logs" / "raw",
            ROOT_DIR / "csv_logs" / "custom_testsets",
        ),
    )
    full_json_gz = resolve_existing_input_path(args.full_json_gz, (ROOT_DIR / "eventlogs",))
    output_json_gz = Path(args.output_json_gz).expanduser().resolve()
    output_debug_csv = Path(args.output_debug_csv).expanduser().resolve() if args.output_debug_csv else None

    payload = read_json_gz(full_json_gz)
    case_key = infer_case_container_key(payload)
    hints = build_schema_hints(payload, case_key)
    discover_json_structure(payload, case_key, hints)
    warn_if_no_explicit_event_labels(hints)

    csv_cases = load_normal_csv_cases(
        normal_test_csv,
        case_id_col=args.case_id_col,
        activity_col=args.activity_col,
        timestamp_col=args.timestamp_col,
    )
    normal_cases = [build_normal_case(csv_case, hints) for csv_case in csv_cases]

    anomalous_case_records = extract_anomalous_cases(payload, hints)
    anomalous_cases = [case for _, case in anomalous_case_records]

    validate_case_shapes(normal_cases, anomalous_cases)

    merged_payload = build_merged_payload(payload, hints, normal_cases, anomalous_case_records)
    write_json_gz(output_json_gz, merged_payload)
    invalidate_output_cache(output_json_gz)

    if output_debug_csv is not None:
        write_debug_csv(output_debug_csv, normal_cases, anomalous_cases, hints, payload)
        print(f"Debug CSV written to: {output_debug_csv}")

    print_summary(len(normal_cases), len(anomalous_cases), output_json_gz)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
