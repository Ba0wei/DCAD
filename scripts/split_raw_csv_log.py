"""
CLI usage example:
python3 split_raw_csv_log.py --log-name wide-0.00-1
"""

import argparse
import csv
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "data" / "processed" / "raw"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "splits"
BOUNDARY_MARKERS = {"", "▶", "■", "START", "END"}
ACTIVITY_CANDIDATES = ("concept:name", "name", "activityNameEN", "activityNameNL")


@dataclass
class CaseRecord:
    index: int
    case_id: str
    rows: list[dict[str, str]]
    variant: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split one raw CSV event log by trace variants.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing raw CSV event logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used to save the split CSV logs.",
    )
    parser.add_argument(
        "--log-name",
        required=True,
        help="Log file name to split, for example BPIC15_1.csv or BPIC15_1.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Target ratio of cases assigned to the train split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible splitting.",
    )
    return parser.parse_args()


def choose_activity_column(fieldnames: list[str]) -> str:
    for candidate in ACTIVITY_CANDIDATES:
        if candidate in fieldnames:
            return candidate
    raise ValueError(f"Could not find an activity column in CSV header: {fieldnames}")


def clean_activity(raw_value: str | None) -> str | None:
    value = (raw_value or "").strip()
    if value in BOUNDARY_MARKERS:
        return None
    return value


def resolve_log_path(input_dir: Path, log_name: str) -> Path:
    filename = log_name if log_name.lower().endswith(".csv") else f"{log_name}.csv"
    csv_path = input_dir / filename
    if not csv_path.is_file():
        raise FileNotFoundError(f"Log file not found: {csv_path}")
    return csv_path


def load_cases(csv_path: Path) -> tuple[list[str], list[CaseRecord]]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{csv_path} is missing a header row.")
        if "case_id" not in reader.fieldnames:
            raise ValueError(f"{csv_path} does not contain a 'case_id' column.")

        fieldnames = reader.fieldnames
        activity_column = choose_activity_column(fieldnames)
        cases: list[CaseRecord] = []
        case_index = 0
        current_case_id = None
        current_rows: list[dict[str, str]] = []
        current_variant: list[str] = []

        for row in reader:
            case_id = row["case_id"]
            if case_id != current_case_id:
                if current_case_id is not None:
                    cases.append(
                        CaseRecord(
                            index=case_index,
                            case_id=current_case_id,
                            rows=current_rows,
                            variant=tuple(current_variant),
                        )
                    )
                    case_index += 1
                current_case_id = case_id
                current_rows = []
                current_variant = []

            current_rows.append(row)
            activity = clean_activity(row.get(activity_column))
            if activity is not None:
                current_variant.append(activity)

        if current_case_id is not None:
            cases.append(
                CaseRecord(
                    index=case_index,
                    case_id=current_case_id,
                    rows=current_rows,
                    variant=tuple(current_variant),
                )
            )

    if not cases:
        raise ValueError(f"{csv_path} does not contain any cases.")
    return fieldnames, cases


def count_case_activities(cases: list[CaseRecord]) -> Counter[str]:
    activity_counts: Counter[str] = Counter()
    for case in cases:
        activity_counts.update(case.variant)
    return activity_counts


def can_move_group_without_unknown_activities(
    train_activity_counts: Counter[str],
    group_activity_counts: Counter[str],
) -> bool:
    return all(train_activity_counts[activity] > count for activity, count in group_activity_counts.items())


def split_by_trace_variants(cases: list[CaseRecord], train_ratio: float, rng: random.Random) -> tuple[list[CaseRecord], list[CaseRecord]]:
    variant_groups: dict[tuple[str, ...], list[CaseRecord]] = defaultdict(list)
    for case in cases:
        variant_groups[case.variant].append(case)

    grouped_cases = list(variant_groups.values())
    if len(grouped_cases) < 2:
        raise ValueError("At least two trace variants are required to create a variant test split.")

    rng.shuffle(grouped_cases)

    target_test_cases = max(1, round(len(cases) * (1.0 - train_ratio)))
    train_activity_counts = count_case_activities(cases)
    test_group_ids: set[int] = set()
    test_case_count = 0

    for group in grouped_cases:
        group_activity_counts = count_case_activities(group)
        if not can_move_group_without_unknown_activities(train_activity_counts, group_activity_counts):
            continue

        current_gap = abs(target_test_cases - test_case_count)
        next_case_count = test_case_count + len(group)
        next_gap = abs(target_test_cases - next_case_count)
        should_move = not test_group_ids or next_gap <= current_gap
        if not should_move:
            continue

        train_activity_counts.subtract(group_activity_counts)
        test_group_ids.add(id(group))
        test_case_count = next_case_count

    train_cases = [case for group in grouped_cases if id(group) not in test_group_ids for case in group]
    test_cases = [case for group in grouped_cases if id(group) in test_group_ids for case in group]
    if not test_cases:
        raise ValueError(
            "Could not create a variant test split without introducing unknown test activities. "
            "Every held-out variant would remove at least one activity from train."
        )
    return train_cases, test_cases


def write_cases(output_path: Path, fieldnames: list[str], cases: list[CaseRecord]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_cases = sorted(cases, key=lambda case: case.index)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in ordered_cases:
            writer.writerows(case.rows)


def build_output_paths(output_dir: Path, stem: str) -> dict[str, Path]:
    return {
        "variant_train": output_dir / f"{stem}_variant_train.csv",
        "variant_test": output_dir / f"{stem}_variant_test.csv",
    }


def print_split_summary(name: str, train_cases: list[CaseRecord], test_cases: list[CaseRecord]) -> None:
    total_cases = len(train_cases) + len(test_cases)
    train_variants = {case.variant for case in train_cases}
    test_variants = {case.variant for case in test_cases}
    train_activities = set(count_case_activities(train_cases))
    test_activities = set(count_case_activities(test_cases))
    unknown_test_activities = test_activities - train_activities
    unseen_test_variants = test_variants - train_variants
    shared_variants = test_variants & train_variants
    if unknown_test_activities:
        raise ValueError(
            "Test split contains activities that are unknown to train: "
            f"{sorted(unknown_test_activities)}"
        )
    if shared_variants:
        raise ValueError(f"Test split contains variants that are also present in train: {len(shared_variants)}")

    print(f"{name}:")
    print(f"  train cases: {len(train_cases)} ({len(train_cases) / total_cases:.2%})")
    print(f"  test cases : {len(test_cases)} ({len(test_cases) / total_cases:.2%})")
    print(f"  train variants: {len(train_variants)}")
    print(f"  test variants : {len(test_variants)}")
    print(f"  train activities: {len(train_activities)}")
    print(f"  test activities : {len(test_activities)}")
    print(f"  unknown test activities: {len(unknown_test_activities)}")
    print(f"  unseen test variants   : {len(unseen_test_variants)}")


def main() -> None:
    args = parse_args()
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1.")

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    csv_path = resolve_log_path(input_dir, args.log_name)

    fieldnames, cases = load_cases(csv_path)
    stem = csv_path.stem
    output_paths = build_output_paths(output_dir, stem)

    print(f"Input log: {csv_path}")
    print(f"Output dir: {output_dir}")

    variant_rng = random.Random(args.seed)
    variant_train, variant_test = split_by_trace_variants(cases, args.train_ratio, variant_rng)
    write_cases(output_paths["variant_train"], fieldnames, variant_train)
    write_cases(output_paths["variant_test"], fieldnames, variant_test)
    print_split_summary("Trace variant split", variant_train, variant_test)
    print(f"  saved train: {output_paths['variant_train']}")
    print(f"  saved test : {output_paths['variant_test']}")


if __name__ == "__main__":
    main()
