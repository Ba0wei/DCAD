#!/usr/bin/env python3
"""Generate normal-only BPIC splits with controlled variant coverage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("BPIC12", *(f"BPIC15_{i}" for i in range(1, 6)), "BPIC17")
RATIOS = (0.9, 0.8, 0.6, 0.4, 0.2)
ALGORITHM_VERSION = "trace_order_v1"
BOUNDARY_MARKERS = {"", "▶", "■", "START", "END"}
SUMMARY_FIELDS = (
    "dataset", "base_seed", "repeat", "seed", "algorithm_version",
    "requested_observed_ratio", "actual_observed_ratio",
    "actual_unseen_ratio", "total_variants", "observed_variants", "unseen_variants",
    "total_cases", "train_cases", "test_cases", "train_case_ratio", "test_case_ratio",
    "train_events", "test_events", "train_csv_rows", "test_csv_rows",
    "train_activities", "test_activities", "unknown_test_activities",
    "shared_variants", "activity_cover_variants", "input_sha256", "split_directory",
    "validation_ratio", "validation_seed", "max_len", "gradient_train_cases",
    "gradient_train_variants", "gradient_train_processed_variants", "gradient_train_activities",
    "gradient_train_variant_coverage", "validation_cases", "validation_variants",
    "validation_processed_variants", "validation_activities",
    "unknown_normal_test_activities_after_validation", "unknown_validation_activities",
)


@dataclass(frozen=True)
class AuditConfig:
    validation_ratio: float = 0.1
    seed: int = 2026
    max_len: int | None = None


@dataclass
class DiscoveryOrder:
    variants: list[tuple[str, ...]]
    startup_case_ids: list[str]
    trace_case_ids: list[str]


def derive_seed(dataset: str, base_seed: int, repeat: int) -> int:
    payload = f"{ALGORITHM_VERSION}|{dataset}|{base_seed}|{repeat}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


@dataclass(frozen=True)
class Case:
    case_id: str
    sequence: tuple[str, ...]
    row_count: int
    row_sha256: str


@dataclass
class Log:
    path: Path
    fieldnames: list[str]
    cases: list[Case]
    variants: dict[tuple[str, ...], list[str]]
    input_sha256: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_cases(path: Path) -> tuple[list[str], list[Case]]:
    """Read only compact case records; digest all original column values in order."""
    cases: list[Case] = []
    closed: set[str] = set()
    current_id: str | None = None
    sequence: list[str] = []
    row_count = 0
    digest = hashlib.sha256()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        fields = next(reader, [])
        if len(set(fields)) != len(fields) or not {"case_id", "name"} <= set(fields):
            raise ValueError(f"{path}: unique columns including 'case_id' and 'name' are required.")
        case_col, name_col = fields.index("case_id"), fields.index("name")
        for row in reader:
            if len(row) != len(fields):
                raise ValueError(f"{path}:{reader.line_num}: unexpected number of CSV fields.")
            case_id = row[case_col]
            if not case_id.strip():
                raise ValueError(f"{path}:{reader.line_num}: empty case_id.")
            if case_id != current_id:
                if current_id is not None:
                    cases.append(Case(current_id, tuple(sequence), row_count, digest.hexdigest()))
                    closed.add(current_id)
                if case_id in closed:
                    raise ValueError(f"{path}: case {case_id!r} is not contiguous in the CSV.")
                current_id, sequence, row_count = case_id, [], 0
                digest = hashlib.sha256()
            # JSON encodes row boundaries and field boundaries without ambiguity.
            digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")
            row_count += 1
            activity = row[name_col].strip()
            if activity not in BOUNDARY_MARKERS:
                sequence.append(activity)
    if current_id is not None:
        cases.append(Case(current_id, tuple(sequence), row_count, digest.hexdigest()))
    if not cases:
        raise ValueError(f"{path}: no cases.")
    if any(not case.sequence for case in cases):
        raise ValueError(f"{path}: a case has no activities after boundary removal.")
    if len({case.case_id.strip() for case in cases}) != len(cases):
        raise ValueError(f"{path}: case IDs collide after the model reader strips whitespace.")
    return fields, cases


def load_log(path: Path) -> Log:
    digest = file_sha256(path)
    fields, cases = read_cases(path)
    variants: dict[tuple[str, ...], list[str]] = {}
    for case in cases:
        variants.setdefault(case.sequence, []).append(case.case_id)
    return Log(path.resolve(), fields, cases, variants, digest)


def ratio_directory(ratio: float) -> str:
    percentage = format(Decimal(str(ratio)) * 100, "f")
    if "." in percentage:
        percentage = percentage.rstrip("0").rstrip(".")
    return "observed_" + percentage.replace(".", "p")


def variant_order(log: Log, seed: int, ratios: list[float]) -> DiscoveryOrder:
    """Greedy startup traces, shuffled remaining traces, then variant discovery."""
    variants = list(log.variants)
    budgets = [round(ratio * len(variants)) for ratio in ratios]
    if any(budget < 1 or budget >= len(variants) for budget in budgets):
        raise ValueError("Rounded variant budgets must leave at least one variant in each partition.")
    rng = random.Random(seed)
    activities = {variant: set(variant) for variant in variants}
    uncovered = set().union(*activities.values())
    remaining = list(log.cases)
    rng.shuffle(remaining)
    cover = []
    while uncovered:
        # max returns the first maximum: the seeded order breaks ties reproducibly.
        chosen = max(remaining, key=lambda case: len(activities[case.sequence] & uncovered))
        cover.append(chosen)
        uncovered.difference_update(activities[chosen.sequence])
        remaining.remove(chosen)
    cover_variants = {case.sequence for case in cover}
    if len(cover_variants) > min(budgets):
        raise ValueError(
            f"{log.path.stem}, seed={seed}: greedy activity cover needs {len(cover)} variants, "
            f"but the smallest budget is {min(budgets)}. No split written for this seed. "
            "This is a construction failure, not a proof that no smaller cover exists."
        )
    rng.shuffle(remaining)
    traces = cover + remaining
    return DiscoveryOrder(list(dict.fromkeys(case.sequence for case in traces)),
                          [case.case_id for case in cover], [case.case_id for case in traces])


def split_cases(log: Log, observed: set[tuple[str, ...]]) -> tuple[list[Case], list[Case]]:
    return ([case for case in log.cases if case.sequence in observed],
            [case for case in log.cases if case.sequence not in observed])


def statistics(log: Log, observed: set[tuple[str, ...]], ratio: float) -> dict:
    train, test = split_cases(log, observed)
    train_variants = {case.sequence for case in train}
    test_variants = {case.sequence for case in test}
    train_ids, test_ids = {case.case_id for case in train}, {case.case_id for case in test}
    train_activities = set().union(*(set(v) for v in train_variants))
    test_activities = set().union(*(set(v) for v in test_variants))
    if not train or not test or train_ids & test_ids or train_variants & test_variants:
        raise ValueError("Partitions must be nonempty with disjoint cases and variants.")
    if len(train_ids | test_ids) != len(log.cases) or train_variants | test_variants != set(log.variants):
        raise ValueError("Partition does not cover the source log exactly.")
    if len(train_variants) != round(ratio * len(log.variants)):
        raise ValueError("Observed variant count does not match the requested budget.")
    if test_activities - train_activities:
        raise ValueError("Normal test activities are missing from the training partition.")
    return {
        "total_variants": len(log.variants), "observed_variants": len(train_variants),
        "unseen_variants": len(test_variants),
        "actual_observed_ratio": len(train_variants) / len(log.variants),
        "actual_unseen_ratio": len(test_variants) / len(log.variants),
        "total_cases": len(log.cases), "train_cases": len(train), "test_cases": len(test),
        "train_case_ratio": len(train) / len(log.cases), "test_case_ratio": len(test) / len(log.cases),
        "train_events": sum(len(case.sequence) for case in train),
        "test_events": sum(len(case.sequence) for case in test),
        "train_csv_rows": sum(case.row_count for case in train),
        "test_csv_rows": sum(case.row_count for case in test),
        "train_activities": len(train_activities), "test_activities": len(test_activities),
        "unknown_test_activities": 0, "shared_variants": 0,
    }


def validate_csv(path: Path, fields: list[str], expected: list[Case]) -> None:
    actual_fields, actual_cases = read_cases(path)
    if actual_fields != fields or actual_cases != expected:
        raise ValueError(f"{path}: read-back validation failed (columns, cases, sequences, rows or order).")


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def audit_validation(train_path: Path, test_path: Path, train: list[Case], test: list[Case],
                     variant_ids: dict[tuple[str, ...], str], config: AuditConfig) -> dict:
    """Replay recovered DCAD preprocessing and validation; never infer coverage from vocab."""
    if not 0 <= config.validation_ratio < 1 or (config.max_len is not None and config.max_len < 1):
        raise ValueError("Invalid validation_ratio or max_len.")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from dcad.variant_ratio import io as recovered
    from dcad.variant_split import split_dataset_by_variant

    raw_sequences = recovered.read_activity_sequences_from_csv(str(train_path), "case_id", "name")
    test_sequences = recovered.read_activity_sequences_from_csv(str(test_path), "case_id", "name")
    sequences = recovered.truncate_sequences(raw_sequences, config.max_len)
    if len(sequences) != len(train) or len(test_sequences) != len(test):
        raise ValueError("Model preprocessing changed case membership; audit cannot align case IDs.")
    vocab, _ = recovered.build_vocab(sequences)
    encoded = recovered.encode_sequences(sequences, vocab)
    dataset = recovered.ActivitySequenceDataset(encoded)
    result = split_dataset_by_variant(dataset, encoded, config.validation_ratio, config.seed)
    training_ids, validation_ids = set(result.train_indices), set(result.validation_indices)
    if training_ids & validation_ids or training_ids | validation_ids != set(range(len(train))):
        raise ValueError("Internal validation split lost or duplicated cases.")

    def partition(indices: list[int]) -> dict:
        tokens = {activity for index in indices for activity in sequences[index]}
        original_variants = {variant_ids[train[index].sequence] for index in indices}
        return {
            "case_count": len(indices), "case_ids": [train[index].case_id for index in indices],
            "variant_count": len(original_variants), "variant_ids": sorted(original_variants),
            "processed_variant_count": len({tuple(encoded[index]) for index in indices}),
            "activity_count": len(tokens - BOUNDARY_MARKERS),
            "activities": sorted(tokens - BOUNDARY_MARKERS), "model_tokens": sorted(tokens),
            "model_token_count": len(tokens),
            "variant_coverage_of_full_log": len(original_variants) / len(variant_ids),
        }

    gradient = partition(result.train_indices)
    validation = partition(result.validation_indices)
    test_tokens = {activity for sequence in test_sequences for activity in sequence}
    # Check full normal test sequences: truncating them here could hide an unseen activity.
    unknown_test = sorted(test_tokens - set(gradient["model_tokens"]))
    unknown_validation = sorted(set(validation["model_tokens"]) - set(gradient["model_tokens"]))
    if unknown_test or unknown_validation:
        raise ValueError(
            f"{train_path}: unknown activities after preprocessing/validation: "
            f"normal_test={unknown_test}, validation={unknown_validation}; config={config}"
        )
    return {
        "passed": True,
        "parameters": {"validation_ratio": config.validation_ratio, "seed": config.seed,
                       "max_len": config.max_len, "activity_col": "name", "case_id_col": "case_id"},
        "implementation_sha256": {
            name: file_sha256(ROOT / name)
            for name in ("dcad/variant_ratio/io.py", "dcad/variant_split.py")
        },
        "coverage_basis": "actual post-preprocessing gradient sequences, including boundary tokens; full normal test",
        "gradient_train": gradient, "validation": validation,
        "normal_test_case_count": len(test),
        "normal_test_activity_count": len(test_tokens - BOUNDARY_MARKERS),
        "unknown_normal_test_activities": unknown_test,
        "unknown_validation_activities": unknown_validation,
    }


def generate_seed(log: Log, dataset: str, base_seed: int, repeat: int,
                  ratios: list[float], output: Path, audit_config: AuditConfig) -> None:
    seed = derive_seed(dataset, base_seed, repeat)
    discovery = variant_order(log, seed, ratios)
    order, cover_count = discovery.variants, len(discovery.startup_case_ids)
    if discovery != variant_order(log, seed, ratios):
        raise ValueError("Trace discovery is not reproducible.")
    variant_ids = {variant: f"v{index:06d}" for index, variant in enumerate(log.variants, 1)}
    selections = {ratio: set(order[:round(ratio * len(order))]) for ratio in ratios}
    previous: set[tuple[str, ...]] = set()
    for ratio in sorted(ratios):
        if not previous <= selections[ratio]:
            raise ValueError("Observed sets are not nested.")
        statistics(log, selections[ratio], ratio)
        previous = selections[ratio]
    destinations = {ratio: output / dataset / f"seed_{seed}" / ratio_directory(ratio) for ratio in ratios}
    with ExitStack() as stack:
        writers = {}
        for ratio, directory in destinations.items():
            directory.mkdir(parents=True, exist_ok=True)
            # An interrupted overwrite must not leave an old success manifest.
            (directory / "split_manifest.json").unlink(missing_ok=True)
            for partition in ("train", "test"):
                path = directory / f"{dataset}_variant_{partition}.csv"
                handle = stack.enter_context(path.open("w", encoding="utf-8-sig", newline=""))
                writers[ratio, partition] = csv.writer(handle)
                writers[ratio, partition].writerow(log.fieldnames)
        case_writers = {
            case.case_id: [writers[ratio, "train" if case.sequence in selections[ratio] else "test"]
                           for ratio in ratios]
            for case in log.cases
        }
        source = stack.enter_context(log.path.open("r", encoding="utf-8-sig", newline=""))
        reader = csv.reader(source)
        if next(reader) != log.fieldnames:
            raise ValueError("Source CSV changed during generation.")
        case_column = log.fieldnames.index("case_id")
        for row in reader:
            for writer in case_writers[row[case_column]]:
                writer.writerow(row)
    if file_sha256(log.path) != log.input_sha256:
        raise ValueError("Source file changed during generation; no manifests will be published.")
    for ratio, directory in destinations.items():
        observed = selections[ratio]
        train, test = split_cases(log, observed)
        files = {}
        for partition, expected in (("train", train), ("test", test)):
            path = directory / f"{dataset}_variant_{partition}.csv"
            validate_csv(path, log.fieldnames, expected)
            files[partition] = {"filename": path.name, "sha256": file_sha256(path)}
        metrics = statistics(log, observed, ratio)
        audit = audit_validation(directory / files["train"]["filename"],
                                 directory / files["test"]["filename"], train, test,
                                 variant_ids, audit_config)
        manifest = {
            "schema_version": 2, "dataset": dataset, "seed": seed,
            "base_seed": base_seed, "repeat": repeat, "algorithm_version": ALGORITHM_VERSION,
            "activity_col": "name", "case_id_col": "case_id",
            "boundary_markers": sorted(BOUNDARY_MARKERS),
            "requested_observed_ratio": ratio, "rounding": "Python round(ratio * total_variants)",
            "observed_scope": "outer training candidate set; gradient-training coverage recorded separately",
            "selection": "greedy startup traces then shuffled remaining traces; first-discovered variants",
            "activity_cover_variants": cover_count,
            "startup_case_ids": discovery.startup_case_ids,
            "trace_order_case_ids": discovery.trace_case_ids,
            "variant_discovery_order": [variant_ids[variant] for variant in order],
            "input_path": str(log.path), "input_sha256": log.input_sha256,
            "fieldnames": log.fieldnames, "statistics": metrics, "files": files,
            "validation": {"read_back_verified": True, "nested_observed_sets": True,
                           "reproducible_discovery_order": True},
            "validation_audit": audit,
            "variants": [
                {"variant_id": f"v{index:06d}", "activity_sequence": list(variant),
                 "case_ids": case_ids, "partition": "train" if variant in observed else "test"}
                for index, (variant, case_ids) in enumerate(log.variants.items(), 1)
            ],
        }
        atomic_json(directory / "split_manifest.json", manifest)
        print(f"  {dataset} seed={seed} {ratio_directory(ratio)}: "
              f"variants {len(observed)}/{len(order) - len(observed)}, "
              f"cases {len(train)}/{len(test)}; read-back and validation coverage verified", flush=True)


def write_summary(output: Path) -> int:
    rows = []
    for path in sorted(output.glob("BPIC*/seed_*/observed_*/split_manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        row = {key: manifest[key] for key in (
            "dataset", "base_seed", "repeat", "seed", "algorithm_version",
            "requested_observed_ratio", "activity_cover_variants", "input_sha256"
        )}
        row.update(manifest["statistics"])
        audit = manifest["validation_audit"]
        row.update(validation_ratio=audit["parameters"]["validation_ratio"],
                   validation_seed=audit["parameters"]["seed"], max_len=audit["parameters"]["max_len"])
        for prefix in ("gradient_train", "validation"):
            for column, key in (("cases", "case_count"), ("variants", "variant_count"),
                                ("processed_variants", "processed_variant_count"),
                                ("activities", "activity_count")):
                row[f"{prefix}_{column}"] = audit[prefix][key]
        row["gradient_train_variant_coverage"] = audit["gradient_train"]["variant_coverage_of_full_log"]
        row["unknown_normal_test_activities_after_validation"] = len(audit["unknown_normal_test_activities"])
        row["unknown_validation_activities"] = len(audit["unknown_validation_activities"])
        row["split_directory"] = str(path.parent.relative_to(output))
        rows.append(row)
    temporary = output / "summary.csv.tmp"
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / "summary.csv")
    mapping = {}
    for row in rows:
        key = (row["dataset"], row["seed"])
        mapping[key] = {name: row[name] for name in ("dataset", "base_seed", "repeat", "seed", "algorithm_version")}
    atomic_json(output / "seed_mapping.json", {
        "derivation": "int.from_bytes(sha256(f'{algorithm_version}|{dataset}|{base_seed}|{repeat}'.encode()).digest()[:4], 'big')",
        "repeat_index_starts_at": 1, "seeds": list(mapping.values()),
    })
    return len(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--observed-ratios", nargs="+", type=float, default=list(RATIOS))
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1, help="Independent repetitions per dataset (indexed from 1).")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--validation-seed", type=int, default=2026)
    parser.add_argument("--max-len", type=int, default=None, help="Recovered preprocessing truncation; default: no truncation.")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data" / "processed" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "variant_ratio" / "splits")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace selected existing splits.")
    parser.add_argument("--audit-only", action="store_true",
                        help="Verify existing CSVs and rerun validation with current parameters; no files modified.")
    args = parser.parse_args(argv)
    for name in ("datasets", "observed_ratios"):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            parser.error(f"--{name.replace('_', '-')} must not contain duplicates.")
    if any(not 0 < ratio < 1 for ratio in args.observed_ratios):
        parser.error("Observed ratios must be strictly between 0 and 1.")
    if args.repeats < 1 or not 0 <= args.validation_ratio < 1 or (args.max_len is not None and args.max_len < 1):
        parser.error("repeats/max-len must be positive; validation-ratio must be in [0, 1).")
    if args.overwrite and args.audit_only:
        parser.error("--audit-only cannot be combined with --overwrite.")
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def audit_existing(log: Log, dataset: str, seed: int, ratios: list[float],
                   output: Path, config: AuditConfig) -> None:
    """Read-only re-audit, also checking membership against reproducible discovery."""
    discovery = variant_order(log, seed, ratios)
    variant_ids = {variant: f"v{index:06d}" for index, variant in enumerate(log.variants, 1)}
    for ratio in ratios:
        directory = output / dataset / f"seed_{seed}" / ratio_directory(ratio)
        manifest = json.loads((directory / "split_manifest.json").read_text(encoding="utf-8"))
        if manifest["input_sha256"] != log.input_sha256 or manifest.get("algorithm_version") != ALGORITHM_VERSION:
            raise ValueError(f"{directory}: source hash or algorithm version mismatch.")
        if (manifest["trace_order_case_ids"] != discovery.trace_case_ids
                or manifest["startup_case_ids"] != discovery.startup_case_ids
                or manifest["variant_discovery_order"] != [variant_ids[v] for v in discovery.variants]):
            raise ValueError(f"{directory}: discovery metadata is not reproducible.")
        observed = set(discovery.variants[:round(ratio * len(log.variants))])
        train, test = split_cases(log, observed)
        expected_variants = [
            {"variant_id": variant_ids[v], "activity_sequence": list(v), "case_ids": ids,
             "partition": "train" if v in observed else "test"}
            for v, ids in log.variants.items()
        ]
        if manifest["variants"] != expected_variants:
            raise ValueError(f"{directory}: variant membership metadata mismatch.")
        if manifest["statistics"] != statistics(log, observed, ratio):
            raise ValueError(f"{directory}: statistics mismatch.")
        for partition, expected in (("train", train), ("test", test)):
            path = directory / manifest["files"][partition]["filename"]
            if file_sha256(path) != manifest["files"][partition]["sha256"]:
                raise ValueError(f"{path}: output hash mismatch.")
            validate_csv(path, log.fieldnames, expected)
        audit = audit_validation(directory / manifest["files"]["train"]["filename"],
                                 directory / manifest["files"]["test"]["filename"],
                                 train, test, variant_ids, config)
        saved = manifest["validation_audit"]
        if (saved["parameters"] == audit["parameters"]
                and {k: v for k, v in saved.items() if k != "implementation_sha256"}
                != {k: v for k, v in audit.items() if k != "implementation_sha256"}):
            raise ValueError(f"{directory}: validation audit is not reproducible.")
        print(f"  Re-audited {dataset} seed={seed} {ratio_directory(ratio)}: "
              f"gradient cases={audit['gradient_train']['case_count']}, "
              f"variant coverage={audit['gradient_train']['variant_coverage_of_full_log']:.6f}; "
              "unknown normal-test/validation activities=0/0", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = AuditConfig(args.validation_ratio, args.validation_seed, args.max_len)
    # Check every requested destination before writing anything.
    for dataset in args.datasets:
        source = args.input_dir / f"{dataset}.csv"
        if not source.is_file():
            raise FileNotFoundError(source)
        seeds = [derive_seed(dataset, args.base_seed, repeat) for repeat in range(1, args.repeats + 1)]
        if len(set(seeds)) != len(seeds):
            raise ValueError("Derived seed collision; choose another base seed.")
        for seed in seeds:
            for ratio in args.observed_ratios:
                destination = args.output_dir / dataset / f"seed_{seed}" / ratio_directory(ratio)
                if args.audit_only:
                    if not (destination / "split_manifest.json").is_file():
                        raise FileNotFoundError(destination / "split_manifest.json")
                elif destination.exists() and not args.overwrite:
                    raise FileExistsError(f"{destination} exists; use --overwrite to replace it explicitly.")
    for dataset in args.datasets:
        print(f"Reading {dataset} (variant and activity column: name)...", flush=True)
        log = load_log(args.input_dir / f"{dataset}.csv")
        for repeat in range(1, args.repeats + 1):
            if args.audit_only:
                audit_existing(log, dataset, derive_seed(dataset, args.base_seed, repeat),
                               args.observed_ratios, args.output_dir, config)
            else:
                generate_seed(log, dataset, args.base_seed, repeat, args.observed_ratios, args.output_dir, config)
                write_summary(args.output_dir)
    if args.audit_only:
        print(f"Read-only audit complete: {len(args.datasets) * args.repeats * len(args.observed_ratios)} splits; {config}")
        return
    count = write_summary(args.output_dir)
    print(f"Complete: {count} validated splits listed in {args.output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
