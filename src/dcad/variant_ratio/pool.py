#!/usr/bin/env python3
"""Build fixed BPIC anomaly pools and normal:anomaly ~= 2:1 test sets."""
from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import io
import json
import random
import shutil
import sqlite3
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from dcad.variant_ratio.injection import (
    EventLogStats, build_control_flow_anomalies, event_attribute_keys, parse_xes_header,
)
from dcad.processmining.case import Case
from dcad.processmining.event import Event
from dcad.processmining.log import EventLog
from dcad.variant_ratio import split
from dcad.anomaly import label_to_targets

VERSION = "anomaly_pool_v1"
BOUNDARIES = split.BOUNDARY_MARKERS
TYPES = ("SkipSequence", "Rework", "Early", "Late", "Insert")
IMPLEMENTATIONS = (
    "src/dcad/variant_ratio/pool.py", "src/dcad/variant_ratio/split.py",
    "src/dcad/variant_ratio/injection.py", "src/dcad/variant_ratio/io.py",
    "src/dcad/variant_split.py", "src/dcad/generation/anomaly.py",
    "src/dcad/generation/attribute_generator.py", "src/dcad/processmining/case.py",
    "src/dcad/processmining/event.py", "src/dcad/processmining/log.py", "src/dcad/anomaly.py",
)
SUMMARY_FIELDS = (
    "dataset", "repeat", "split_seed", "observed_ratio", "normal_traces", "anomalous_traces",
    "total_traces", "normal_to_anomaly_ratio", "anomaly_fraction", "pool_capacity",
    "pool_sha256", "test_sha256", "test_path", "unknown_normal_activities",
    *TYPES,
)


def derive_seed(dataset: str, base_seed: int, purpose: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{VERSION}|{dataset}|{base_seed}|{purpose}".encode()).digest()[:4], "big")


def anomaly_count(normal_count: int) -> int:
    if normal_count < 1:
        raise ValueError("A mixed test set requires at least one normal trace.")
    return (normal_count + 1) // 2


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def object_hash(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def case_from_json(value: dict) -> Case:
    return Case(id=value["id"], events=[Event(name=e["name"], timestamp=e.get("timestamp"),
                timestamp_end=e.get("timestamp_end"), **e.get("attributes", {})) for e in value["events"]],
                **value["attributes"])


def csv_cases(path: Path):
    """Stream original contiguous cases, using only cleaned name for activities."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"name", "case_id"} <= set(reader.fieldnames):
            raise ValueError(f"{path}: case_id and name are required.")
        current, events, closed = None, [], set()
        for row in reader:
            cid = row["case_id"]
            if not cid.strip():
                raise ValueError(f"{path}: empty case ID.")
            if cid != current:
                if current is not None:
                    if not events:
                        raise ValueError(f"{path}: empty cleaned case {current}.")
                    yield Case(id=current, events=events, label="normal")
                    closed.add(current)
                if cid in closed:
                    raise ValueError(f"{path}: noncontiguous case {cid}.")
                current, events = cid, []
            name = row["name"].strip()
            if name in BOUNDARIES:
                continue
            attrs = {key: (value or "").strip() for key, value in row.items()
                     if key not in {"case_id", "event_position", "name", "timestamp", "timestamp_end"}}
            events.append(Event(name=name, timestamp=row.get("timestamp") or row.get("time:timestamp"),
                                timestamp_end=row.get("timestamp_end") or row.get("dateStop"), **attrs))
        if current is not None:
            if not events:
                raise ValueError(f"{path}: empty cleaned case {current}.")
            yield Case(id=current, events=events, label="normal")


class MotherStore:
    """Temporary disk index: random case access without retaining the full event log."""
    def __init__(self, directory: Path, source: split.Log, metadata: dict):
        self.connection = sqlite3.connect(directory / "normal_cases.sqlite")
        self.connection.execute("CREATE TABLE cases (i INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
        self.source = source
        values = None
        count = 0
        for count, case in enumerate(csv_cases(source.path), 1):
            expected = source.cases[count - 1]
            if case.id != expected.case_id or tuple(case.trace) != expected.sequence:
                raise ValueError("Normal mother preprocessing disagrees with the split variant definition.")
            if values is None:
                keys = event_attribute_keys(metadata, case.events[0].attributes)
                values = {key: set() for key in keys if key != "name"}
            for event in case.events:
                for key in values:
                    values[key].add(event.attributes.get(key, ""))
            self.connection.execute("INSERT INTO cases VALUES (?, ?)", (count - 1, canonical(case.json)))
        self.connection.commit()
        if count != len(source.cases):
            raise ValueError("Normal mother case count changed.")
        activities = sorted({activity for variant in source.variants for activity in variant})
        attribute_values = {"name": activities, **{k: sorted(v) for k, v in (values or {}).items()}}
        # The existing Insert injector draws randint(1, len(values)); never silently change it.
        if len(activities) < 2 or any(len(v) < 2 for k, v in attribute_values.items() if k != "name"):
            raise ValueError("The existing Insert injector requires >=2 activities and >=2 values per selected attribute.")
        self.stats = EventLogStats(metadata, activities, attribute_values, count)

    def __len__(self):
        return len(self.source.cases)

    def __getitem__(self, index: int) -> Case:
        payload = self.connection.execute("SELECT payload FROM cases WHERE i=?", (index,)).fetchone()[0]
        return case_from_json(json.loads(payload))

    def close(self):
        self.connection.close()


class PoolExhausted(ValueError):
    def __init__(self, report: dict):
        self.report = report
        super().__init__(f"Anomaly generation exhausted {report['attempts']} attempts: {report}")


@contextmanager
def numpy_seed(seed: int):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def rejection_reason(case: Case, normal_variants: set[tuple[str, ...]]) -> str | None:
    if not case.events:
        return "empty_sequence"
    if not isinstance(case.attributes.get("label"), dict):
        return "normal_label"
    if tuple(case.trace) in normal_variants:
        return "normal_variant_collision"
    return None


def generate_pool(mothers, normal_variants: set[tuple[str, ...]], injectors: list,
                  dataset: str, capacity: int, base_seed: int, max_attempts: int | None = None):
    seed = derive_seed(dataset, base_seed, "generation")
    order_seed = derive_seed(dataset, base_seed, "order")
    rng = random.Random(seed)
    maximum = 100 * capacity if max_attempts is None else max_attempts
    cases, provenance = [], []
    rejected, attempted_types = Counter(), Counter()
    with numpy_seed(seed):
        for attempt in range(1, maximum + 1):
            # Clone even when the store returns a fresh object; never alter a mother.
            mother = mothers[rng.randrange(len(mothers))]
            case = Case.clone(mother)
            injector = injectors[rng.randrange(len(injectors))]
            attempted_types[str(injector)] += 1
            injector.apply_to_case(case)
            reason = rejection_reason(case, normal_variants)
            if reason:
                rejected[reason] += 1
                continue
            pool_id = f"anomaly:{dataset}:{len(cases) + 1:06d}"
            case.id = pool_id
            label = case.attributes["label"]
            cases.append(case.json)
            provenance.append({"pool_case_id": pool_id, "source_case_id": mother.id, "attempt": attempt,
                               "anomaly_type": label["anomaly"], "case_sha256": object_hash(case.json)})
            if len(cases) == capacity:
                break
    report = {"attempts": sum(attempted_types.values()), "maximum_attempts": maximum,
              "accepted": len(cases), "rejections": dict(sorted(rejected.items())),
              "attempted_types": dict(sorted(attempted_types.items()))}
    if len(cases) != capacity:
        raise PoolExhausted(report)
    indices = list(range(capacity))
    random.Random(order_seed).shuffle(indices)
    cases = [cases[i] for i in indices]
    provenance = [provenance[i] for i in indices]
    report.update(generation_seed=seed, order_seed=order_seed,
                  accepted_types=dict(sorted(Counter(c['attributes']['label']['anomaly'] for c in cases).items())),
                  distinct_anomaly_variants=len({tuple(e['name'] for e in c['events']) for c in cases}))
    report['duplicate_variant_cases'] = capacity - report['distinct_anomaly_variants']
    return cases, provenance, report


def write_log(path: Path, metadata: dict, cases) -> None:
    """Stream deterministic gzip (no wall-clock time or temporary filename in header)."""
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as handle:
                handle.write('{"attributes":' + canonical(metadata) + ',"cases":[')
                first = True
                for case in cases:
                    if not first:
                        handle.write(",")
                    handle.write(canonical(case))
                    first = False
                handle.write("]}\n")


def check_targets(case: dict, num_attributes: int) -> None:
    label = case["attributes"]["label"]
    targets = label_to_targets(label, len(case["events"]) + 2, num_attributes)
    if targets.shape != (len(case["events"]) + 2, num_attributes):
        raise ValueError("Invalid event target shape.")
    if label == "normal":
        if np.any(targets):
            raise ValueError("Normal trace has anomalous event targets.")
    elif not np.any(targets[:, 0]):
        raise ValueError("Anomalous trace has no control-flow targets.")


def verify_log(path: Path, expected, metadata: dict) -> dict:
    """Read back every case and replay the actual inference reader and label decoder."""
    from dcad.variant_ratio.io import load_test_traces
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload["attributes"] != metadata:
        raise ValueError(f"{path}: metadata changed on read-back.")
    count = 0
    ids = set()
    num_attributes = len(event_attribute_keys(metadata, payload['cases'][0]['events'][0]['attributes']))
    for count, wanted in enumerate(expected, 1):
        if count > len(payload['cases']) or payload['cases'][count - 1] != wanted:
            raise ValueError(f"{path}: read-back case mismatch at {count}.")
        case = payload['cases'][count - 1]
        if case['id'] in ids:
            raise ValueError(f"{path}: duplicate case ID.")
        ids.add(case['id'])
        if any(e['name'] in BOUNDARIES for e in case['events']):
            raise ValueError("Serialized test logs must not contain boundary events.")
        check_targets(case, num_attributes)
    if count != len(payload['cases']):
        raise ValueError(f"{path}: extra cases on read-back.")
    sequence_digests = [object_hash([e['name'] for e in c['events']]) for c in payload['cases']]
    del payload
    gc.collect()
    traces = load_test_traces(str(path.resolve()))
    if [object_hash(t) for t in traces] != sequence_digests:
        raise ValueError("Actual inference reader changed test activity sequences.")
    return {"read_back_verified": True, "inference_reader_verified": True,
            "event_labels_verified": True, "case_count": count}


@dataclass
class NormalSplit:
    path: Path
    manifest: dict
    manifest_sha256: str
    train_path: Path
    test_path: Path
    test_case_ids: list[str]


def discover_splits(split_dir: Path, dataset: str, raw_path: Path) -> list[NormalSplit]:
    results = []
    log = split.load_log(raw_path)
    raw_hash = log.input_sha256
    for path in sorted((split_dir / dataset).glob("seed_*/observed_*/split_manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get('dataset') != dataset or manifest.get('activity_col') != 'name':
            raise ValueError(f"{path}: unsupported dataset/activity column.")
        if manifest['input_sha256'] != raw_hash:
            raise ValueError(f"{path}: raw input hash mismatch.")
        audit = manifest['validation_audit']
        if not audit['passed'] or audit['unknown_normal_test_activities'] or audit['unknown_validation_activities']:
            raise ValueError(f"{path}: normal split validation audit did not pass.")
        # Recompute the audit with the published readers. Historical implementation
        # hashes remain provenance; they are not hashes of the relocated code.
        split.audit_existing(log, dataset, manifest['seed'],
                             [manifest['requested_observed_ratio']], split_dir,
                             split.AuditConfig(audit['parameters']['validation_ratio'],
                                               audit['parameters']['seed'],
                                               audit['parameters']['max_len']))
        paths = {}
        for partition in ('train', 'test'):
            target = (path.parent / manifest['files'][partition]['filename']).resolve()
            if target.parent != path.parent.resolve():
                raise ValueError("Split CSV must reside alongside its manifest.")
            if split.file_sha256(target) != manifest['files'][partition]['sha256']:
                raise ValueError(f"{target}: input CSV hash mismatch.")
            paths[partition] = target
        test_ids = [cid for variant in manifest['variants'] if variant['partition'] == 'test'
                    for cid in variant['case_ids']]
        # Full variant sequences and trace permutations are large; retain only the
        # verified fields needed for assembly rather than 105 full manifests.
        compact = {key: manifest[key] for key in
                   ('dataset', 'repeat', 'seed', 'requested_observed_ratio', 'statistics', 'files')}
        compact['validation_audit'] = {
            'parameters': audit['parameters'],
            'gradient_train': {'activities': audit['gradient_train']['activities']},
        }
        results.append(NormalSplit(path, compact, split.file_sha256(path),
                                   paths['train'], paths['test'], test_ids))
    if not results:
        raise ValueError(f"No normal split manifests for {dataset} in {split_dir}.")
    # Refuse silently skipping a split that was interrupted before publishing its manifest.
    directories = [p for p in (split_dir / dataset).glob('seed_*/observed_*') if p.is_dir()]
    if {p.resolve() for p in directories} != {r.path.parent.resolve() for r in results}:
        raise ValueError(f"{dataset}: incomplete normal split directories.")
    return results


def normal_cases_for_test(record: NormalSplit):
    expected = record.test_case_ids
    expected_ids = set(expected)
    if len(expected) != len(expected_ids):
        raise ValueError("Normal split manifest repeats a case ID.")
    known = set(record.manifest['validation_audit']['gradient_train']['activities'])
    seen = set()
    for case in csv_cases(record.test_path):
        if case.id not in expected_ids or case.id in seen:
            raise ValueError("Normal CSV does not match manifest membership.")
        if set(case.trace) - known:
            raise ValueError(f"Unknown normal activities after validation: {sorted(set(case.trace) - known)}")
        seen.add(case.id)
        case.id = f"normal:{case.id}"
        yield case.json
    if seen != expected_ids or len(seen) != record.manifest['statistics']['test_cases']:
        raise ValueError("Normal test cases were lost.")


def publish_log(directory: Path, filename: str, metadata: dict, factory) -> tuple[dict, dict]:
    path = directory / filename
    temporary = directory / (filename + '.tmp.json.gz')
    write_log(temporary, metadata, factory())
    checks = verify_log(temporary, factory(), metadata)
    digest = split.file_sha256(temporary)
    temporary.replace(path)
    return {"filename": filename, "sha256": digest}, checks


def implementation_hashes() -> dict:
    return {name: split.file_sha256(ROOT / name) for name in IMPLEMENTATIONS}


def build_dataset(args, dataset: str, records: list[NormalSplit]) -> None:
    raw_path = args.raw_dir / f'{dataset}.csv'
    xes_path = args.xes_dir / f'{dataset}.xes.gz'
    if not xes_path.is_file():
        xes_path = args.xes_dir / f'{dataset}.xes'
    xes_hash = split.file_sha256(xes_path)
    metadata = parse_xes_header(xes_path)
    source = split.load_log(raw_path)
    required = max(anomaly_count(r.manifest['statistics']['test_cases']) for r in records)
    config = json.loads(args.pool_config.read_text(encoding='utf-8'))
    capacity = config['capacities'][dataset]
    if config['base_seed'] != args.base_seed or type(capacity) is not int or capacity < required:
        raise ValueError('Pool configuration seed/capacity does not cover the requested splits.')
    impl = implementation_hashes()
    pool_destination = args.output_dir / 'anomaly_pools' / dataset
    mixed_destination = args.output_dir / 'mixed_testsets' / dataset
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Stage a whole dataset: interruption never publishes a partial new pool/test family.
    with tempfile.TemporaryDirectory(prefix=f'.anomaly_stage_{dataset}_', dir=args.output_dir) as temporary:
        stage = Path(temporary)
        pool_dir, mixed_dir = stage / 'pool', stage / 'mixed'
        pool_dir.mkdir(); mixed_dir.mkdir()
        store = MotherStore(stage, source, metadata)
        try:
            injectors = build_control_flow_anomalies(store.stats)
            parameters = {str(a): {k:v for k,v in vars(a).items()
                                   if k not in {'graph','activities','attributes','name'}} for a in injectors}
            cases, provenance, generation = generate_pool(store, set(source.variants), injectors,
                                                          dataset, capacity, args.base_seed)
        except PoolExhausted as exc:
            split.atomic_json(args.output_dir / f'{dataset}_anomaly_failure.json',
                              {'dataset':dataset, 'base_seed':args.base_seed, **exc.report})
            raise
        finally:
            store.close()
        (stage / 'normal_cases.sqlite').unlink()
        print(f"{dataset}: pool accepted {capacity}/{generation['attempts']} attempts; rejected {generation['rejections']}", flush=True)
        pool_file, checks = publish_log(pool_dir, f'{dataset}_anomaly_pool.json.gz', metadata, lambda: iter(cases))
        pool_manifest = {
            'schema_version':1, 'algorithm_version':VERSION, 'dataset':dataset, 'base_seed':args.base_seed,
            'activity_col':'name', 'boundary_markers':sorted(BOUNDARIES), 'capacity':capacity,
            'raw_path':str(raw_path), 'raw_sha256':source.input_sha256,
            'xes_metadata_path':str(xes_path), 'xes_sha256':xes_hash, 'implementation_sha256':impl,
            'injector_parameters':parameters, 'generation':generation,
            'collision_policy':'reject any full normal name variant; anomaly variants need not be unique',
            'file':pool_file, 'validation':checks, 'ordered_cases':provenance,
        }
        split.atomic_json(pool_dir / 'pool_manifest.json', pool_manifest)
        pool_manifest_hash = split.file_sha256(pool_dir / 'pool_manifest.json')
        for record in records:
            m = record.manifest
            normal_count = m['statistics']['test_cases']
            k = anomaly_count(normal_count)
            selected = cases[:k]
            relative = record.path.parent.relative_to(args.split_dir / dataset)
            target = mixed_dir / relative
            target.mkdir(parents=True)
            factory = lambda r=record, chosen=selected: chain(normal_cases_for_test(r), iter(chosen))
            filename = f"{dataset}_repeat_{m['repeat']}_seed_{m['seed']}_{split.ratio_directory(m['requested_observed_ratio'])}_custom_test.json.gz"
            result_file, validation = publish_log(target, filename, metadata, factory)
            if split.file_sha256(record.test_path) != m['files']['test']['sha256']:
                raise ValueError("Normal CSV changed while constructing mixed test.")
            type_counts = dict(sorted(Counter(c['attributes']['label']['anomaly'] for c in selected).items()))
            normal_mapping = [{'case_id':c['id'], 'source_case_id':c['id'][len('normal:'):]} for c in normal_cases_for_test(record)]
            manifest = {
                'schema_version':1, 'algorithm_version':VERSION, 'dataset':dataset,
                'repeat':m['repeat'], 'split_seed':m['seed'], 'observed_ratio':m['requested_observed_ratio'],
                'normal_split_manifest':str(record.path), 'normal_split_manifest_sha256':record.manifest_sha256,
                'normal_test_csv':str(record.test_path), 'normal_test_csv_sha256':m['files']['test']['sha256'],
                'pool_manifest':str(pool_destination / 'pool_manifest.json'), 'pool_manifest_sha256':pool_manifest_hash,
                'pool_path':str(pool_destination / pool_file['filename']), 'pool_sha256':pool_file['sha256'],
                'pool_capacity':capacity, 'normal_traces':normal_count, 'anomalous_traces':k,
                'total_traces':normal_count+k, 'target_normal_to_anomaly_ratio':2,
                'normal_to_anomaly_ratio':normal_count/k, 'anomaly_fraction':k/(normal_count+k),
                'rounding':'ceil(normal_traces / 2)', 'anomaly_types':type_counts,
                'normal_cases':normal_mapping, 'anomaly_case_ids':[c['id'] for c in selected],
                'normal_validation_parameters':m['validation_audit']['parameters'],
                'unknown_normal_activities':[], 'file':result_file, 'validation':validation,
            }
            split.atomic_json(target / 'test_manifest.json', manifest)
            print(f"  {dataset} repeat={m['repeat']} {relative.name}: normal={normal_count}, anomaly={k}; verified", flush=True)
        if split.file_sha256(raw_path) != source.input_sha256 or split.file_sha256(xes_path) != xes_hash:
            raise ValueError("Source CSV/XES changed during construction.")
        if implementation_hashes() != impl:
            raise ValueError("Implementation changed during construction.")
        for record in records:
            if split.file_sha256(record.path) != record.manifest_sha256:
                raise ValueError("Normal split manifest changed during construction.")
        publish_family(pool_dir, mixed_dir, pool_destination, mixed_destination, args.overwrite)
    (args.output_dir / f'{dataset}_anomaly_failure.json').unlink(missing_ok=True)


def publish_family(pool_stage: Path, mixed_stage: Path, pool: Path, mixed: Path, overwrite: bool):
    """Publish a matching pair with a marker so a restart detects a partial commit."""
    marker = pool.parent.parent / f'.anomaly_publish_{pool.name}.json'
    if (pool.exists() or mixed.exists() or marker.exists()) and not overwrite:
        raise FileExistsError(f"{pool.name}: output exists; use --overwrite.")
    split.atomic_json(marker, {'dataset':pool.name, 'status':'publishing; rerun with --overwrite if interrupted'})
    for src, dst in ((pool_stage, pool), (mixed_stage, mixed)):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        src.replace(dst)
    marker.unlink()


def write_summary(output: Path) -> int:
    rows = []
    for path in sorted((output / 'mixed_testsets').glob('BPIC*/seed_*/observed_*/test_manifest.json')):
        m = json.loads(path.read_text())
        if (output / f".anomaly_publish_{m['dataset']}.json").exists():
            continue
        row = {key:m[key] for key in SUMMARY_FIELDS if key in m}
        row.update(unknown_normal_activities=len(m['unknown_normal_activities']),
                   test_sha256=m['file']['sha256'], test_path=str((path.parent / m['file']['filename']).relative_to(output)))
        row.update({kind:m['anomaly_types'].get(kind,0) for kind in TYPES})
        rows.append(row)
    temporary = output / 'mixed_testsets' / 'summary.csv.tmp'
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open('w',encoding='utf-8-sig',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=SUMMARY_FIELDS)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(temporary.with_suffix(''))
    return len(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=split.DATASETS, default=list(split.DATASETS))
    parser.add_argument('--base-seed', type=int, default=42)
    parser.add_argument('--raw-dir', type=Path, default=ROOT/'data/processed/raw')
    parser.add_argument('--xes-dir', type=Path, default=ROOT/'data/original/real')
    parser.add_argument('--split-dir', type=Path, default=ROOT/'data/variant_ratio/splits')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'data/variant_ratio')
    parser.add_argument('--pool-config', type=Path, default=ROOT/'data/variant_ratio/pool_config.json',
                        help='Fixed release pool capacities; use a custom JSON for other experiments.')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    if len(args.datasets) != len(set(args.datasets)):
        parser.error('Duplicate datasets.')
    for key in ('raw_dir','xes_dir','split_dir','output_dir'):
        setattr(args,key,getattr(args,key).expanduser().resolve())
    return args


def main(argv=None):
    args = parse_args(argv)
    records = {}
    # Preflight every selected dataset before changing any outputs.
    for dataset in args.datasets:
        for part in ('anomaly_pools','mixed_testsets'):
            target = args.output_dir/part/dataset
            if target.exists() and not args.overwrite:
                raise FileExistsError(f'{target} exists; use --overwrite to rebuild pool and all its test sets.')
        incomplete = list(args.output_dir.glob(f'.anomaly_stage_{dataset}_*'))
        marker = args.output_dir/f'.anomaly_publish_{dataset}.json'
        if (incomplete or marker.exists()) and not args.overwrite:
            raise FileExistsError(f'{dataset}: interrupted generation; rerun with --overwrite.')
        if not any((args.xes_dir/f'{dataset}{suffix}').is_file() for suffix in ('.xes.gz','.xes')):
            raise FileNotFoundError(f'{dataset}: missing XES metadata in {args.xes_dir}')
        records[dataset] = discover_splits(args.split_dir,dataset,args.raw_dir/f'{dataset}.csv')
    for dataset in args.datasets:
        if args.overwrite:
            for old_stage in args.output_dir.glob(f'.anomaly_stage_{dataset}_*'):
                shutil.rmtree(old_stage)
        build_dataset(args,dataset,records[dataset])
        write_summary(args.output_dir)
        gc.collect()
    count = write_summary(args.output_dir)
    print(f'Complete: {len(args.datasets)} anomaly pools built; {count} mixed tests in {args.output_dir / "mixed_testsets/summary.csv"}',flush=True)


if __name__ == '__main__':
    main()
