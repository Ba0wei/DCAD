"""Validate the published ratio datasets without using historical machine paths."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import gzip
from itertools import chain
import json
from pathlib import Path

from . import pool, split


def read_json(path):
    with (gzip.open(path, 'rt', encoding='utf-8') if path.suffix == '.gz'
          else path.open(encoding='utf-8')) as stream:
        return json.load(stream)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_index(root: Path) -> dict:
    index = read_json(root / 'release_index.json')
    require(index['base_seed'] == 42 and index['repeat'] == 1, 'Unexpected release protocol')
    paths = set()
    for entry in index['files']:
        relative = Path(entry['path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe index path')
        require(entry['path'] not in paths, 'Duplicate index path')
        paths.add(entry['path'])
        target = root / relative
        require(target.is_file() and target.stat().st_size == entry['bytes'], f'Missing/changed file: {target}')
        require(split.file_sha256(target) == entry['sha256'], f'Hash mismatch: {target}')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
              and p not in {root / 'release_index.json', root / 'audit_report.json'}}
    require(paths == actual, 'Index does not cover the release files exactly')
    require({(r['dataset'], r['ratio']) for r in index['splits']} ==
            {(d, r) for d in split.DATASETS for r in split.RATIOS}
            and len(index['splits']) == 35, 'Expected exactly 35 dataset/ratio pairs')
    return index


def validate_release(root: Path, raw: Path, xes: Path) -> dict:
    index = validate_index(root)
    config = read_json(root / 'pool_config.json')
    reports = []
    for dataset in split.DATASETS:
        print(f'Validating {dataset}...', flush=True)
        records = pool.discover_splits(root / 'splits', dataset, raw / f'{dataset}.csv')
        require(len(records) == 5, f'{dataset}: expected five splits')
        require(all(r.manifest['repeat'] == 1 and r.manifest['seed'] == split.derive_seed(dataset, 42, 1)
                    for r in records), 'Unexpected repeat/seed')
        log = split.load_log(raw / f'{dataset}.csv')
        folder = root / 'anomaly_pools' / dataset
        pm = read_json(folder / 'pool_manifest.json')
        pool_path = folder / pm['file']['filename']
        require(pm['dataset'] == dataset and pm['base_seed'] == 42, 'Pool identity mismatch')
        require(pm['raw_sha256'] == log.input_sha256, 'Pool raw source mismatch')
        xes_path = xes / Path(pm['xes_metadata_path']).name
        require(split.file_sha256(xes_path) == pm['xes_sha256'], 'XES source mismatch')
        require(split.file_sha256(pool_path) == pm['file']['sha256'], 'Pool hash mismatch')
        payload = read_json(pool_path)
        cases, metadata = payload['cases'], payload['attributes']
        require(len(cases) == pm['capacity'] == config['capacities'][dataset], 'Pool capacity mismatch')
        require(config['base_seed'] == pm['base_seed'], 'Pool seed configuration mismatch')
        require(len(pm['ordered_cases']) == len(cases), 'Pool provenance length mismatch')
        source_ids = {c.case_id for c in log.cases}
        for case, provenance in zip(cases, pm['ordered_cases']):
            require(case['id'] == provenance['pool_case_id'] and
                    pool.object_hash(case) == provenance['case_sha256'], 'Pool case provenance mismatch')
            require(provenance['source_case_id'] in source_ids, 'Unknown anomaly source case')
            require(tuple(e['name'] for e in case['events']) not in log.variants,
                    'Anomaly collides with a normal variant')
        pool.verify_log(pool_path, iter(cases), metadata)
        for record in records:
            m = record.manifest
            rel = record.path.parent.relative_to(root / 'splits')
            tm_path = root / 'mixed_testsets' / rel / 'test_manifest.json'
            tm = read_json(tm_path)
            normal_count = m['statistics']['test_cases']
            count = pool.anomaly_count(normal_count)
            require((tm['dataset'], tm['repeat'], tm['split_seed'], tm['observed_ratio']) ==
                    (dataset, 1, m['seed'], m['requested_observed_ratio']), 'Mixed-test identity mismatch')
            require(tm['normal_split_manifest_sha256'] == split.file_sha256(record.path), 'Split manifest link mismatch')
            require(tm['normal_test_csv_sha256'] == split.file_sha256(record.test_path), 'Normal test link mismatch')
            require(tm['pool_manifest_sha256'] == split.file_sha256(folder / 'pool_manifest.json')
                    and tm['pool_sha256'] == pm['file']['sha256'], 'Pool link mismatch')
            require(tm['normal_traces'] == normal_count and tm['anomalous_traces'] == count
                    and tm['total_traces'] == normal_count + count, 'Mixed-test counts mismatch')
            require(tm['anomaly_case_ids'] == [c['id'] for c in cases[:count]], 'Pool prefix mismatch')
            expected_types = dict(Counter(c['attributes']['label']['anomaly'] for c in cases[:count]))
            require(tm['anomaly_types'] == expected_types, 'Anomaly type counts mismatch')
            path = tm_path.parent / tm['file']['filename']
            require(split.file_sha256(path) == tm['file']['sha256'], 'Mixed-test hash mismatch')
            pool.verify_log(path, chain(pool.normal_cases_for_test(record), iter(cases[:count])), metadata)
            reports.append({'dataset': dataset, 'ratio': m['requested_observed_ratio'],
                            'seed': m['seed'], 'normal_cases': normal_count, 'anomaly_cases': count,
                            'passed': True})
        del payload, cases, log, records
        gc.collect()
    return {'schema_version': 1, 'base_seed': 42, 'repeat': 1,
            'release_index_sha256': split.file_sha256(root / 'release_index.json'),
            'implementation_sha256': {**pool.implementation_hashes(),
                                      'src/dcad/variant_ratio/release.py': split.file_sha256(Path(__file__))},
            'checks': ['file inventory and hashes', 'source rows and variant membership',
                       'discovery reproducibility and nested observed sets',
                       'historical versus current validation membership and activity coverage',
                       'pool provenance and normal-variant collision rejection',
                       'mixed-test normal cases, pool prefixes, reader and event labels'],
            'splits': reports, 'passed': True}


def compare_rebuilt(rebuilt: Path, reference: Path):
    """Compare data bytes; new manifests correctly have new source-code provenance."""
    index = validate_index(reference)
    data = [e for e in index['files'] if e['path'].endswith(('.csv', '.json.gz'))
            and not e['path'].endswith('summary.csv')]
    expected = {e['path'] for e in data}
    actual = {str(p.relative_to(rebuilt)) for p in rebuilt.rglob('*') if p.is_file()
              and (p.suffix == '.csv' or p.name.endswith('.json.gz')) and p.name != 'summary.csv'}
    require(actual == expected, 'Rebuilt data inventory differs from the release')
    for entry in data:
        require(split.file_sha256(rebuilt / entry['path']) == entry['sha256'],
                f"Rebuilt data differs: {entry['path']}")
    print(f'Byte-identical rebuilt data: {len(data)} files', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=split.ROOT / 'data/variant_ratio')
    parser.add_argument('--raw-dir', type=Path, default=split.ROOT / 'data/processed/raw')
    parser.add_argument('--xes-dir', type=Path, default=split.ROOT / 'data/original/real')
    parser.add_argument('--report', type=Path, help='Write a new audit report; otherwise validation is read-only.')
    parser.add_argument('--compare-rebuilt', type=Path, help='Compare rebuilt data bytes against this release.')
    args = parser.parse_args(argv)
    if args.compare_rebuilt:
        compare_rebuilt(args.compare_rebuilt.resolve(), args.data_dir.resolve())
        return
    report = validate_release(args.data_dir.resolve(), args.raw_dir.resolve(), args.xes_dir.resolve())
    if args.report:
        split.atomic_json(args.report, report)
    print('Validated 35 splits, 7 fixed anomaly pools and 35 mixed tests.', flush=True)


if __name__ == '__main__':
    main()
