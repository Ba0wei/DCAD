# Data layout

The project includes twelve logs: `wide`, `medium`, `large`, `huge`, `gigantic`, `BPIC12`, `BPIC15_1`–`BPIC15_5`, and `BPIC17`.

- `data/original/real`: original public BPI Challenge XES logs.
- `data/original/synthetic`: source process models for synthetic logs.
- `data/processed/raw`: event-level CSV exports.
- `data/processed/splits`: trace-variant train/test splits.
- `data/processed/custom_test`: labeled JSON.GZ evaluation logs.
- `data/pnml`: process models used to derive TN-DCAD alignment costs.

Alignment-cost files are not versioned. The TN-DCAD core expects the columns `case_id` and normalized `alignment_cost`; selected computation logic is provided, while the complete generation pipeline is outside this release.

The BPI Challenge logs remain subject to the terms and citation requirements of their original providers. Inclusion here does not relicense those datasets.

## Variant-ratio experiments

These experiments measure detection performance as the fraction of observed normal
process variants changes. The release includes **35 splits**: BPIC12, BPIC15_1–5,
and BPIC17, each at observed/unseen ratios 20/80, 40/60, 60/40, 80/20 and 90/10.
All use **base seed 42, repeat 1**, corresponding to the existing experiment result
tables. Other repetitions, training runs and result plots are outside this data release.

### Download and layout

Run `git lfs install` and `git lfs pull` after cloning. CSVs, compressed logs and
large provenance manifests use Git LFS; indexes and summaries use ordinary Git.
The additional data occupy approximately **1.83 GiB** when downloaded.

```text
data/variant_ratio/
├── release_index.json          # relative paths, sizes and SHA-256 for release files
├── audit_report.json           # checks performed with the published implementation
├── pool_config.json            # fixed anomaly-pool capacities and base seed
├── splits/
│   ├── summary.csv
│   ├── seed_mapping.json
│   └── <dataset>/seed_<derived>/observed_<percentage>/
│       ├── <dataset>_variant_train.csv
│       ├── <dataset>_variant_test.csv
│       └── split_manifest.json
├── anomaly_pools/<dataset>/
│   ├── <dataset>_anomaly_pool.json.gz
│   └── pool_manifest.json
└── mixed_testsets/
    ├── summary.csv
    └── <dataset>/seed_<derived>/observed_<percentage>/
        ├── <dataset>_repeat_1_seed_<derived>_observed_<percentage>_custom_test.json.gz
        └── test_manifest.json
```

The original BPIC inputs are reused from `data/processed/raw` and
`data/original/real`. The train CSV contains only normal cases; its paired test
CSV contains only unseen normal cases. Use the **mixed JSON.GZ** for anomaly
assessment: it includes those unseen normal cases and labeled anomalies in the
existing `attributes/cases/events` format.

### Split protocol and seeds

A variant is the complete `name` activity sequence after stripping whitespace
and excluding empty values and `▶`, `■`, `START`, `END`. Original CSV fields,
rows and event order are preserved. After a seeded trace shuffle, a greedy
startup set covers all normal activities; ties follow shuffled trace order.
Remaining traces are shuffled again. Variants are ordered by first discovery,
and each ratio selects the first `round(ratio * total_variants)` variants
(Python rounding). Case frequency therefore affects discovery order. This is
not uniform sampling of variants or an optimization for a training-case ratio.
All cases belonging to a selected variant enter the training candidate set.
The five observed sets are nested, and train/test cases and variants are disjoint.
An infeasible greedy activity-cover budget produces an error.

The base seed is **not** the actual random seed used for each dataset:

```python
seed = int.from_bytes(
    hashlib.sha256(f"trace_order_v1|{dataset}|42|1".encode()).digest()[:4], "big"
)
```

| Dataset | Derived split seed |
|---|---:|
| BPIC12 | 307694762 |
| BPIC15_1 | 1649758743 |
| BPIC15_2 | 3572032885 |
| BPIC15_3 | 1317870673 |
| BPIC15_4 | 1466331127 |
| BPIC15_5 | 3437344897 |
| BPIC17 | 380423602 |

The internal validation target is 10% of cases, held out by whole variant groups
with seed 2026 and no sequence truncation. The actual fraction can differ because
of variant sizes and activity-coverage constraints. Audits use the model reader,
which retains nonempty boundary tokens, and check tokens present in actual
gradient-training sequences, rather than vocabulary membership. Full normal-test
and validation sequences must contain no unknown training tokens.
`actual_observed_ratio` describes the **outer training candidate set**;
`gradient_train_variant_coverage` describes coverage after validation holdout.

### Fixed anomaly pools

Each dataset has one shared, ordered anomaly pool. Mothers are sampled uniformly
with replacement from the original normal cases; the five injectors are selected
equiprobably: SkipSequence (maximum length 2), Rework/Early/Late (maximum length
and distance 5), and Insert (maximum 5 events). Candidates with normal labels,
empty sequences, or sequences equal to any original normal variant are rejected.
Accepted type frequencies can therefore differ from 20%. Insert may introduce
`Random activity ...` tokens; these are outside the normal activity-coverage rule.
Duplicate anomaly variants are allowed, but pool case IDs are unique.

Each mixed test retains all unseen normal cases and appends the first
`ceil(normal_case_count / 2)` anomaly cases. Thus normal:anomaly is approximately
**2:1**, independently of the 10% validation parameter. Smaller anomaly selections
are prefixes of larger ones. Generation and ordering seeds are derived separately
from `anomaly_pool_v1|<dataset>|42|generation` and `...|order` using the same SHA-256
procedure as above.

The released pools retain their original capacities, which were calculated over
all three research repetitions. `pool_config.json` fixes these capacities even
though only repeat 1 is published. Reducing a pool to the minimum needed by these
35 splits would change its shuffle and the experimental test sets.

### Validation and rebuilding

After the installation described in the README, run from the repository root:

```bash
python scripts/validate_variant_ratio.py
# Optional: save a fresh audit outside the release directory.
python scripts/validate_variant_ratio.py --report /tmp/variant_ratio_audit.json
# Only recheck normal splits and validation activity coverage:
python scripts/split_by_variant_ratio.py --audit-only
```

The full validator checks the release inventory and hashes, original row values,
case and variant membership, deterministic discovery, validation membership,
activity coverage, pool provenance, normal-variant collision rejection, mixed-test
contents and event labels. It resolves data through the release layout, not through
historical machine paths.

Rebuild into a separate directory to preserve the published artifacts:

```bash
python scripts/split_by_variant_ratio.py --output-dir /tmp/dcad-ratio-rebuild/splits
python scripts/build_variant_ratio_testsets.py \
  --split-dir /tmp/dcad-ratio-rebuild/splits \
  --output-dir /tmp/dcad-ratio-rebuild
python scripts/validate_variant_ratio.py --compare-rebuilt /tmp/dcad-ratio-rebuild
```

Defaults select seven datasets, five ratios, base seed 42 and repeat 1. Use
`--datasets`, `--observed-ratios`, `--base-seed`, `--repeats`, `--input-dir`,
`--validation-ratio`, `--validation-seed` and `--max-len` to configure normal splits.
The mixed-test builder accepts `--raw-dir`, `--xes-dir`, `--split-dir`,
`--output-dir` and `--pool-config`; custom seed/capacity settings require a matching
pool config. Both builders refuse existing outputs unless `--overwrite` is given.
The pool builder stages outputs and marks interrupted publication; recover using
the same settings and `--overwrite`. It rebuilds the pool and its mixed tests together.

Existing source manifests are preserved byte-for-byte, including historical
absolute paths and source-code hashes. They document the original experiment;
those paths are not runtime dependencies. `release_index.json` supplies portable
locations, and `audit_report.json` identifies the current validation implementation.
Rebuilt manifests correctly record new paths and implementation hashes, while
`--compare-rebuilt` compares the actual CSV and JSON.GZ data bytes. It does not
require historical and newly generated manifests to be identical.

The data tools require no GPU or training pipeline. The reconstruction environment uses
Python 3.10.12, NumPy 2.2.6, PyTorch 2.12.0+cu130, PM4Py 2.7.22.3,
pandas 2.3.3 and lxml 6.1.1. Dependency versions can affect anomaly generation,
so use the published data and hashes as the reference. The package
continues to expose algorithm cores rather than complete training/benchmark runners.
Original dataset terms and citation requirements above also apply here.
