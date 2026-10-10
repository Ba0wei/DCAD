# Data layout

The project includes twelve logs: `wide`, `medium`, `large`, `huge`, `gigantic`, `BPIC12`, `BPIC15_1`–`BPIC15_5`, and `BPIC17`.

- `data/original/real`: original public BPI Challenge XES logs.
- `data/original/synthetic`: source process models for synthetic logs.
- `data/processed/raw`: event-level CSV exports.
- `data/processed/splits`: trace-variant train/test splits.
- `data/processed/custom_test`: labeled JSON.GZ evaluation logs.
- `data/pnml`: process models used to derive TN-DCAD alignment costs.
- `data/variant_ratio/splits`: normal train/unseen-test CSVs at different observed variant ratios, with split manifests, summaries and a seed mapping.
- `data/variant_ratio/anomaly_pools`: one fixed, labeled anomaly pool per BPIC log, with provenance manifests.
- `data/variant_ratio/mixed_testsets`: unseen normal cases combined with anomalies from the fixed pools, at a normal:anomaly ratio of approximately 2:1.
- `data/variant_ratio/pool_config.json`: fixed pool capacities required to reconstruct the released test sets.
- `data/variant_ratio/release_index.json`: relative file paths, sizes and SHA-256 hashes used to verify the release and compare rebuilt data.

The variant-ratio data cover BPIC12, BPIC15_1–5 and BPIC17 at observed/unseen ratios 20/80, 40/60, 60/40, 80/20 and 90/10: 35 splits using base seed 42, repeat 1. Ratios refer to normal variants, not case counts. These data reuse the original BPIC inputs above; split directories follow `<dataset>/seed_<derived>/observed_<percentage>/`.

Download the data with Git LFS as described in the [project README](../README.md). Construction tools are provided for [normal variant splits](../scripts/split_by_variant_ratio.py) and [fixed anomaly pools and mixed test sets](../scripts/build_variant_ratio_testsets.py); use `--output-dir` to generate data in a separate directory.

Validate the released data from the repository root:

```bash
python scripts/validate_data.py
python scripts/validate_variant_ratio.py
```

Alignment-cost files are not versioned. The TN-DCAD core expects the columns `case_id` and normalized `alignment_cost`; selected computation logic is provided, while the complete generation pipeline is outside this release.

The BPI Challenge logs remain subject to the terms and citation requirements of their original providers. Inclusion here does not relicense those datasets.
