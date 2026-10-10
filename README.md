# DCAD: Masked Diffusion for Control-Flow Anomaly Detection

This repository provides the implementation of **DCAD**, a masked diffusion framework for control-flow anomaly detection on unseen process variants. It includes the code for DCAD and its two adaptive variants, **EM-DCAD** and **TN-DCAD**, together with the processed datasets used in our experiments.

**Paper:** *Control-Flow Anomaly Detection on Unseen Process Variants via Masked Diffusion*

## Setup

```bash
git lfs install
git lfs pull
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
```

## Core Components

- **DCAD:** masked diffusion modeling for control-flow anomaly detection.
- **EM-DCAD:** event-aware adaptive masking based on transition rarity.
- **TN-DCAD:** trace-aware noise scheduling based on process conformance information.
- **Training and evaluation:** batch-level optimization, adaptive masking, trace-weighted objectives, and anomaly scoring.
- **Data processing:** utilities for preparing event logs and experimental datasets.

The training components include deterministic seed control and reproducibility settings. See [the data guide](data/README.md) for the dataset layout.

The main implementations are organized in `dcad/model.py`, `dcad.py`, `em_dcad.py`, `tn_dcad.py`, and `scoring.py`.

## Data preparation

The repository includes routines for XES/CSV conversion, trace-variant splitting, anomaly injection, custom-test construction, process-model discovery, and alignment-cost computation.

The main-experiment dataset files can be checked with `python scripts/validate_data.py`.

The [variant-ratio experiments](data/README.md#data-layout) include 35 normal splits and labeled mixed test sets for seven BPIC logs at five observed/unseen variant ratios (base seed 42, repeat 1), together with fixed anomaly pools and standalone data construction tools. Validate them with `python scripts/validate_variant_ratio.py`.

## License

No license is granted for this repository at present. Third-party datasets and inherited components remain subject to their original terms and notices.
