# DCAD: Masked Diffusion for Control-Flow Anomaly Detection

Implementation of DCAD, EM-DCAD and TN-DCAD for anomaly detection on unseen process variants.

**Paper:** *Control-Flow Anomaly Detection on Unseen Process Variants via Masked Diffusion*

## Setup

```bash
git lfs install
git lfs pull
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The migration was checked with Python 3.10, NumPy 2.2.6, pandas 2.3.3, PyTorch 2.12.0+cu130, PM4Py 2.7.22.3, lxml 6.1.1 and NetworkX 3.4.2. Install a PyTorch build suitable for your hardware; CPU execution is supported.

## Variant-ratio experiments

The [data guide](data/README.md) describes the 35 released splits: seven BPIC logs, five observed/unseen normal-variant ratios, base seed 42 and repeat 1.

Run from the repository root:

```bash
python scripts/run_bpic12_variant_ratio.py --stage formal
python scripts/run_bpic15_variant_ratio.py --stage formal
python scripts/run_bpic17_variant_ratio.py --stage formal
```

Each command trains DCAD, EM-DCAD and TN-DCAD at all five ratios, saves the best validation checkpoint, scores the mixed test sets and writes trace/event F1 and AUPR to `results.csv` under `outputs/variant_ratio/`. BPIC15 processes all five logs; select individual logs with `--datasets BPIC15_2`. Use `--stage check` for a read-only input check, `--stage smoke` for one DCAD epoch at 80/20, or `--resume` to continue an interrupted run. Existing outputs are protected unless `--overwrite` is specified.

The runners encode the experiment settings: at most 500 epochs, batch size 32, validation ratio 0.1, patience 50, training/mask seed 2026, validation-mask seed 102026, and inference seed 1 with five time samples and five mask samples. DCAD/EM-DCAD use time range [0, 0.6] and adaptive weights 0/0.1. TN-DCAD uses [0.05, 0.5], beta 0.5 and costs grouped by stable rank at 25%/75%. Each ratio's TN reference model and alignment costs are generated from its outer training candidates, including internal validation cases. The reference selection targets 25% case coverage and uses inductive-miner noise threshold 1.

AUPR is average precision; F1 is the maximum over the test-set precision-recall curve. Trace scores are maximum event NLL; event metrics exclude boundary tokens and padding. Hardware and dependency versions can change numerical results.

## Individual training and TN alignment

`dcad/em_dcad.py` contains the full DCAD/EM-DCAD trainer, `dcad/tn_dcad.py` the TN trainer, and `dcad/scoring.py` the experiment inference and metric implementation. The model is defined in `dcad/model.py`.

For the main-experiment data, generate TN reference models and alignment costs with:

```bash
python scripts/Inductive_Miner_Infrequent.py --datasets BPIC12
python scripts/compute_train_alignment_costs.py --datasets BPIC12
```

Train on the released main split and score its mixed test set:

```bash
python scripts/train_dcad.py --event-log BPIC12 --adaptive-weight 0 --save-path outputs/BPIC12/DCAD
python scripts/train_dcad.py --event-log BPIC12 --adaptive-weight 0.1 --save-path outputs/BPIC12/EM-DCAD
python scripts/train_tn_dcad.py --event-log BPIC12 --save-path outputs/BPIC12/TN-DCAD
python scripts/infer_anomaly_scores.py --model-path outputs/BPIC12/DCAD/model.pt --vocab-path outputs/BPIC12/DCAD/vocab.json --config-path outputs/BPIC12/DCAD/config.json --dataset BPIC12_custom_test
```

These individual commands use the migrated trainers' defaults. Exact main-table, baseline and ablation configurations are not included in this release. The complete published batch protocol covers the repeat-1 variant-ratio experiments.

## Data preparation

Data conversion, variant splitting, anomaly injection and test-set construction tools are included. See [data/README.md](data/README.md) for construction and validation commands. Historical manifest paths and code hashes record the original provenance; runtime paths resolve within this checkout, and runners verify data hashes and validation membership with the current implementation.

## License

No license is granted for this repository at present. Third-party datasets and inherited components remain subject to their original terms and notices.
