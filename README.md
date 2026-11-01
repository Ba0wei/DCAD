# DCAD: Masked Diffusion for Control-Flow Anomaly Detection on Unseen Process Variants 

Implementation of DCAD, EM-DCAD and TN-DCAD for anomaly detection on unseen process variants.

**Paper:** *Control-Flow Anomaly Detection on Unseen Process Variants via Masked Diffusion*

## Setup

```bash
git lfs install
git lfs pull
pip install -e .
```

Run the following commands from the repository root.

## Data

`data/processed/` contains the twelve logs and their train/test splits and labeled test sets. `data/variant_ratio/` contains 35 splits for seven BPIC logs at five observed/unseen variant ratios. See [data/README.md](data/README.md) for the layout and construction tools.

## Training

Example using BPIC12:

```bash
python scripts/train_dcad.py --event-log BPIC12 --adaptive-weight 0 --save-path outputs/BPIC12/DCAD
python scripts/train_dcad.py --event-log BPIC12 --adaptive-weight 0.1 --save-path outputs/BPIC12/EM-DCAD
python scripts/compute_train_alignment_costs.py --datasets BPIC12
python scripts/train_tn_dcad.py --event-log BPIC12 --save-path outputs/BPIC12/TN-DCAD
```

Defaults: DCAD/EM `t_max=0.6`; EM λ=0.1, γ=3; TN T=0.5, β=0.5. Training/mask seed is 2026 and validation-mask seed is 102026. TN uses the released reference models; rebuild them with `scripts/Inductive_Miner_Infrequent.py` if needed.

For the variant-ratio data, use `scripts/run_bpic12_variant_ratio.py`, `run_bpic15_variant_ratio.py` or `run_bpic17_variant_ratio.py` with `--stage formal`. These runners include TN alignment, training and evaluation.

## Evaluation

```bash
python scripts/infer_anomaly_scores.py \
  --model-path outputs/BPIC12/DCAD/model.pt \
  --vocab-path outputs/BPIC12/DCAD/vocab.json \
  --config-path outputs/BPIC12/DCAD/config.json \
  --dataset BPIC12_custom_test
```

Use the corresponding artifacts for other methods and logs. Defaults: inference seed 1, five time samples, five mask samples, `multi_t` and `actual_mask_ratio`; `t_max` is read from the saved configuration. Outputs include trace/event F1 and AUPR. Baseline and ablation configurations are outside this release.

## License

No license is granted for this repository at present. Third-party datasets and inherited components remain subject to their original terms and notices.
