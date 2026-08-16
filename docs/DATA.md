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
