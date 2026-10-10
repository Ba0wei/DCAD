from pathlib import Path

import pandas as pd
import pm4py


BASE_DIR = Path(__file__).resolve().parents[1]
# LOG_NAMES = [
#     "wide",
#     "medium",
#     "large",
#     "huge",
#     "gigantic",
#     "BPIC12",
#     "BPIC17",
#     "BPIC15_1",
#     "BPIC15_2",
#     "BPIC15_3",
#     "BPIC15_4",
#     "BPIC15_5",
# ]

LOG_NAMES = [
    "BPIC15_1",
    "BPIC15_2",
    "BPIC15_3",
    "BPIC15_4",
    "BPIC15_5",
]

def assign_cost_levels(costs: pd.Series) -> pd.Series:
    ranked = costs.sort_values(kind="mergesort").index
    levels = pd.Series(index=costs.index, dtype="object")
    n = len(ranked)
    low_end = int(n * 0.25)
    high_start = int(n * 0.75)
    levels.loc[ranked[:low_end]] = "low"
    levels.loc[ranked[low_end:high_start]] = "medium"
    levels.loc[ranked[high_start:]] = "high"
    return levels


def compute_log_alignment_costs(log_name: str, csv_path=None, pnml_path=None, output_path=None) -> None:
    csv_path = Path(csv_path) if csv_path is not None else BASE_DIR / "data" / "processed" / "splits" / f"{log_name}_variant_train.csv"
    pnml_path = Path(pnml_path) if pnml_path is not None else BASE_DIR / "data" / "pnml" / f"{log_name}_imf.pnml"
    output_path = Path(output_path) if output_path is not None else BASE_DIR / "outputs" / "alignment_costs" / f"{log_name}_train_alignment_cost.csv"

    df = pd.read_csv(csv_path)
    log = pm4py.format_dataframe(
        df,
        case_id="case_id",
        activity_key="name",
        timestamp_key="timestamp",
    )
    net, im, fm = pm4py.read_pnml(str(pnml_path))
    costs = pm4py.conformance_diagnostics_alignments(
        log,
        net,
        im,
        fm,
        return_diagnostics_dataframe=True,
    )[["case_id", "cost"]].rename(columns={"cost": "raw_alignment_cost"})

    min_cost = costs["raw_alignment_cost"].min()
    max_cost = costs["raw_alignment_cost"].max()
    costs["alignment_cost"] = (
        (costs["raw_alignment_cost"] - min_cost) / (max_cost - min_cost or 1.0)
    )
    costs["cost_level"] = assign_cost_levels(costs["alignment_cost"])
    costs = costs[["case_id", "alignment_cost", "raw_alignment_cost", "cost_level"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    costs.to_csv(output_path, index=False)
    print(f"{log_name}: wrote {len(costs)} traces to {output_path}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=LOG_NAMES)
    parser.add_argument('--csv-path', type=Path)
    parser.add_argument('--pnml-path', type=Path)
    parser.add_argument('--output-path', type=Path)
    args = parser.parse_args()
    if any((args.csv_path, args.pnml_path, args.output_path)) and len(args.datasets) != 1:
        parser.error('Explicit paths require exactly one dataset.')
    for log_name in args.datasets:
        compute_log_alignment_costs(log_name, args.csv_path, args.pnml_path, args.output_path)


if __name__ == "__main__":
    main()
