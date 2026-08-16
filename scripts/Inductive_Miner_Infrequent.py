import math
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pm4py


START_ACTIVITY = "__START__"
END_ACTIVITY = "__END__"
REFERENCE_COVERAGE_RATIO = 0.25
NOISE_THRESHOLD = 1
BASE_DIR = Path(__file__).resolve().parents[1]
# LOG_NAMES: list[str] = [
#     "wide",
#     "medium",
#     "large",
#     "huge",
#     "gigantic",
#     "BPIC12",
#     "BPIC15_1",
#     "BPIC15_2",
#     "BPIC15_3",
#     "BPIC15_4",
#     "BPIC15_5",
#     "BPIC17"
# ]

LOG_NAMES: list[str] = [
    "BPIC15_1",
    "BPIC15_2",
    "BPIC15_3",
    "BPIC15_4",
    "BPIC15_5",
]


def train_csv_path(log_name: str) -> Path:
    return BASE_DIR / "data" / "processed" / "splits" / f"{log_name}_variant_train.csv"


def output_pnml_path(log_name: str) -> Path:
    return BASE_DIR / "data" / "pnml" / f"{log_name}_imf.pnml"


def build_variants(
    df: pd.DataFrame,
) -> tuple[dict[object, int], list[tuple[str, ...]], list[int]]:
    case_variants = df.groupby("case_id", sort=False)["name"].agg(tuple)
    variant_to_id: dict[tuple[str, ...], int] = {}
    case_to_variant_id: dict[object, int] = {}
    variant_sequences: list[tuple[str, ...]] = []
    variant_frequencies: list[int] = []

    for case_id, variant in case_variants.items():
        if variant not in variant_to_id:
            variant_to_id[variant] = len(variant_sequences)
            variant_sequences.append(variant)
            variant_frequencies.append(0)

        variant_id = variant_to_id[variant]
        case_to_variant_id[case_id] = variant_id
        variant_frequencies[variant_id] += 1

    return case_to_variant_id, variant_sequences, variant_frequencies


def variant_edges(variant: tuple[str, ...]) -> list[tuple[str, str]]:
    activities = (START_ACTIVITY, *variant, END_ACTIVITY)
    return list(zip(activities, activities[1:]))


def dfg_centrality_scores(
    variant_sequences: list[tuple[str, ...]],
    variant_frequencies: list[int],
) -> list[float]:
    edge_counts: dict[tuple[str, str], int] = defaultdict(int)
    source_totals: dict[str, int] = defaultdict(int)
    unique_activities = {
        activity
        for variant in variant_sequences
        for activity in variant
    }
    alpha = 1.0
    num_targets = len(unique_activities) + 1

    for variant_id, variant in enumerate(variant_sequences):
        frequency = variant_frequencies[variant_id]
        for source, target in variant_edges(variant):
            edge_counts[(source, target)] += frequency
            source_totals[source] += frequency

    scores: list[float] = []
    for variant in variant_sequences:
        edge_scores = []
        for source, target in variant_edges(variant):
            probability = (
                (edge_counts[(source, target)] + alpha)
                / (source_totals[source] + alpha * num_targets)
            )
            edge_scores.append(-math.log(probability))
        scores.append(sum(edge_scores) / len(edge_scores))

    return scores


def select_reference_variants(
    log_name: str,
    variant_sequences: list[tuple[str, ...]],
    variant_frequencies: list[int],
    total_cases: int,
) -> list[int]:
    target_case_count = math.ceil(total_cases * REFERENCE_COVERAGE_RATIO)

    if log_name.startswith("BPIC15"):
        scores = dfg_centrality_scores(variant_sequences, variant_frequencies)
        ranked_variant_ids = sorted(
            range(len(variant_sequences)),
            key=lambda variant_id: (
                scores[variant_id],
                -variant_frequencies[variant_id],
                variant_id,
            ),
        )
    else:
        ranked_variant_ids = sorted(
            range(len(variant_sequences)),
            key=lambda variant_id: (-variant_frequencies[variant_id], variant_id),
        )

    selected_variant_ids: list[int] = []
    selected_case_count = 0
    for variant_id in ranked_variant_ids:
        selected_variant_ids.append(variant_id)
        selected_case_count += variant_frequencies[variant_id]
        if selected_case_count >= target_case_count:
            break

    return selected_variant_ids


def run_log(log_name: str) -> None:
    csv_path = train_csv_path(log_name)
    pnml_path = output_pnml_path(log_name)

    df = pd.read_csv(csv_path)
    case_to_variant_id, variant_sequences, variant_frequencies = build_variants(df)
    reference_variant_ids = select_reference_variants(
        log_name=log_name,
        variant_sequences=variant_sequences,
        variant_frequencies=variant_frequencies,
        total_cases=len(case_to_variant_id),
    )

    reference_set = set(reference_variant_ids)
    reference_case_ids = {
        case_id
        for case_id, variant_id in case_to_variant_id.items()
        if variant_id in reference_set
    }
    reference_df = df[df["case_id"].isin(reference_case_ids)].copy()
    log = pm4py.format_dataframe(
        reference_df,
        case_id="case_id",
        activity_key="name",
        timestamp_key="timestamp",
    )

    net, im, fm = pm4py.discover_petri_net_inductive(
        log,
        noise_threshold=NOISE_THRESHOLD,
    )
    pnml_path.parent.mkdir(parents=True, exist_ok=True)
    pm4py.write_pnml(
        net,
        im,
        fm,
        str(pnml_path),
    )

    print(f"Log: {log_name}")
    print(f"Places: {len(net.places)}")
    print(f"Transitions: {len(net.transitions)}")
    print(f"Arcs: {len(net.arcs)}")


def main() -> None:
    for log_name in LOG_NAMES:
        run_log(log_name)


if __name__ == "__main__":
    main()
