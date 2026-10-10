"""Core alignment-cost computation used by TN-DCAD.

Dataset selection, path resolution, persistence, and command-line orchestration
are intentionally outside this public core module.
"""

from __future__ import annotations

import pandas as pd
import pm4py


def compute_normalized_alignment_costs(event_frame: pd.DataFrame, pnml_path: str) -> pd.DataFrame:
    """Return case-level normalized alignment costs for a prepared event frame."""
    required = {"case_id", "name", "timestamp"}
    if missing := required - set(event_frame.columns):
        raise ValueError(f"Event frame is missing columns: {sorted(missing)}")

    event_log = pm4py.format_dataframe(
        event_frame,
        case_id="case_id",
        activity_key="name",
        timestamp_key="timestamp",
    )
    net, initial_marking, final_marking = pm4py.read_pnml(pnml_path)
    costs = pm4py.conformance_diagnostics_alignments(
        event_log,
        net,
        initial_marking,
        final_marking,
        return_diagnostics_dataframe=True,
    )[["case_id", "cost"]].rename(columns={"cost": "raw_alignment_cost"})

    minimum = costs["raw_alignment_cost"].min()
    maximum = costs["raw_alignment_cost"].max()
    costs["alignment_cost"] = (
        costs["raw_alignment_cost"] - minimum
    ) / (maximum - minimum or 1.0)
    return costs[["case_id", "alignment_cost", "raw_alignment_cost"]]
