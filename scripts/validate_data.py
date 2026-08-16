#!/usr/bin/env python3
"""Validate the twelve-dataset release layout and basic file readability."""
import csv
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("wide", "medium", "large", "huge", "gigantic", "BPIC12", "BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5", "BPIC17")
REAL = {name for name in DATASETS if name.startswith("BPIC")}

def require(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty data file: {path}")

def csv_header(path: Path) -> set[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return set(next(csv.reader(handle)))

def main() -> None:
    for name in DATASETS:
        original = ROOT / "data" / "original" / ("real" if name in REAL else "synthetic") / (f"{name}.xes.gz" if name in REAL else f"{name}.plg")
        raw_name = f"{name}.csv" if name in REAL else f"{name}-0.00-1.csv"
        raw = ROOT / "data" / "processed" / "raw" / raw_name
        train = ROOT / "data" / "processed" / "splits" / f"{name}_variant_train.csv"
        test = ROOT / "data" / "processed" / "splits" / f"{name}_variant_test.csv"
        custom = ROOT / "data" / "processed" / "custom_test" / f"{name}_custom_test.json.gz"
        pnml = ROOT / "data" / "pnml" / f"{name}_imf.pnml"
        for path in (original, raw, train, test, custom, pnml): require(path)
        for path in (raw, train, test):
            missing = {"case_id", "name"} - csv_header(path)
            if missing: raise ValueError(f"{path} is missing {sorted(missing)}")
        with gzip.open(custom, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        cases = payload.get("cases", payload.get("traces"))
        if not isinstance(cases, list) or not cases: raise ValueError(f"No cases in {custom}")
        print(f"{name}: OK ({len(cases)} evaluation cases)")

if __name__ == "__main__": main()
