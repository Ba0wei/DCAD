"""
CLI usage:
单个文件：
python3 export_csv_logs.py generator/real-life_Logs/BPIC17_Offer_log.xes.gz
批量模式：
python3 export_csv_logs.py --only raw
python3 export_csv_logs.py temp_eventlog2csv/*.json.gz
"""

import argparse
import csv
import gzip
import os
from pathlib import Path

from dcad.fs import CSV_LOG_DIR

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_LOG_DIR = PROJECT_DIR / "data" / "original" / "real"
ANOMALOUS_LOG_DIR = PROJECT_DIR / "data" / "processed" / "custom_test"

RAW_SUFFIXES = (".xes", ".xes.gz")
ANOMALOUS_SUFFIXES = (".json", ".json.gz")
IGNORED_GLOBAL_EVENT_KEYS = {
    "concept:name",
    "time:timestamp",
    "lifecycle:transition",
    "EventID",
    "activityNameEN",
    "activityNameNL",
    "dateFinished",
    "question",
    "product",
    "EventOrigin",
    "Action",
    "organization involved",
    "impact",
    "concept:instance",
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def csv_name_for(path):
    name = path.name
    if name.endswith(".xes.gz"):
        return name[:-7] + ".csv"
    if name.endswith(".json.gz"):
        return name[:-8] + ".csv"
    if name.endswith(".xes"):
        return name[:-4] + ".csv"
    if name.endswith(".json"):
        return name[:-5] + ".csv"
    return path.stem + ".csv"


def export_logs(source_dir, target_dir, suffixes, loader):
    ensure_dir(target_dir)
    exported = []
    for path in sorted(source_dir.iterdir()):
        if not path.is_file():
            continue
        if not any(path.name.endswith(suffix) for suffix in suffixes):
            continue
        event_log = loader(path)
        csv_path = Path(target_dir) / csv_name_for(path)
        event_log.save_csv(str(csv_path))
        exported.append(csv_path)
    return exported


def open_xes(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


def local_name(element):
    from lxml import etree

    return etree.QName(element).localname


def clear_element(element):
    element.clear()
    parent = element.getparent()
    while parent is not None and element.getprevious() is not None:
        del parent[0]


def parse_xes_attributes(element):
    attrs = {}
    for child in element:
        key = child.attrib.get("key")
        if key is None or key.startswith("_"):
            continue
        attrs[key] = child.attrib.get("value", "")
    return attrs


def read_xes_metadata(path):
    from lxml import etree

    classifiers = []
    global_event_keys = []
    event_keys = []
    seen_global_event_keys = set()
    seen_event_keys = set()

    with open_xes(path) as infile:
        for _, element in etree.iterparse(infile, events=("end",)):
            tag = local_name(element)

            if tag == "classifier" and element.attrib.get("keys"):
                classifiers.append(
                    {
                        "name": element.attrib.get("name", ""),
                        "keys": element.attrib.get("keys", "").split(),
                    }
                )
                clear_element(element)
            elif tag == "global" and element.attrib.get("scope") == "event":
                for child in element:
                    key = child.attrib.get("key")
                    if key and not key.startswith("_") and key not in seen_global_event_keys:
                        seen_global_event_keys.add(key)
                        global_event_keys.append(key)
                clear_element(element)
            elif tag == "event":
                for key in parse_xes_attributes(element):
                    if key not in seen_event_keys:
                        seen_event_keys.add(key)
                        event_keys.append(key)
                clear_element(element)
            elif tag in {"extension", "trace"}:
                clear_element(element)

    primary_event_keys = [
        key for key in global_event_keys if key not in IGNORED_GLOBAL_EVENT_KEYS
    ]
    if not global_event_keys:
        primary_event_keys = event_keys

    extra_event_keys = [
        key for key in event_keys if key not in primary_event_keys and key != "name"
    ]
    columns = ["case_id", "event_position", "name", "timestamp"]
    columns.extend(key for key in primary_event_keys if key not in columns)
    columns.extend(key for key in extra_event_keys if key not in columns)
    return classifiers, columns, primary_event_keys


def event_name_from_classifier(event_attrs, classifiers, warned_missing_keys):
    if not classifiers:
        return ""

    keys = classifiers[0]["keys"]
    missing_keys = tuple(key for key in keys if key not in event_attrs)
    if missing_keys:
        if missing_keys not in warned_missing_keys:
            print(f'Classifier key(s) {", ".join(missing_keys)} could not be found in event.')
            warned_missing_keys.add(missing_keys)
        return None
    return "+".join(event_attrs[key] for key in keys)


def write_symbol_row(writer, case_id, event_position, symbol, primary_event_keys):
    row = {
        "case_id": case_id,
        "event_position": event_position,
        "name": symbol,
        "timestamp": "",
    }
    for key in primary_event_keys:
        row[key] = symbol
    writer.writerow(row)


def export_xes_to_csv(path, csv_path):
    from lxml import etree
    from dcad.processmining.log import EventLog

    classifiers, columns, primary_event_keys = read_xes_metadata(path)
    warned_missing_keys = set()

    with open_xes(path) as infile, open(csv_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()

        case_index = 0
        for _, trace in etree.iterparse(infile, events=("end",)):
            if local_name(trace) != "trace":
                if local_name(trace) in {"extension", "global", "classifier"}:
                    clear_element(trace)
                continue

            trace_attrs = parse_xes_attributes(trace)
            case_id = trace_attrs.get("concept:name")
            if case_id is None:
                case_id = case_index

            event_position = 0
            write_symbol_row(writer, case_id, event_position, EventLog.start_symbol, primary_event_keys)

            for child in trace:
                if local_name(child) != "event":
                    continue
                event_attrs = parse_xes_attributes(child)
                name = event_name_from_classifier(event_attrs, classifiers, warned_missing_keys)
                if name is None:
                    continue

                event_position += 1
                row = {
                    "case_id": case_id,
                    "event_position": event_position,
                    "name": name,
                    "timestamp": event_attrs.get("time:timestamp", ""),
                }
                row.update(event_attrs)
                writer.writerow(row)

            write_symbol_row(
                writer,
                case_id,
                event_position + 1,
                EventLog.end_symbol,
                primary_event_keys,
            )
            case_index += 1
            clear_element(trace)


def export_log_file(path, output_root):
    from dcad.processmining.log import EventLog

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input log file does not exist: {path}")

    if path.name.endswith(RAW_SUFFIXES):
        target_dir = Path(output_root) / "raw"
        loader = None
    elif path.name.endswith(ANOMALOUS_SUFFIXES):
        target_dir = Path(output_root) / "anomalous"
        loader = EventLog.from_json
    else:
        raise ValueError(
            f"Unsupported input log format: {path}. "
            "Expected .xes, .xes.gz, .json, or .json.gz."
        )

    ensure_dir(target_dir)
    csv_path = target_dir / csv_name_for(path)
    if path.name.endswith(RAW_SUFFIXES):
        export_xes_to_csv(path, csv_path)
    else:
        event_log = loader(path)
        event_log.save_csv(str(csv_path))
    return csv_path


def export_raw_logs(output_root):
    target_dir = Path(output_root) / "raw"
    ensure_dir(target_dir)
    exported = []
    for path in sorted(RAW_LOG_DIR.iterdir()):
        if not path.is_file():
            continue
        if not any(path.name.endswith(suffix) for suffix in RAW_SUFFIXES):
            continue
        csv_path = target_dir / csv_name_for(path)
        export_xes_to_csv(path, csv_path)
        exported.append(csv_path)
    return exported


def export_anomalous_logs(output_root):
    from dcad.processmining.log import EventLog

    return export_logs(
        source_dir=ANOMALOUS_LOG_DIR,
        target_dir=Path(output_root) / "anomalous",
        suffixes=ANOMALOUS_SUFFIXES,
        loader=EventLog.from_json,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Export raw and anomalous logs to CSV.")
    parser.add_argument(
        "input_files",
        nargs="*",
        help="Optional specific log file(s) to export, e.g. generator/real-life_Logs/BPIC12.xes.gz.",
    )
    parser.add_argument(
        "--output-dir",
        default=CSV_LOG_DIR,
        help="Target root directory for exported CSV logs.",
    )
    parser.add_argument(
        "--only",
        choices=["all", "raw", "anomalous"],
        default="all",
        help="Choose which logs to export.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    if args.input_files:
        exported = [export_log_file(path, args.output_dir) for path in args.input_files]
        for csv_path in exported:
            print(f"Exported {csv_path}")
        return

    if args.only in ("all", "raw"):
        raw_exported = export_raw_logs(args.output_dir)
        print(f"Exported {len(raw_exported)} raw log(s) to {Path(args.output_dir) / 'raw'}")

    if args.only in ("all", "anomalous"):
        anomalous_exported = export_anomalous_logs(args.output_dir)
        print(f"Exported {len(anomalous_exported)} anomalous log(s) to {Path(args.output_dir) / 'anomalous'}")


if __name__ == "__main__":
    main()
