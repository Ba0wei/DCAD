import argparse
import gzip
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

import sys
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.append(str(SCRIPT_DIR))
sys.path.append(str(PROJECT_DIR))

from dcad.generation.anomaly import EarlyAnomaly
from dcad.generation.anomaly import InsertAnomaly
from dcad.generation.anomaly import LateAnomaly
from dcad.generation.anomaly import NoneAnomaly
from dcad.generation.anomaly import ReworkAnomaly
from dcad.generation.anomaly import SkipSequenceAnomaly
from dcad.generation.attribute_generator import CategoricalAttributeGenerator
from dcad.processmining.case import Case
from dcad.processmining.event import Event
from dcad.processmining.log import EventLog
from dcad.fs import EVENTLOG_DIR

DEFAULT_ANOMALY_P = [0.1]
# REAL_LIFE_LOG_DIR = Path(__file__).parent / "real-life_Logs"
REAL_LIFE_LOG_DIR = PROJECT_DIR / "data" / "original" / "real"
XES_ATTRIBUTE_TAGS = {"string", "date", "int", "float", "boolean", "id", "list", "container"}


class EventLogStats:
    def __init__(self, attributes, unique_activities, unique_attribute_values, num_cases):
        self.attributes = attributes
        self.unique_activities = unique_activities
        self.unique_attribute_values = unique_attribute_values
        self.num_cases = num_cases

def get_log_files(path=None):
    log_dir = Path(REAL_LIFE_LOG_DIR) if path is None else Path(path)
    return sorted(
        str(log_dir / file_name)
        for file_name in os.listdir(log_dir)
        if file_name.endswith((".xes", ".xes.gz"))
    )


def build_control_flow_anomalies(event_log):
    # anomalies = [
    #     SkipSequenceAnomaly(max_sequence_size=2),
    #     ReworkAnomaly(max_distance=5, max_sequence_size=3),
    #     EarlyAnomaly(max_distance=5, max_sequence_size=2),
    #     LateAnomaly(max_distance=5, max_sequence_size=2),
    #     InsertAnomaly(max_inserts=2),
    # ]

    anomalies = [
        SkipSequenceAnomaly(max_sequence_size=2),
        ReworkAnomaly(max_distance=5, max_sequence_size=5),
        EarlyAnomaly(max_distance=5, max_sequence_size=5),
        LateAnomaly(max_distance=5, max_sequence_size=5),
        InsertAnomaly(max_inserts=5),
    ]

    event_attributes = [
        CategoricalAttributeGenerator(name=name, values=values)
        for name, values in event_log.unique_attribute_values.items()
        if name != "name"
    ]

    for anomaly in anomalies:
        anomaly.activities = event_log.unique_activities
        anomaly.attributes = event_attributes

    return anomalies


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
        if key is not None:
            attrs[key] = child.attrib.get("value", "")
    return attrs


def parse_xes_attribute(attribute):
    key = attribute.attrib.get("key")
    if key is None:
        return None, None

    nested_attributes = [parse_xes_attribute(child) for child in attribute]
    nested_attributes = [(k, v) for k, v in nested_attributes if k is not None]
    parsed = {
        "type": local_name(attribute),
        "value": attribute.attrib.get("value", ""),
    }
    if nested_attributes:
        parsed["attr"] = dict(nested_attributes)
    return key, parsed


def event_name_from_classifier(event_attrs, classifiers, warned_missing_keys=None):
    if not classifiers:
        return ""

    keys = classifiers[0]["keys"]
    missing_keys = tuple(key for key in keys if key not in event_attrs)
    if missing_keys:
        if warned_missing_keys is not None and missing_keys not in warned_missing_keys:
            print(f'Classifier key(s) {", ".join(missing_keys)} could not be found in event.')
            warned_missing_keys.add(missing_keys)
        return None
    return "+".join(event_attrs[key] for key in keys)


def parse_xes_header(path):
    from lxml import etree

    extensions = []
    global_attributes = {}
    classifiers = []
    log_attributes = {}

    with open_xes(path) as infile:
        for _, element in etree.iterparse(infile, events=("end",)):
            tag = local_name(element)
            parent = element.getparent()
            parent_tag = local_name(parent) if parent is not None else None

            if tag == "trace":
                clear_element(element)
                break

            if parent_tag != "log":
                continue

            if tag == "extension":
                extensions.append(dict(element.attrib))
                clear_element(element)
            elif tag == "global":
                scope = element.attrib["scope"]
                global_attributes[scope] = {}
                for attribute in element:
                    key = attribute.attrib.get("key")
                    if key is None:
                        continue
                    global_attributes[scope][key] = {
                        "type": local_name(attribute),
                        "value": attribute.attrib.get("value", ""),
                    }
                clear_element(element)
            elif tag == "classifier":
                classifiers.append(
                    {
                        "name": element.attrib.get("name", ""),
                        "keys": element.attrib.get("keys", "").split(),
                    }
                )
                clear_element(element)
            elif tag in XES_ATTRIBUTE_TAGS:
                key, attribute = parse_xes_attribute(element)
                if key is not None:
                    log_attributes[key] = attribute
                clear_element(element)

    log_attributes["extensions"] = extensions
    log_attributes["global_attributes"] = global_attributes
    log_attributes["classifiers"] = classifiers
    return log_attributes


def event_attribute_keys(log_attributes, first_event_attrs):
    attributes = ["name"]
    global_event_attrs = log_attributes.get("global_attributes", {}).get("event")
    if global_event_attrs:
        ignored = [
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
        ]
        attributes += sorted(key for key in global_event_attrs if key not in ignored)
    else:
        attributes += sorted(key for key in first_event_attrs if not key.startswith("_"))
    return attributes


def collect_xes_stats(path):
    from lxml import etree

    log_attributes = parse_xes_header(path)
    classifiers = log_attributes["classifiers"]
    warned_missing_keys = set()
    unique_activities = set()
    selected_attribute_keys = None
    if log_attributes.get("global_attributes", {}).get("event"):
        selected_attribute_keys = event_attribute_keys(log_attributes, {})
    selected_event_values = {}
    first_event_attrs = None
    num_cases = 0

    with open_xes(path) as infile:
        for _, element in etree.iterparse(infile, events=("end",)):
            if local_name(element) != "trace":
                continue

            num_cases += 1
            for child in element:
                if local_name(child) != "event":
                    continue

                event_attrs = parse_xes_attributes(child)
                if first_event_attrs is None:
                    first_event_attrs = event_attrs
                    if selected_attribute_keys is None:
                        selected_attribute_keys = event_attribute_keys(log_attributes, first_event_attrs)

                name = event_name_from_classifier(event_attrs, classifiers, warned_missing_keys)
                if name is None:
                    continue
                unique_activities.add(name)

                for key in selected_attribute_keys:
                    if key != "name" and key in event_attrs and not key.startswith("_"):
                        selected_event_values.setdefault(key, set()).add(event_attrs[key])

            clear_element(element)

    if selected_attribute_keys is None:
        selected_attribute_keys = event_attribute_keys(log_attributes, first_event_attrs or {})
    unique_attribute_values = {"name": sorted(unique_activities)}
    for key in selected_attribute_keys:
        if key == "name":
            continue
        unique_attribute_values[key] = sorted(selected_event_values.get(key, []))

    return EventLogStats(
        attributes=log_attributes,
        unique_activities=sorted(unique_activities),
        unique_attribute_values=unique_attribute_values,
        num_cases=num_cases,
    )


def parse_xes_case(trace, classifiers, warned_missing_keys):
    events = []
    attributes = {}

    for child in trace:
        tag = local_name(child)
        if tag == "event":
            event_attrs = parse_xes_attributes(child)
            name = event_name_from_classifier(event_attrs, classifiers, warned_missing_keys)
            if name is None:
                continue
            events.append(
                Event(
                    name=name,
                    timestamp=event_attrs.get("time:timestamp"),
                    **event_attrs,
                )
            )
        else:
            key = child.attrib.get("key")
            if key is not None:
                attributes[key] = child.attrib.get("value", "")

    case_id = attributes.get("concept:name")
    if "id" in attributes:
        del attributes["id"]
    return Case(id=case_id, events=events, **attributes)


def stream_json_cases(output_path, attributes, case_iter):
    with gzip.open(output_path, "wt", encoding="utf-8") as outfile:
        outfile.write('{"attributes":')
        json.dump(attributes, outfile, sort_keys=True, separators=(",", ": "))
        outfile.write(',"cases":[')
        first = True
        for case in case_iter:
            if not first:
                outfile.write(",")
            json.dump(case.json, outfile, sort_keys=True, separators=(",", ": "))
            first = False
        outfile.write("]}")


def inject_control_flow_anomalies_streaming(event_log_path, anomaly_p):
    from lxml import etree

    stats = collect_xes_stats(event_log_path)
    anomalies = build_control_flow_anomalies(stats)
    classifiers = stats.attributes["classifiers"]
    warned_missing_keys = set()
    progress_desc = f"Inject {Path(event_log_path).name} @ {anomaly_p:.2f}"

    def injected_cases():
        with open_xes(event_log_path) as infile:
            trace_iter = etree.iterparse(infile, events=("end",))
            with tqdm(total=stats.num_cases, desc=progress_desc, leave=False) as progress:
                for _, trace in trace_iter:
                    if local_name(trace) != "trace":
                        continue

                    case = parse_xes_case(trace, classifiers, warned_missing_keys)
                    if np.random.uniform(0, 1) <= anomaly_p:
                        np.random.choice(anomalies).apply_to_case(case)
                    else:
                        NoneAnomaly().apply_to_case(case)
                    yield case
                    clear_element(trace)
                    progress.update(1)

    base_name = os.path.split(event_log_path)[1].split(".")[0]
    output_path = os.path.join(EVENTLOG_DIR, f"{base_name}-{anomaly_p:.2f}.json.gz")
    stream_json_cases(output_path, stats.attributes, injected_cases())
    return output_path


def inject_control_flow_anomalies(event_log_path, anomaly_p):
    if str(event_log_path).endswith((".xes", ".xes.gz")):
        return inject_control_flow_anomalies_streaming(event_log_path, anomaly_p)

    event_log = EventLog.from_xes(event_log_path)
    anomalies = build_control_flow_anomalies(event_log)

    for case in tqdm(event_log, desc=f"Inject {Path(event_log_path).name} @ {anomaly_p:.2f}", leave=False):
        if np.random.uniform(0, 1) <= anomaly_p:
            np.random.choice(anomalies).apply_to_case(case)
        else:
            NoneAnomaly().apply_to_case(case)

    base_name = os.path.split(event_log_path)[1].split(".")[0]
    output_path = os.path.join(EVENTLOG_DIR, f"{base_name}-{anomaly_p:.2f}.json.gz")
    event_log.save_json(output_path)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Inject control-flow anomalies into real-life event logs.")
    parser.add_argument(
        "input_files",
        nargs="*",
        help="Optional specific XES/XES.GZ log file(s), e.g. generator/real-life_Logs/BPIC17.xes.gz.",
    )
    parser.add_argument(
        "--log-dir",
        default=str(REAL_LIFE_LOG_DIR),
        help="Directory containing raw XES/XES.GZ real-life logs.",
    )
    parser.add_argument(
        "--ps",
        nargs="+",
        type=float,
        default=DEFAULT_ANOMALY_P,
        help="Anomaly ratios to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used during anomaly injection.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    event_log_paths = args.input_files or get_log_files(args.log_dir)
    combinations = [(event_log_path, p) for event_log_path in event_log_paths for p in args.ps]
    for event_log_path, anomaly_p in tqdm(combinations, desc="Add control-flow anomalies"):
        inject_control_flow_anomalies(event_log_path, anomaly_p)


if __name__ == "__main__":
    main()
