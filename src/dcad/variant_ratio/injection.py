"""Shared anomaly factories and XES metadata readers for ratio datasets."""
import gzip
from dcad.generation.anomaly import (EarlyAnomaly, InsertAnomaly, LateAnomaly,
                                     ReworkAnomaly, SkipSequenceAnomaly)
from dcad.generation.attribute_generator import CategoricalAttributeGenerator
XES_ATTRIBUTE_TAGS = {"string", "date", "int", "float", "boolean", "id", "list", "container"}

class EventLogStats:
    def __init__(self, attributes, unique_activities, unique_attribute_values, num_cases):
        self.attributes = attributes
        self.unique_activities = unique_activities
        self.unique_attribute_values = unique_attribute_values
        self.num_cases = num_cases


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
