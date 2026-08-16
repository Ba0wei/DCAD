# Copyright 2018 Timo Nolle
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

import gzip
import pickle as pickle

import numpy as np

from dcad.processmining.event import Event
from dcad.anomaly import label_to_targets
from dcad.enums import AttributeType
from dcad.enums import Class
from dcad.fs import EventLogFile


def _get_event_log_class():
    from dcad.processmining.log import EventLog

    return EventLog


class SimpleLabelEncoder:
    def fit_transform(self, values):
        self.classes_ = np.asarray(sorted(set(values)))
        mapping = {value: index for index, value in enumerate(self.classes_)}
        return np.asarray([mapping[value] for value in values], dtype="int32")

    def transform(self, values):
        mapping = {value: index for index, value in enumerate(self.classes_)}
        return np.asarray([mapping[value] for value in values], dtype="int32")


def to_categorical(y, num_classes=None, dtype="float32"):
    y = np.array(y, dtype="int")
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


class Dataset(object):
    def __init__(self, dataset_name=None, beta=0, label_percent=0):
        self.dataset_name = dataset_name
        self.attribute_types = None
        self.attribute_keys = None
        self.classes = None
        self.labels = None
        self.encoders = None

        self._mask = None
        self._attribute_dims = None
        self._case_lens = None
        self._features = None
        self._event_log = None

        if self.dataset_name is not None:
            self.load(self.dataset_name)

    def load(self, dataset_name):
        el_file = EventLogFile(dataset_name)
        self.dataset_name = el_file.name

        if el_file.cache_file.exists():
            try:
                self._load_dataset_from_cache(el_file.cache_file)
                return
            except ModuleNotFoundError:
                pass

        if el_file.path.exists():
            EventLog = _get_event_log_class()
            self._event_log = EventLog.load(el_file.path)
            self.from_event_log(self._event_log)
            self._cache_dataset(el_file.cache_file)
        else:
            raise FileNotFoundError()

    @property
    def onehot_train_targets(self):
        return [
            np.pad(f[:, 1:], ((0, 0), (0, 1), (0, 0)), mode="constant") if t == AttributeType.CATEGORICAL else f
            for f, t in zip(self.onehot_features, self.attribute_types)
        ]

    def _load_dataset_from_cache(self, file):
        with gzip.open(file, "rb") as f:
            (
                self._features,
                self.classes,
                self.labels,
                self._case_lens,
                self._attribute_dims,
                self.encoders,
                self.attribute_types,
                self.attribute_keys,
            ) = pickle.load(f)

    def _cache_dataset(self, file):
        with gzip.open(file, "wb") as f:
            pickle.dump(
                (
                    self._features,
                    self.classes,
                    self.labels,
                    self._case_lens,
                    self._attribute_dims,
                    self.encoders,
                    self.attribute_types,
                    self.attribute_keys,
                ),
                f,
            )

    @property
    def mask(self):
        if self._mask is None:
            self._mask = np.ones(self._features[0].shape, dtype=bool)
            for m, j in zip(self._mask, self.case_lens):
                m[:j] = False
        return self._mask

    @property
    def event_log(self):
        if self.dataset_name is None:
            raise ValueError(f"dataset {self.dataset_name} cannot be found")

        if self._event_log is None:
            EventLog = _get_event_log_class()
            self._event_log = EventLog.load(EventLogFile(self.dataset_name).path)
        return self._event_log

    @property
    def binary_targets(self):
        if self.classes is not None and len(self.classes) > 0:
            targets = np.copy(self.classes)
            targets[targets > Class.ANOMALY] = Class.ANOMALY
            return targets
        return None

    def __len__(self):
        return self.num_cases

    @property
    def text_labels(self):
        return np.array(["Normal" if l == "normal" else l["anomaly"] for l in self.labels])

    @property
    def unique_text_labels(self):
        return sorted(set(self.text_labels))

    @property
    def unique_anomaly_text_labels(self):
        return [l for l in self.unique_text_labels if l != "Normal"]

    def get_indices_for_type(self, t):
        if len(self.text_labels) > 0:
            return np.where(self.text_labels == t)[0]
        return range(int(self.num_cases))

    @property
    def case_target(self):
        z = np.zeros(self.num_cases)
        for i in self.cf_anomaly_indices:
            z[i] = 1
        return z

    @property
    def normal_indices(self):
        return self.get_indices_for_type("Normal")

    @property
    def cf_anomaly_indices(self):
        if len(self.text_labels) > 0:
            return np.where(np.logical_and(self.text_labels != "Normal", self.text_labels != "Attribute"))[0]
        return range(int(self.num_cases))

    @property
    def anomaly_indices(self):
        if len(self.text_labels) > 0:
            return np.where(self.text_labels != "Normal")[0]
        return range(int(self.num_cases))

    @property
    def case_lens(self):
        return self._case_lens

    @property
    def attribute_dims(self):
        if self._attribute_dims is None:
            self._attribute_dims = np.asarray(
                [f.max() if t == AttributeType.CATEGORICAL else 1 for f, t in zip(self._features, self.attribute_types)]
            )
        return self._attribute_dims

    @property
    def num_attributes(self):
        return len(self.features)

    @property
    def num_cases(self):
        return len(self.features[0])

    @property
    def num_events(self):
        return sum(self.case_lens)

    @property
    def max_len(self):
        return self.features[0].shape[1]

    @property
    def _reverse_features(self):
        reverse_features = [np.copy(f) for f in self._features]
        for f in reverse_features:
            for _f, m in zip(f, self.mask):
                _f[~m] = _f[~m][::-1]
        return reverse_features

    @property
    def features(self):
        return self._features

    @property
    def flat_features(self):
        return np.dstack(self.features)

    @property
    def onehot_features(self):
        return [
            to_categorical(f)[:, :, 1:] if t == AttributeType.CATEGORICAL else np.expand_dims(f, axis=2)
            for f, t in zip(self._features, self.attribute_types)
        ]

    @property
    def flat_onehot_features(self):
        return np.concatenate(self.onehot_features, axis=2)

    @staticmethod
    def remove_time_dimension(x):
        return x.reshape((x.shape[0], np.product(x.shape[1:])))

    @property
    def flat_features_2d(self):
        return self.remove_time_dimension(self.flat_features)

    @property
    def flat_onehot_features_2d(self):
        return self.remove_time_dimension(self.flat_onehot_features)

    @staticmethod
    def _get_classes_and_labels_from_event_log(event_log):
        labels = np.asarray(
            [case.attributes["label"] for case in event_log if case.attributes is not None and "label" in case.attributes]
        )

        num_events = event_log.max_case_len + 2
        num_attributes = event_log.num_event_attributes
        targets = np.asarray([label_to_targets(label, num_events, num_attributes) for label in labels])

        return targets, labels

    @staticmethod
    def _from_event_log(event_log, include_attributes=None):
        if include_attributes is None:
            include_attributes = event_log.event_attribute_keys

        feature_columns = dict(name=[])
        case_lens = []
        attr_types = event_log.get_attribute_types(include_attributes)

        start_event = dict(
            (a, event_log.start_symbol if t == AttributeType.CATEGORICAL else 0.0)
            for a, t in zip(include_attributes, attr_types)
        )
        start_event = Event(timestamp=None, **start_event)

        end_event = dict(
            (a, event_log.end_symbol if t == AttributeType.CATEGORICAL else 0.0)
            for a, t in zip(include_attributes, attr_types)
        )
        end_event = Event(timestamp=None, **end_event)

        for case in event_log.cases:
            case_lens.append(case.num_events + 2)
            for event in [start_event] + case.events + [end_event]:
                for attribute in event_log.event_attribute_keys:
                    if attribute == "name":
                        attr = event.name
                    elif attribute in include_attributes:
                        attr = event.attributes[attribute]
                    else:
                        continue

                    if attribute not in feature_columns:
                        feature_columns[attribute] = []
                    feature_columns[attribute].append(attr)

        encoders = {}
        for key, attribute_type in zip(feature_columns.keys(), attr_types):
            if attribute_type == AttributeType.CATEGORICAL:
                encoder = SimpleLabelEncoder()
                feature_columns[key] = encoder.fit_transform(feature_columns[key]) + 1
                encoders[key] = encoder
            elif attribute_type == AttributeType.NUMERICAL:
                f = np.asarray(feature_columns[key])
                feature_columns[key] = (f - f.mean()) / f.std()

        case_lens = np.array(case_lens)
        offsets = np.concatenate(([0], np.cumsum(case_lens)[:-1]))
        features = [np.zeros((case_lens.shape[0], case_lens.max()), dtype="int32") for _ in range(len(feature_columns))]
        for i, (offset, case_len) in enumerate(zip(offsets, case_lens)):
            for k, key in enumerate(feature_columns):
                x = feature_columns[key]
                features[k][i, :case_len] = x[offset : offset + case_len]

        return features, case_lens, attr_types, encoders

    def from_event_log(self, event_log):
        self._features, self._case_lens, self.attribute_types, self.encoders = self._from_event_log(event_log)
        self.classes, self.labels = self._get_classes_and_labels_from_event_log(event_log)
        self.attribute_keys = [a.replace(":", "_").replace(" ", "_") for a in self.event_log.event_attribute_keys]
