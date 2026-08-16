import numpy as np


def precision_recall_curve(y_true, probas_pred):
    y_true = np.asarray(y_true, dtype=int).ravel()
    scores = np.asarray(probas_pred, dtype=float).ravel()
    if y_true.shape[0] != scores.shape[0]:
        raise ValueError(
            f"y_true and probas_pred length mismatch: {y_true.shape[0]} != {scores.shape[0]}"
        )
    if y_true.size == 0:
        return np.asarray([1.0]), np.asarray([0.0]), np.asarray([])

    positive_count = int(np.sum(y_true == 1))
    if positive_count == 0:
        thresholds = np.unique(scores)[::-1]
        return (
            np.ones(thresholds.size + 1, dtype=float),
            np.zeros(thresholds.size + 1, dtype=float),
            thresholds,
        )

    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores = scores[order]
    sorted_true = (y_true[order] == 1).astype(int)

    distinct_value_indices = np.where(np.diff(sorted_scores))[0]
    threshold_indices = np.r_[distinct_value_indices, sorted_true.size - 1]

    tps = np.cumsum(sorted_true)[threshold_indices]
    fps = 1 + threshold_indices - tps
    thresholds = sorted_scores[threshold_indices]

    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / positive_count

    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    return precision.astype(float), recall.astype(float), thresholds


def average_precision_score(y_true, probas_pred):
    precision, recall, _ = precision_recall_curve(y_true, probas_pred)
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def cal_best_PRF(y_true, probas_pred):
    precisions, recalls, _ = precision_recall_curve(y_true, probas_pred)

    f1s = (2 * precisions * recalls) / np.clip(precisions + recalls, 1e-12, None)
    f1s[np.isnan(f1s)] = 0

    best_index = np.argmax(f1s)
    aupr = average_precision_score(y_true, probas_pred)

    return precisions[best_index], recalls[best_index], f1s[best_index], aupr


def _metrics_dict(y_true, scores):
    precision, recall, f1, aupr = cal_best_PRF(y_true, scores)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "aupr": aupr,
    }


def _as_1d_scores(scores, expected_size, label_name):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError(f"{label_name} scores must be a 1D array, got shape {scores.shape}.")
    if scores.shape[0] != expected_size:
        raise ValueError(
            f"{label_name} scores length mismatch: expected {expected_size}, got {scores.shape[0]}."
        )
    return scores


def evaluate_trace_level(dataset, trace_scores):
    y_true = np.asarray(dataset.case_target, dtype=int)
    scores = _as_1d_scores(trace_scores, dataset.num_cases, "Trace-level")
    return _metrics_dict(y_true, scores)


def _flatten_event_targets(dataset):
    valid_positions = ~dataset.mask
    for case_index, case_len in enumerate(dataset.case_lens):
        case_len = int(case_len)
        if case_len >= 2:
            valid_positions[case_index, 0] = False
            valid_positions[case_index, case_len - 1] = False
    event_targets = np.asarray(dataset.binary_targets[:, :, 0], dtype=int)
    return event_targets[valid_positions], valid_positions


def evaluate_event_level(dataset, event_scores):
    y_true, valid_positions = _flatten_event_targets(dataset)
    scores = np.asarray(event_scores, dtype=float)

    if scores.ndim == 2:
        expected_shape = dataset.binary_targets[:, :, 0].shape
        if scores.shape != expected_shape:
            raise ValueError(
                f"Event-level scores shape mismatch: expected {expected_shape}, got {scores.shape}."
            )
        scores = scores[valid_positions]
    elif scores.ndim == 1:
        expected_size = int(valid_positions.sum())
        if scores.shape[0] != expected_size:
            raise ValueError(
                f"Event-level scores length mismatch: expected {expected_size}, got {scores.shape[0]}."
            )
    else:
        raise ValueError(f"Event-level scores must be a 1D or 2D array, got shape {scores.shape}.")

    return _metrics_dict(y_true, scores)


def evaluate_scores(dataset, trace_scores=None, event_scores=None):
    metrics = {}
    if trace_scores is not None:
        metrics["trace"] = evaluate_trace_level(dataset, trace_scores)
    if event_scores is not None:
        metrics["event"] = evaluate_event_level(dataset, event_scores)
    return metrics
