"""Full DCAD-family inference and metric output, preserving experiment scoring."""
from __future__ import annotations
import argparse
import csv
import gzip
import json
import random
import sys
from pathlib import Path
from typing import Any
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import numpy as np
import torch
import torch.nn as nn
from dcad.model import DCADActivityModel as MDMActivityModel
from dcad.dataset import Dataset
from dcad.eval import average_precision_score, evaluate_scores, precision_recall_curve
PAD_TOKEN = '[PAD]'
MASK_TOKEN = '[MASK]'
UNK_TOKEN = '[UNK]'
START_TOKEN = '▶'
END_TOKEN = '■'
SUPPORTED_MODEL_TYPES = ('MDM',)
MODEL_TYPE_ALIASES = {model_type.lower(): model_type for model_type in SUPPORTED_MODEL_TYPES}
DEFAULT_METRICS_CSV = REPO_ROOT / 'metrics' / 'f1_aupr.csv'
BERT_INFERENCE_MASK_RATIO = 0.15

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compute trace-level and event-level anomaly scores for a supported anomaly model.')
    parser.add_argument('--model-type', '--model_type', default='MDM', help=f"Model architecture identifier. Supported: {', '.join(SUPPORTED_MODEL_TYPES)}. Default: MDM.")
    parser.add_argument('--model-path', default=None, help='Path to the trained model checkpoint.')
    parser.add_argument('--vocab-path', default=None, help='Path to the saved vocab JSON.')
    parser.add_argument('--config-path', default=None, help='Path to the saved config JSON.')
    parser.add_argument('--dataset', required=True, help='Dataset name under data/processed/custom_test/ or a direct path to a .json/.json.gz test set.')
    parser.add_argument('--model-name', default=None, help='Optional model name written to the metrics CSV. If omitted, uses the public model type.')
    parser.add_argument('--output-metrics-csv', default=None, help='Optional metrics CSV path. Defaults to metrics/f1_aupr.csv.')
    parser.add_argument('--eps', type=float, default=1e-12, help='Small constant used in -log(prob + eps). Default: 1e-12.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducible inference masking. ')
    parser.add_argument('--num-mask-samples', type=int, default=5, help='Number of independent random mask samples per target for MDM multi_t inference. Ignored when --mdm-inference-mode single_t. Default: 5.')
    parser.add_argument('--num-samples-t-cont', type=int, default=5, help='Number of continuous t samples for MDM multi_t inference. Ignored when --mdm-inference-mode single_t. Default: 5.')
    parser.add_argument('--mdm-inference-mode', choices=('multi_t', 'single_t'), default='multi_t', help='MDM inference strategy. multi_t keeps the original multi-t, multi-mask mean NLL; single_t samples one t and one random mask set per trace. Default: multi_t.')
    parser.add_argument('--mdm-noise-condition', choices=('actual_mask_ratio', 'sampled_t', 'none'), default='actual_mask_ratio', help='Value passed to the MDM noise-conditioning input during inference. Default: actual_mask_ratio.')
    return parser.parse_args()

def _require_special_token_id(token_to_id: dict[str, int], token: str) -> int:
    if token not in token_to_id:
        raise KeyError(f"Required token '{token}' is missing from the training vocabulary.")
    return int(token_to_id[token])

def _normalize_model_type(model_type: str) -> str:
    normalized = model_type.strip().lower()
    if normalized not in MODEL_TYPE_ALIASES:
        raise ValueError(f"Unsupported --model-type '{model_type}'. Supported: {', '.join(SUPPORTED_MODEL_TYPES)}.")
    return MODEL_TYPE_ALIASES[normalized]

def _display_model_type(model_type: str) -> str:
    return model_type

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_vocab(path: str | Path) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if 'token_to_id' not in payload:
        raise ValueError(f"Invalid vocab file: {path}. Missing 'token_to_id'.")
    token_to_id = {str(token): int(idx) for (token, idx) in payload['token_to_id'].items()}
    payload['token_to_id'] = token_to_id
    _require_special_token_id(token_to_id, PAD_TOKEN)
    _require_special_token_id(token_to_id, MASK_TOKEN)
    _require_special_token_id(token_to_id, UNK_TOKEN)
    return payload

def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f'Invalid config file: {path}. Expected a JSON object.')
    return config

def resolve_mdm_t_max_from_config(config: dict[str, Any], config_path: str | Path) -> float:
    if 't_max' not in config:
        raise ValueError(f"MDM config is missing required 't_max': {config_path}")
    t_max = float(config['t_max'])
    if not 0.0 < t_max <= 1.0:
        raise ValueError(f'config t_max must be in the interval (0, 1]; got {t_max}.')
    return t_max

def _resolve_existing_path(path_arg: str | Path, description: str) -> Path:
    path = Path(path_arg).expanduser()
    if path.is_absolute():
        if path.exists():
            return path
        raise FileNotFoundError(f'{description} not found: {path}')
    repo_path = REPO_ROOT / path
    if repo_path.exists():
        return repo_path
    raise FileNotFoundError(f'{description} not found: {repo_path}')

def _resolve_repo_relative_path(path_arg: str | Path) -> Path:
    path = Path(path_arg).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path

def resolve_artifact_paths(args: argparse.Namespace, model_type: str) -> tuple[Path, Path, Path]:
    missing = [name for (name, value) in (('--model-path', args.model_path), ('--vocab-path', args.vocab_path), ('--config-path', args.config_path)) if value is None]
    if missing:
        raise ValueError(f"{', '.join(missing)} must be provided for --model-type {model_type}.")
    return (_resolve_existing_path(args.model_path, 'Model checkpoint'), _resolve_existing_path(args.vocab_path, 'Vocab file'), _resolve_existing_path(args.config_path, 'Config file'))

def build_model_from_config(model_type: str, config: dict[str, Any], vocab: dict[str, Any]) -> nn.Module:
    model_type = _normalize_model_type(model_type)
    token_to_id = vocab['token_to_id']
    max_len = config.get('max_len')
    if max_len is None:
        raise ValueError('config.json contains max_len=None, so the positional embedding size cannot be recovered safely. Please use a checkpoint saved with an explicit max_len.')
    return MDMActivityModel(vocab_size=len(token_to_id), d_model=int(config['d_model']), nhead=int(config['nhead']), num_layers=int(config['num_layers']), dim_feedforward=int(config['dim_feedforward']), dropout=float(config['dropout']), max_len=int(max_len), pad_token_id=_require_special_token_id(token_to_id, PAD_TOKEN))
    raise AssertionError(f'Unhandled model type: {model_type}')

def load_model_weights(model: nn.Module, model_path: str | Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' not in checkpoint:
        raise ValueError(f"Invalid checkpoint: {model_path}. Missing 'model_state_dict'.")
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(device)
    model.eval()
    return model

def resolve_dataset_path(dataset_arg: str) -> Path:
    candidate = Path(dataset_arg).expanduser()
    if candidate.exists():
        return candidate.resolve()
    if not any((suffix == '.json' for suffix in candidate.suffixes)):
        candidate = Path(str(candidate) + '.json.gz')
        if candidate.exists():
            return candidate.resolve()
    eventlogs_dir = REPO_ROOT / 'data' / 'processed' / 'custom_test'
    eventlog_candidate = eventlogs_dir / candidate.name
    if eventlog_candidate.exists():
        return eventlog_candidate.resolve()
    custom_test_candidate = eventlogs_dir / f'{Path(dataset_arg).name}_custom_test.json.gz'
    if custom_test_candidate.exists():
        return custom_test_candidate.resolve()
    raise FileNotFoundError(f"Could not find dataset '{dataset_arg}'. Tried '{Path(dataset_arg).expanduser()}', '{eventlog_candidate}', and '{custom_test_candidate}'.")

def _open_json_maybe_gzip(path: Path):
    if path.suffix == '.gz':
        return gzip.open(path, 'rt', encoding='utf-8')
    return open(path, 'r', encoding='utf-8')

def load_test_traces(dataset_arg: str) -> list[list[str]]:
    dataset_path = resolve_dataset_path(dataset_arg)
    with _open_json_maybe_gzip(dataset_path) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f'Dataset file must contain a JSON object: {dataset_path}')
    if 'cases' in payload:
        case_key = 'cases'
    elif 'traces' in payload:
        case_key = 'traces'
    else:
        raise ValueError(f"Dataset file does not contain 'cases' or 'traces': {dataset_path}")
    traces: list[list[str]] = []
    for (case_index, case) in enumerate(payload[case_key]):
        events = case.get('events')
        if not isinstance(events, list):
            raise ValueError(f"Case at index {case_index} in {dataset_path} does not contain a valid 'events' list.")
        trace: list[str] = []
        for (event_index, event) in enumerate(events):
            name = event.get('name')
            if name is None:
                raise ValueError(f"Event at case index {case_index}, event index {event_index} in {dataset_path} does not contain a 'name' field.")
            trace.append(str(name))
        traces.append(trace)
    return traces

def encode_trace(trace: list[str], token_to_id: dict[str, int]) -> list[int]:
    unk_id = _require_special_token_id(token_to_id, UNK_TOKEN)
    return [token_to_id.get(activity, unk_id) for activity in trace]

def _resolve_metrics_model_name(model_type: str, requested_model_name: str | None) -> str:
    if requested_model_name is not None and requested_model_name.strip():
        return requested_model_name.strip()
    return model_type

@torch.no_grad()
def compute_event_scores_for_trace(model: nn.Module, model_type: str, encoded_trace: list[int], true_ids: list[int], mask_token_id: int, device: torch.device, eps: float, num_mask_samples: int=5, num_samples_t_cont: int=5, t_max: float=0.5, mdm_inference_mode: str='multi_t', mdm_noise_condition: str='actual_mask_ratio', log_scoring: bool=False, start_token_id: int | None=None, end_token_id: int | None=None, random_token_ids: list[int] | None=None) -> np.ndarray:
    if len(encoded_trace) != len(true_ids):
        raise ValueError('encoded_trace and true_ids must have the same length.')
    if len(encoded_trace) == 0:
        return np.zeros((0,), dtype=np.float32)
    model_input_ids = list(encoded_trace)
    score_positions = list(range(len(encoded_trace)))
    if start_token_id is not None and end_token_id is not None:
        model_input_ids = [start_token_id] + model_input_ids + [end_token_id]
        score_positions = [position + 1 for position in score_positions]
    seq_len = len(model_input_ids)
    if seq_len > model.max_len:
        raise ValueError(f'Trace length {len(encoded_trace)} requires model input length {seq_len}, which exceeds model max_len {model.max_len}.')
    num_events = len(encoded_trace)
    input_ids = torch.tensor([model_input_ids] * num_events, dtype=torch.long, device=device)
    attention_mask = torch.ones((num_events, seq_len), dtype=torch.long, device=device)
    row_indices = torch.arange(num_events, device=device)
    masked_positions = torch.tensor(score_positions, dtype=torch.long, device=device)
    target_token_ids = torch.tensor(true_ids, dtype=torch.long, device=device)
    input_ids[row_indices, masked_positions] = mask_token_id
    if not 0.0 < t_max <= 1.0:
        raise ValueError(f't_max must be in the interval (0, 1]; got {t_max}.')
    if mdm_inference_mode not in {'multi_t', 'single_t'}:
        raise ValueError(f"mdm_inference_mode must be either 'multi_t' or 'single_t'; got {mdm_inference_mode!r}.")
    if mdm_noise_condition not in {'actual_mask_ratio', 'sampled_t', 'none'}:
        raise ValueError(f"mdm_noise_condition must be 'actual_mask_ratio', 'sampled_t', or 'none'; got {mdm_noise_condition!r}.")
    boundary_token_ids = {token_id for token_id in (start_token_id, end_token_id) if token_id is not None}
    real_event_positions = [position for position in score_positions if int(model_input_ids[position]) not in boundary_token_ids]
    real_event_count = len(real_event_positions)
    if mdm_inference_mode == 'multi_t':
        if num_mask_samples < 1:
            raise ValueError(f'num_mask_samples must be at least 1; got {num_mask_samples}.')
        if num_samples_t_cont < 1:
            raise ValueError(f'num_samples_t_cont must be at least 1; got {num_samples_t_cont}.')
        if log_scoring:
            print('MDM scoring mode: continuous_t_mean_nll')
            print(f'num_samples_t_cont: {num_samples_t_cont}')
            print(f't_max: {t_max}')
            print(f'num_mask_samples: {num_mask_samples}')
            print(f'Each target is fixed as [MASK]; non-target real positions are randomly masked from sampled t_cont; the model receives {mdm_noise_condition}.')
        t_cont_values = [random.random() * t_max for _ in range(num_samples_t_cont)]
        conditioning_values = []
        total_eval_steps = num_samples_t_cont * num_mask_samples
        expanded_input_ids = input_ids.repeat(total_eval_steps, 1)
        expanded_attention_mask = attention_mask.repeat(total_eval_steps, 1)
        expanded_positions = masked_positions.repeat(total_eval_steps)
        expanded_targets = target_token_ids.repeat(total_eval_steps)
        for (t_index, mask_ratio) in enumerate(t_cont_values):
            if real_event_count == 0:
                num_to_mask = 0
            else:
                num_to_mask = max(1, int(round(real_event_count * float(mask_ratio))))
                num_to_mask = min(num_to_mask, real_event_count)
            extra_to_mask = max(0, num_to_mask - 1)
            actual_mask_ratio = float(num_to_mask) / real_event_count if real_event_count else 0.0
            conditioning_values.append(float(mask_ratio) if mdm_noise_condition == 'sampled_t' else 0.0 if mdm_noise_condition == 'none' else actual_mask_ratio)
            for sample_index in range(num_mask_samples):
                eval_step_index = t_index * num_mask_samples + sample_index
                for (event_index, target_pos) in enumerate(score_positions):
                    if extra_to_mask == 0:
                        continue
                    candidate_positions = [position for position in real_event_positions if position != target_pos]
                    extra_positions = random.sample(candidate_positions, k=extra_to_mask)
                    expanded_row = eval_step_index * num_events + event_index
                    expanded_input_ids[expanded_row, extra_positions] = mask_token_id
        mask_ratios = torch.tensor(conditioning_values, dtype=torch.float32, device=device).repeat_interleave(num_mask_samples).unsqueeze(1).expand(total_eval_steps, num_events).reshape(-1)
        expanded_row_indices = torch.arange(num_events * total_eval_steps, device=device)
        logits = model(input_ids=expanded_input_ids, attention_mask=expanded_attention_mask, mask_ratios=mask_ratios)
        masked_logits = logits[expanded_row_indices, expanded_positions, :]
        probabilities = torch.softmax(masked_logits, dim=-1)
        true_probabilities = probabilities.gather(1, expanded_targets.unsqueeze(1)).squeeze(1)
        true_probabilities = true_probabilities.view(total_eval_steps, num_events)
        scores = -torch.log(true_probabilities.clamp_min(eps)).mean(dim=0)
    else:
        if log_scoring:
            print('MDM scoring mode: single_t_random_mask_nll')
            print(f't_max: {t_max}')
            print(f'One continuous t is sampled per trace. Each target is fixed as [MASK]; non-target real positions are randomly masked once with sampled t_cont; the model receives {mdm_noise_condition}.')
        mask_ratio = random.random() * t_max
        expanded_input_ids = input_ids.clone()
        if real_event_count == 0:
            num_to_mask = 0
        else:
            num_to_mask = max(1, int(round(real_event_count * float(mask_ratio))))
            num_to_mask = min(num_to_mask, real_event_count)
        extra_to_mask = max(0, num_to_mask - 1)
        actual_mask_ratio = float(num_to_mask) / real_event_count if real_event_count else 0.0
        conditioning_value = float(mask_ratio) if mdm_noise_condition == 'sampled_t' else 0.0 if mdm_noise_condition == 'none' else actual_mask_ratio
        for (event_index, target_pos) in enumerate(score_positions):
            if extra_to_mask == 0:
                continue
            candidate_positions = [position for position in real_event_positions if position != target_pos]
            extra_positions = random.sample(candidate_positions, k=extra_to_mask)
            expanded_input_ids[event_index, extra_positions] = mask_token_id
        mask_ratios = torch.full((num_events,), conditioning_value, dtype=torch.float32, device=device)
        logits = model(input_ids=expanded_input_ids, attention_mask=attention_mask, mask_ratios=mask_ratios)
        masked_logits = logits[row_indices, masked_positions, :]
        probabilities = torch.softmax(masked_logits, dim=-1)
        true_probabilities = probabilities.gather(1, target_token_ids.unsqueeze(1)).squeeze(1)
        scores = -torch.log(true_probabilities.clamp_min(eps))
    return scores.detach().cpu().numpy().astype(np.float32)

def compute_trace_scores(event_score_rows: list[np.ndarray], aggregation: str='max') -> np.ndarray:
    if aggregation not in {'max', 'mean'}:
        raise ValueError(f'Unsupported trace score aggregation: {aggregation}')
    trace_scores = np.zeros((len(event_score_rows),), dtype=np.float32)
    for (index, row) in enumerate(event_score_rows):
        if row.size > 0:
            if aggregation == 'max':
                trace_scores[index] = float(np.max(row))
            else:
                trace_scores[index] = float(np.mean(row))
    return trace_scores

def save_scores(trace_scores: np.ndarray, event_scores: np.ndarray, trace_path: str | Path, event_path: str | Path) -> None:
    trace_path = Path(trace_path).expanduser()
    event_path = Path(event_path).expanduser()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(trace_path, trace_scores)
    np.save(event_path, event_scores)

def _normalize_dataset_name_for_metrics(dataset_arg: str) -> str:
    dataset_path = Path(dataset_arg).expanduser()
    if dataset_path.exists():
        name = dataset_path.name
    else:
        name = dataset_arg
    if name.endswith('.json.gz'):
        return name[:-len('.json.gz')]
    if name.endswith('.json'):
        return name[:-len('.json')]
    return Path(name).stem

def _default_metrics_csv_path() -> Path:
    return DEFAULT_METRICS_CSV

def build_metrics_row(model_name: str, dataset_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {'model': model_name, 'dataset': dataset_name}
    if 'trace' in metrics:
        row.update({f'trace_{key}': value for (key, value) in metrics['trace'].items()})
    if 'event' in metrics:
        row.update({f'event_{key}': value for (key, value) in metrics['event'].items()})
    return row

def _valid_real_event_positions(dataset: Dataset) -> np.ndarray:
    valid_positions = ~dataset.mask
    for (case_index, case_len) in enumerate(dataset.case_lens):
        case_len = int(case_len)
        if case_len >= 2:
            valid_positions[case_index, 0] = False
            valid_positions[case_index, case_len - 1] = False
    return valid_positions

def save_metrics_csv(metrics_row: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['model', 'dataset', 'trace_precision', 'trace_recall', 'trace_f1', 'trace_aupr', 'event_precision', 'event_recall', 'event_f1', 'event_aupr']
    file_exists = output_path.exists()
    if file_exists:
        with open(output_path, 'r', newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = reader.fieldnames or []
            existing_rows = list(reader)
        if existing_fieldnames != fieldnames:
            with open(output_path, 'w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in existing_rows:
                    writer.writerow({field: row.get(field, '') for field in fieldnames})
                writer.writerow({field: metrics_row.get(field, '') for field in fieldnames})
            return
    with open(output_path, 'a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({field: metrics_row.get(field, '') for field in fieldnames})

def _resolve_score_output_model_name(model_type: str, requested_model_name: str | None) -> str:
    if requested_model_name is not None and requested_model_name.strip():
        return requested_model_name.strip()
    return model_type

def _default_score_output_path(model_name: str, dataset_name: str, score_kind: str, filename_prefix: str='') -> Path:
    safe_model_name = _sanitize_name_for_filename(model_name)
    safe_dataset_name = _sanitize_name_for_filename(dataset_name)
    safe_filename_prefix = _sanitize_name_for_filename(filename_prefix) if filename_prefix else ''
    return REPO_ROOT / 'outputs' / safe_model_name / f'{safe_filename_prefix}{safe_model_name}_{safe_dataset_name}_{score_kind}.npy'

def _flatten_event_targets_and_scores(dataset: Dataset, event_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid_positions = _valid_real_event_positions(dataset)
    event_targets = np.asarray(dataset.binary_targets[:, :, 0], dtype=int)
    return (event_targets[valid_positions], np.asarray(event_scores, dtype=float)[valid_positions])

def main() -> None:
    args = parse_args()
    model_type = _normalize_model_type(args.model_type)
    (model_path, vocab_path, config_path) = resolve_artifact_paths(args, model_type)
    set_seed(args.seed)
    if model_type == 'MDM' and args.mdm_inference_mode == 'multi_t':
        if args.num_mask_samples < 1:
            raise ValueError(f'--num-mask-samples must be at least 1; got {args.num_mask_samples}.')
        if args.num_samples_t_cont < 1:
            raise ValueError(f'--num-samples-t-cont must be at least 1; got {args.num_samples_t_cont}.')
    metrics_model_name = _resolve_metrics_model_name(model_type, args.model_name)
    dataset_name = _normalize_dataset_name_for_metrics(args.dataset)
    score_output_model_name = _resolve_score_output_model_name(model_type=model_type, requested_model_name=args.model_name)
    score_filename_prefix = 'single_t-' if model_type == 'MDM' and args.mdm_inference_mode == 'single_t' else ''
    trace_output_path = _default_score_output_path(score_output_model_name, dataset_name, 'trace_scores', filename_prefix=score_filename_prefix)
    event_output_path = _default_score_output_path(score_output_model_name, dataset_name, 'event_scores', filename_prefix=score_filename_prefix)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Model type: {_display_model_type(model_type)}')
    print(f'Model path: {model_path}')
    print(f'Vocab path: {vocab_path}')
    print(f'Config path: {config_path}')
    vocab = load_vocab(vocab_path)
    config = load_config(config_path)
    mdm_t_max = resolve_mdm_t_max_from_config(config, config_path)
    token_to_id = vocab['token_to_id']
    random_token_ids = [token_id for (token, token_id) in token_to_id.items() if token not in {PAD_TOKEN, MASK_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN}]
    pad_token_id = _require_special_token_id(token_to_id, PAD_TOKEN)
    mask_token_id = _require_special_token_id(token_to_id, MASK_TOKEN)
    start_token_id = token_to_id.get(START_TOKEN)
    end_token_id = token_to_id.get(END_TOKEN)
    if model_type == 'Prediction' and start_token_id is None:
        raise ValueError(f'{_display_model_type(model_type)} requires START token {START_TOKEN!r} in vocab.json.')
    if model_type in {'BERT', 'MDM'} and (start_token_id is None) != (end_token_id is None):
        raise ValueError('Boundary token setup is inconsistent: either both start/end tokens must exist in vocab.json, or neither of them.')
    model = build_model_from_config(model_type=model_type, config=config, vocab=vocab)
    model = load_model_weights(model, model_path, device=device)
    traces = load_test_traces(args.dataset)
    num_cases = len(traces)
    max_raw_trace_len = max((len(trace) for trace in traces), default=0)
    print(f'Loaded {num_cases} traces')
    print(f'Max raw trace length: {max_raw_trace_len}')
    print(f'Model max_len: {model.max_len}')
    uses_boundary_tokens = start_token_id is not None and end_token_id is not None
    max_model_input_len = max_raw_trace_len + (2 if uses_boundary_tokens else 0)
    print(f'Using boundary tokens in model input: {uses_boundary_tokens}')
    print(f'Max model input length: {max_model_input_len}')
    if max_model_input_len > model.max_len:
        boundary_detail = ' including START/END boundary tokens' if uses_boundary_tokens else ''
        raise ValueError(f'At least one test trace has raw length {max_raw_trace_len}, requiring model input length {max_model_input_len}{boundary_detail}, which exceeds {_display_model_type(model_type)} model max_len {model.max_len}. Increase max_len and retrain the model.')
    if args.mdm_inference_mode == 'multi_t':
        print(f'MDM score aggregation: continuous_t_nll_mean (num_samples_t_cont={args.num_samples_t_cont}, t_max={mdm_t_max}, num_mask_samples={args.num_mask_samples}, noise_condition={args.mdm_noise_condition})')
    else:
        print(f'MDM score aggregation: single_t_random_mask_nll (one sampled t per trace, t_max={mdm_t_max}, noise_condition={args.mdm_noise_condition})')
    event_score_rows: list[np.ndarray] = []
    event_scores = np.zeros((num_cases, max_raw_trace_len + 2), dtype=np.float32)
    for (case_index, trace) in enumerate(traces):
        encoded_trace = encode_trace(trace, token_to_id)
        event_row = compute_event_scores_for_trace(model=model, model_type=model_type, encoded_trace=encoded_trace, true_ids=encoded_trace, mask_token_id=mask_token_id, device=device, eps=args.eps, num_mask_samples=args.num_mask_samples, num_samples_t_cont=args.num_samples_t_cont, t_max=mdm_t_max, mdm_inference_mode=args.mdm_inference_mode, mdm_noise_condition=args.mdm_noise_condition, log_scoring=case_index == 0, start_token_id=start_token_id, end_token_id=end_token_id, random_token_ids=random_token_ids)
        event_score_rows.append(event_row)
        if event_row.size > 0:
            event_scores[case_index, 1:1 + event_row.size] = event_row
        if (case_index + 1) % 50 == 0 or case_index + 1 == num_cases:
            print(f'Scored {case_index + 1}/{num_cases} traces')
    trace_aggregation = 'max'
    trace_scores = compute_trace_scores(event_score_rows, aggregation=trace_aggregation)
    if trace_scores.shape != (num_cases,):
        raise ValueError(f'Trace score shape mismatch: expected {(num_cases,)}, got {trace_scores.shape}.')
    expected_event_shape = (num_cases, max_raw_trace_len + 2)
    if event_scores.shape != expected_event_shape:
        raise ValueError(f'Event score shape mismatch: expected {expected_event_shape}, got {event_scores.shape}.')
    for (case_index, event_row) in enumerate(event_score_rows):
        if event_row.size == 0:
            continue
        expected_trace_score = float(np.max(event_row))
        if not np.isclose(trace_scores[case_index], expected_trace_score):
            raise ValueError(f'Trace score mismatch at case index {case_index}: trace_score={trace_scores[case_index]}, expected={expected_trace_score}.')
    save_scores(trace_scores=trace_scores, event_scores=event_scores, trace_path=trace_output_path, event_path=event_output_path)
    metrics_csv_path = _resolve_repo_relative_path(args.output_metrics_csv) if args.output_metrics_csv is not None else _default_metrics_csv_path()
    dataset = Dataset(args.dataset)
    metrics = evaluate_scores(dataset, trace_scores=trace_scores, event_scores=event_scores)
    metrics_row = build_metrics_row(metrics_model_name, dataset_name, metrics)
    save_metrics_csv(metrics_row, metrics_csv_path)
    print(f'Saved trace scores to: {trace_output_path}')
    print(f'Saved event scores to: {event_output_path}')
    print(f'Saved metrics CSV to: {metrics_csv_path}')
if __name__ == '__main__':
    main()
