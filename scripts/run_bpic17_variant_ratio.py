#!/usr/bin/env python3
"""User-run BPIC17 repeat-1 smoke test and 15-run DCAD-family trial.

Default stage is read-only check. Importing this module never starts training.
Independent snapshot of the BPIC12 protocol; keeps active BPIC12 jobs unchanged.
Manifest provenance paths are resolved against this checkout, without editing manifests.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dcad.variant_ratio import split

DATASET, REPEAT = 'BPIC17', 1
RATIOS = (.2, .4, .6, .8, .9)
METHODS = ('DCAD', 'EM-DCAD', 'TN-DCAD')
VERSION = 'bpic17_trial_v1'
SOURCE_FILES = (
    'scripts/run_bpic17_variant_ratio.py',
    'dcad/variant_ratio/split.py',
    'dcad/em_dcad.py', 'dcad/tn_dcad.py',
    'dcad/variant_split.py', 'dcad/model.py',
    'scripts/Inductive_Miner_Infrequent.py',
    'dcad/scoring.py', 'dcad/eval.py', 'dcad/dataset.py',
    'dcad/anomaly.py', 'dcad/fs.py', 'dcad/processmining/log.py',
    'dcad/variant_ratio/io.py', 'dcad/enums.py',
    'dcad/processmining/case.py', 'dcad/processmining/event.py',
)
FIELDS = ('Ratio', 'Method', 'Train traces', 'Unseen normal', 'Anomaly',
          'Trace AUPR', 'Trace F1', 'Event AUPR', 'Event F1', 'Observed trace ratio',
          'Gradient train traces', 'Validation traces')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def train_parameters(method, epochs, batch_size):
    common = dict(event_log=DATASET, case_id_col='case_id', activity_col='name', max_len=None,
                  batch_size=batch_size, epochs=epochs, lr=1e-3, weight_decay=1e-4,
                  seed=2026, mask_seed=2026, validation_mask_seed=102026, num_workers=0,
                  validation_ratio=.1, early_stopping_enabled=True,
                  early_stopping_patience=50, early_stopping_min_delta=1e-4,
                  d_model=128, nhead=4, num_layers=4, dim_feedforward=256, dropout=.1)
    if method == 'TN-DCAD':
        common.update(t_min=.05, t_max=.5, trace_weight_beta=.5,
                      low_start='zero', medium_start='t_min', high_start='t_min',
                      low_cost_quantile=.25, high_cost_quantile=.75)
    else:
        common.update(t_min=0., t_max=.6, adaptive_weight=0. if method == 'DCAD' else .1,
                      adaptive_schedule_gamma=3, rarity_temperature=1., dfg_smoothing_alpha=1.)
    return common


def prepare_ratio(data_root, ratio, hash_cache=None):
    """Read-only: replay validation and bind the current mixed test to its inputs."""
    from dcad import em_dcad as recovered
    from dcad.scoring import load_test_traces
    import gzip
    cache = {} if hash_cache is None else hash_cache

    def checked(path, expected):
        path = Path(path).resolve()
        if path not in cache:
            cache[path] = split.file_sha256(path)
        if cache[path] != expected:
            raise ValueError(f'Input hash mismatch: {path}')
        return path

    seed = split.derive_seed(DATASET, 42, REPEAT)
    relative = Path(DATASET)/f'seed_{seed}'/split.ratio_directory(ratio)
    normal_manifest = data_root/'splits'/relative/'split_manifest.json'
    mixed_manifest = data_root/'mixed_testsets'/relative/'test_manifest.json'
    m, mixed = read_json(normal_manifest), read_json(mixed_manifest)
    if (m['dataset'],m['repeat'],m['seed'],m['requested_observed_ratio']) != (DATASET,REPEAT,seed,ratio):
        raise ValueError('Unexpected normal split identity.')
    if (mixed['dataset'],mixed['repeat'],mixed['split_seed'],mixed['observed_ratio']) != (DATASET,REPEAT,seed,ratio):
        raise ValueError('Unexpected mixed split identity.')
    checked(normal_manifest, mixed['normal_split_manifest_sha256'])
    train_path = checked(normal_manifest.parent/m['files']['train']['filename'],m['files']['train']['sha256'])
    normal_test_path = checked(normal_manifest.parent/m['files']['test']['filename'],m['files']['test']['sha256'])
    raw_path, pool_manifest, pool_path = portable_inputs(data_root, m, mixed)
    checked(raw_path,m['input_sha256'])
    checked(pool_manifest,mixed['pool_manifest_sha256'])
    checked(pool_path,mixed['pool_sha256'])
    test_path = checked(mixed_manifest.parent/mixed['file']['filename'],mixed['file']['sha256'])
    _,train_cases = split.read_cases(train_path)
    _,test_cases = split.read_cases(normal_test_path)
    variants = {tuple(v['activity_sequence']):v['variant_id'] for v in m['variants']}
    audit = split.audit_validation(train_path,normal_test_path,train_cases,test_cases,variants,split.AuditConfig())
    if {k:v for k,v in audit.items() if k != 'implementation_sha256'} != {k:v for k,v in m['validation_audit'].items() if k != 'implementation_sha256'}:
        raise ValueError('Validation replay differs from the saved normal split audit.')
    if not audit['validation']['case_count']:
        raise ValueError('This trial requires a nonempty validation set.')
    sequences = recovered.read_activity_sequences_from_csv(str(train_path),'case_id','name')
    tn = load_module('_trial_tn_reader',ROOT/'dcad/tn_dcad.py')
    tn_cases = tn.read_activity_cases_from_csv(str(train_path),'case_id','name')
    if [seq for _,seq in tn_cases] != sequences:
        raise ValueError('TN and DCAD preprocessing differ.')
    with gzip.open(test_path,'rt',encoding='utf-8') as handle:
        payload=json.load(handle)
    n,k = mixed['normal_traces'],mixed['anomalous_traces']
    if n != len(test_cases) or k != (n+1)//2 or len(payload['cases']) != n+k:
        raise ValueError('Mixed test size/ratio mismatch.')
    if [tuple(e['name'] for e in c['events']) for c in payload['cases'][:n]] != [c.sequence for c in test_cases]:
        raise ValueError('Mixed normal cases differ from the normal test CSV.')
    if any(c['attributes']['label'] != 'normal' for c in payload['cases'][:n]):
        raise ValueError('Mixed normal labels are incorrect.')
    if any(not isinstance(c['attributes']['label'],dict) for c in payload['cases'][n:]):
        raise ValueError('Mixed anomaly labels are incorrect.')
    del payload
    traces=load_test_traces(str(test_path))
    boundary_pair=all(seq[0]=='▶' and seq[-1]=='■' for seq in sequences)
    if not boundary_pair:
        raise ValueError('Expected BPIC CSV START/END boundary tokens on all cases.')
    max_train=max(map(len,sequences)); max_test=max(map(len,traces))
    lengths={'train_csv_path':str(train_path),'train_csv_max_trace_length':max_train,
             'train_csv_max_boundary_input_length':max_train,
             'custom_test_path':str(test_path),'custom_test_dataset':test_path.name,
             'custom_test_max_trace_length':max_test,'custom_test_max_boundary_input_length':max_test+2}
    return {'ratio':ratio,'split_seed':seed,'train_csv':str(train_path),'test_json':str(test_path),
            'input_hashes':{str(p):h for p,h in cache.items() if p in (
                normal_manifest.resolve(), train_path,normal_test_path,test_path,
                raw_path.resolve(),pool_manifest.resolve(),pool_path.resolve())},
            'mixed_manifest':str(mixed_manifest),'mixed_manifest_sha256':split.file_sha256(mixed_manifest),
            'lengths':lengths,'model_max_len':max(max_train,max_test+2),
            'train_case_ids':[c.case_id for c in train_cases],
            'gradient_case_ids':audit['gradient_train']['case_ids'],
            'validation_case_ids':audit['validation']['case_ids'],
            'train_traces':len(train_cases),'normal_traces':n,'anomaly_traces':k,
            'observed_trace_ratio':m['statistics']['train_case_ratio']}


def portable_inputs(data_root, normal, mixed):
    """Resolve recorded provenance into this checkout; never fall back to old files.

    The manifests themselves remain byte-for-byte unchanged and hash-verified.
    Only the three provenance paths need relocation; split CSV and mixed JSON
    filenames are already relative to their manifests.
    """
    expected = (f'{DATASET}.csv', 'pool_manifest.json', f'{DATASET}_anomaly_pool.json.gz')
    recorded = (normal['input_path'], mixed['pool_manifest'], mixed['pool_path'])
    if tuple(Path(p).name for p in recorded) != expected:
        raise ValueError('Unexpected raw/pool provenance filenames.')
    pool_dir = Path(data_root)/'anomaly_pools'/DATASET
    return ROOT/'data/processed/raw'/expected[0], pool_dir/expected[1], pool_dir/expected[2]


def make_job(prepared, method, stage, args, sources):
    epochs=1 if stage=='smoke' else args.epochs
    directory=args.output_dir/stage/split.ratio_directory(prepared['ratio'])/method
    job={'version':VERSION,'stage':stage,'dataset':DATASET,'repeat':REPEAT,'method':method,
         'data':prepared,'run_dir':str(directory),
         'train_parameters':train_parameters(method,epochs,args.batch_size),
         'inference':{'seed':1,'num_mask_samples':5,'num_samples_t_cont':5,
                      'mode':'multi_t','noise_condition':'actual_mask_ratio'},
         'alignment':{'scope':'outer_training_candidates','reference_case_coverage':.25,'noise_threshold':1},
         'source_hashes':sources}
    job['signature']=fingerprint(job)
    return job


def artifact_hashes(directory, names):
    return {name:split.file_sha256(directory/name) for name in names}


def verify_completion(job):
    path=Path(job['run_dir'])/'completed.json'
    if not path.is_file():
        return False
    result=read_json(path)
    if result['signature'] != job['signature']:
        raise ValueError(f'{path}: configuration/data/code changed; use --overwrite to rerun.')
    for name,digest in result['artifacts'].items():
        if split.file_sha256(path.parent/name) != digest:
            raise ValueError(f'{path.parent/name}: completed artifact changed.')
    return True


def prepare_alignment(job, module):
    """Mine only this ratio's outer train; align each distinct variant once."""
    import pm4py
    from pm4py.objects.log.obj import EventLog,Trace,Event
    miner=load_module('_trial_reference_miner',ROOT/'scripts/Inductive_Miner_Infrequent.py')
    directory=Path(job['run_dir'])/'alignment';directory.mkdir(parents=True,exist_ok=True)
    raw=module.read_activity_cases_from_csv(job['data']['train_csv'],'case_id','name')
    groups={}
    for cid,seq in raw:
        groups.setdefault(tuple(seq),[]).append(cid)
    variants=list(groups)
    frequencies=[len(groups[v]) for v in variants]
    selected=miner.select_reference_variants(DATASET,variants,frequencies,len(raw))
    def as_trace(cid,seq):
        return Trace([Event({'concept:name':a}) for a in seq],attributes={'concept:name':str(cid)})
    reference=EventLog([as_trace(cid,variants[i]) for i in selected for cid in groups[variants[i]]])
    net,im,fm=pm4py.discover_petri_net_inductive(reference,noise_threshold=miner.NOISE_THRESHOLD)
    pnml=directory/'reference.pnml'
    pm4py.write_pnml(net,im,fm,str(pnml))
    unique=EventLog([as_trace(i,seq) for i,seq in enumerate(variants)])
    diagnostics=pm4py.conformance_diagnostics_alignments(unique,net,im,fm,
                   return_diagnostics_dataframe=True)
    raw_cost={str(row['case_id']):float(row['cost']) for _,row in diagnostics.iterrows()}
    if set(raw_cost) != {str(i) for i in range(len(variants))} or not all(math.isfinite(v) for v in raw_cost.values()):
        raise ValueError('Alignment failed or returned missing/nonfinite costs.')
    lo,hi=min(raw_cost.values()),max(raw_cost.values())
    costs={cid:raw_cost[str(i)] for i,seq in enumerate(variants) for cid in groups[seq]}
    cost_path=directory/'alignment_cost.csv'
    with cost_path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=('case_id','alignment_cost','raw_alignment_cost'))
        writer.writeheader()
        for cid,_ in raw:
            writer.writerow({'case_id':cid,'alignment_cost':(costs[cid]-lo)/(hi-lo or 1.),'raw_alignment_cost':costs[cid]})
    levels,info=module.assign_cost_levels([cid for cid,_ in raw],
                module.read_alignment_costs(str(cost_path),'case_id','alignment_cost'),.25,.75)
    if info['alignment_cost_missing_count'] or len(levels)!=len(raw):
        raise ValueError('TN alignment does not cover the full outer training set.')
    split.atomic_json(directory/'manifest.json',{'signature':job['signature'],'scope':'outer training incl. validation',
        'reference_case_ids':[cid for i in selected for cid in groups[variants[i]]],
        'reference_coverage':miner.REFERENCE_COVERAGE_RATIO,'noise_threshold':miner.NOISE_THRESHOLD,
        'case_count':len(raw),'variant_count':len(variants),'pm4py_version':pm4py.__version__,
        'cost_info':info,'files':artifact_hashes(directory,('reference.pnml','alignment_cost.csv'))})
    return cost_path


def run_training(job):
    directory=Path(job['run_dir'])
    tn=job['method']=='TN-DCAD'
    module=load_module('_trial_trainer',ROOT/('dcad/tn_dcad.py' if tn else 'dcad/em_dcad.py'))
    kwargs=dict(job['train_parameters'],csv_path=job['data']['train_csv'],save_path=str(directory))
    if tn:
        kwargs['alignment_cost_path']=str(prepare_alignment(job,module))
    # These adapters are local to this child process; repository modules stay unchanged.
    def local_lengths(csv_path):
        if Path(csv_path).resolve()!=Path(job['data']['train_csv']).resolve():
            raise ValueError('Unexpected training CSV in length adapter.')
        return job['data']['lengths']
    module.load_trace_lengths_for_train_csv=local_lengths
    original_split=module.split_dataset_by_variant
    split_seen=[]
    def checked_split(*args,**kwargs):
        actual=original_split(*args,**kwargs)
        ids=job['data']['train_case_ids']
        if ([ids[i] for i in actual.train_indices]!=job['data']['gradient_case_ids'] or
                [ids[i] for i in actual.validation_indices]!=job['data']['validation_case_ids']):
            raise ValueError('Actual trainer validation membership differs from audited split.')
        split_seen.append(True)
        return actual
    module.split_dataset_by_variant=checked_split
    module.train(module.TrainConfig(**kwargs))
    config=read_json(directory/'config.json')
    if not split_seen or config['best_epoch']<1 or config['early_stopping_monitor']!='validation_avg_weighted_masked_ce_loss':
        raise ValueError('Training did not select a validation checkpoint.')
    if not math.isfinite(config['best_monitor_loss']) or config['max_len']!=job['data']['model_max_len']:
        raise ValueError('Invalid validation loss/model length.')
    split.atomic_json(directory/'training_complete.json',{
        'signature':job['signature'],'artifacts':artifact_hashes(directory,('model.pt','config.json','vocab.json'))})


def run_inference(job):
    from dcad import scoring as infer
    from dcad import fs
    import numpy as np
    directory=Path(job['run_dir']); options=job['inference']
    fs.EVENTLOG_CACHE_DIR=str(directory/'dataset_cache')
    infer._default_score_output_path=lambda model_name,dataset_name,score_kind,filename_prefix='': directory/f'{score_kind}.npy'
    metrics=directory/'metrics.csv'
    metrics.unlink(missing_ok=True)
    sys.argv=['infer_anomaly_scores.py','--model-type','MDM',
              '--model-path',str(directory/'model.pt'),'--vocab-path',str(directory/'vocab.json'),
              '--config-path',str(directory/'config.json'),'--dataset',job['data']['test_json'],
              '--model-name',job['method'],'--mdm-inference-mode',options['mode'],
              '--mdm-noise-condition',options['noise_condition'],'--seed',str(options['seed']),
              '--num-mask-samples',str(options['num_mask_samples']),
              '--num-samples-t-cont',str(options['num_samples_t_cont']),
              '--output-metrics-csv',str(metrics)]
    infer.main()
    with metrics.open(newline='') as handle:
        rows=list(csv.DictReader(handle))
    if len(rows)!=1:
        raise ValueError('Expected exactly one inference metrics row.')
    metric=rows[0]
    for key in ('trace_aupr','trace_f1','event_aupr','event_f1'):
        value=float(metric[key])
        if not math.isfinite(value) or not 0<=value<=1:
            raise ValueError(f'Invalid metric {key}: {value}')
    scores=np.load(directory/'trace_scores.npy')
    event_scores=np.load(directory/'event_scores.npy')
    if scores.shape!=(job['data']['normal_traces']+job['data']['anomaly_traces'],):
        raise ValueError('Trace scores are incomplete.')
    if not np.isfinite(scores).all() or not np.isfinite(event_scores).all():
        raise ValueError('Nonfinite anomaly scores.')
    dataset=infer.Dataset(job['data']['test_json'])
    if int(dataset.case_target.sum())!=job['data']['anomaly_traces']:
        raise ValueError('Loaded trace labels differ from mixed test manifest.')
    if event_scores.shape!=dataset.binary_targets[:,:,0].shape:
        raise ValueError('Event scores/labels do not align.')
    return metric


def result_row(job,metric):
    d=job['data'];r=d['ratio']
    return {'Ratio':f'{round(100*r)}/{round(100*(1-r))}','Method':job['method'],
            'Train traces':d['train_traces'],'Unseen normal':d['normal_traces'],'Anomaly':d['anomaly_traces'],
            'Trace AUPR':float(metric['trace_aupr']),'Trace F1':float(metric['trace_f1']),
            'Event AUPR':float(metric['event_aupr']),'Event F1':float(metric['event_f1']),
            'Observed trace ratio':d['observed_trace_ratio'],
            'Gradient train traces':len(d['gradient_case_ids']),'Validation traces':len(d['validation_case_ids'])}


def worker(job_path):
    job=read_json(job_path);directory=Path(job['run_dir'])
    signature=job.pop('signature')
    if fingerprint(job)!=signature:
        raise ValueError('Job manifest signature mismatch.')
    job['signature']=signature
    for path,digest in job['data']['input_hashes'].items():
        if split.file_sha256(Path(path))!=digest:
            raise ValueError(f'Input changed after preflight: {path}')
    if split.file_sha256(Path(job['data']['mixed_manifest']))!=job['data']['mixed_manifest_sha256']:
        raise ValueError('Mixed test manifest changed.')
    for name,digest in job['source_hashes'].items():
        if split.file_sha256(ROOT/name)!=digest:
            raise ValueError(f'Implementation changed: {name}')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; refusing an accidental CPU training run.')
    trained=directory/'training_complete.json'
    if trained.is_file():
        record=read_json(trained)
        if record['signature']!=signature:
            raise ValueError('Partial training configuration mismatch.')
        for name,digest in record['artifacts'].items():
            if split.file_sha256(directory/name)!=digest:
                raise ValueError('Partial checkpoint hash mismatch.')
    else:
        run_training(job)
    metric=run_inference(job)
    names=['model.pt','config.json','vocab.json','training_complete.json','metrics.csv','trace_scores.npy','event_scores.npy']
    if job['method']=='TN-DCAD':
        names+=['alignment/reference.pnml','alignment/alignment_cost.csv','alignment/manifest.json']
    split.atomic_json(directory/'completed.json',{'signature':signature,'row':result_row(job,metric),
        'best_epoch':read_json(directory/'config.json')['best_epoch'],
        'metric_protocol':{'aupr':'average_precision','f1':'best test-set PR-curve F1','events':'exclude boundaries/padding'},
        'artifacts':artifact_hashes(directory,names)})


def write_results(output,stage,expected_signatures=None):
    rows=[]
    for path in (output/stage).glob('observed_*/*/completed.json'):
        result=read_json(path); job=read_json(path.parent/'job.json')
        if expected_signatures is not None and expected_signatures.get(str(path.parent))!=job['signature']:
            continue
        if not verify_completion(job):
            raise ValueError('Unexpected incomplete result.')
        rows.append(result['row'])
    rows.sort(key=lambda r:(int(r['Ratio'].split('/')[0]),METHODS.index(r['Method'])))
    target=output/stage/'results.csv'; target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_suffix('.csv.tmp')
    with temp.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
    temp.replace(target)


def execute_job(job,args):
    import shutil
    directory=Path(job['run_dir'])
    if directory.exists():
        if args.overwrite:
            shutil.rmtree(directory)
        elif args.resume:
            if verify_completion(job):
                print(f"Skip verified completion: {job['stage']} {job['data']['ratio']} {job['method']}",flush=True)
                write_results(args.output_dir,job['stage'],args.expected_signatures)
                return
            previous=directory/'job.json'
            if not previous.is_file() or read_json(previous)['signature']!=job['signature']:
                raise ValueError(f'{directory}: cannot resume a different or unknown configuration.')
        else:
            raise FileExistsError(f'{directory}: use --resume or --overwrite.')
    directory.mkdir(parents=True,exist_ok=True)
    split.atomic_json(directory/'job.json',job)
    env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='0',PYTHONUNBUFFERED='1')
    print(f"Run {job['stage']} ratio={job['data']['ratio']} {job['method']}; log: {directory/'run.log'}",flush=True)
    with (directory/'run.log').open('a' if args.resume else 'w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',str(directory/'job.json')],
                       cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    if not verify_completion(job):
        raise ValueError('Worker exited without a verified completion record.')
    write_results(args.output_dir,job['stage'],args.expected_signatures)
    print(f"Finished: {job['method']} {job['data']['ratio']}",flush=True)


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('check','smoke','formal','all'),default='check')
    parser.add_argument('--data-root',type=Path,default=ROOT/'data/variant_ratio')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/variant_ratio/BPIC17_repeat_1')
    parser.add_argument('--epochs',type=int,default=500,help='Formal training maximum; smoke is always 1 epoch.')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--dry-run',action='store_true',help='Read-only data audit and job listing; never train/mine/write.')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--resume',action='store_true')
    group.add_argument('--overwrite',action='store_true')
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.epochs<1 or args.batch_size<1:
        parser.error('epochs and batch-size must be positive.')
    args.data_root=args.data_root.expanduser().resolve();args.output_dir=args.output_dir.expanduser().resolve()
    return args


def main(argv=None):
    args=parse_args(argv)
    if args.worker:
        worker(args.worker);return
    cache={}
    ratios=(.8,) if args.stage=='smoke' else RATIOS
    data={r:prepare_ratio(args.data_root,r,cache) for r in ratios}
    sources={p:split.file_sha256(ROOT/p) for p in SOURCE_FILES}
    smoke=make_job(data[.8],'DCAD','smoke',args,sources)
    formal=[make_job(data[r],method,'formal',args,sources) for r in ratios for method in METHODS]
    args.expected_signatures={j['run_dir']:j['signature'] for j in [smoke,*formal]}
    for r in ratios:
        d=data[r]
        print(f"{round(100*r)}/{round(100*(1-r))}: outer_train={d['train_traces']}, "
              f"gradient_train={len(d['gradient_case_ids'])}, validation={len(d['validation_case_ids'])}, "
              f"unseen_normal={d['normal_traces']}, anomaly={d['anomaly_traces']}, "
              f"observed_trace_ratio={d['observed_trace_ratio']:.6f}, model_max_len={d['model_max_len']}")
    if args.stage=='check' or args.dry_run:
        if importlib.util.find_spec('pm4py') is None:
            raise RuntimeError('TN-DCAD requires pm4py in this Python environment.')
        print('Read-only check passed. Smoke: DCAD 80/20, 1 epoch. Formal: 5 ratios x 3 methods.')
        print('No training, alignment mining, checkpoint, or result directory was created.')
        return
    if args.stage in ('smoke','all'):
        execute_job(smoke,args)
    if args.stage in ('formal','all'):
        if not verify_completion(smoke):
            raise RuntimeError('Run --stage smoke first; a matching successful smoke result is required.')
        if not args.resume and not args.overwrite:
            existing=[j['run_dir'] for j in formal if Path(j['run_dir']).exists()]
            if existing:
                raise FileExistsError(f'Formal output already exists: {existing[0]}; use --resume or --overwrite.')
        for job in formal:
            execute_job(job,args)
        write_results(args.output_dir,'formal',args.expected_signatures)
        print(f"Formal trial complete: {args.output_dir/'formal/results.csv'}")


if __name__=='__main__':
    main()
