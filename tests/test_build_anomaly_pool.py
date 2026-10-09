import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from dcad.processmining.case import Case
from dcad.processmining.event import Event
from dcad.variant_ratio import pool
from dcad.variant_ratio import split


class ScriptedInjector:
    def __init__(self):
        self.calls = 0

    def __str__(self):
        return 'Insert'

    def apply_to_case(self, case):
        self.calls += 1
        if self.calls == 1:
            case.attributes['label'] = 'normal'
        elif self.calls == 2:
            case.events = []
        elif self.calls == 3:
            case.attributes['label'] = {'anomaly': 'Insert', 'attr': {'indices': [0]}}
        elif self.calls == 4:
            case.events = [Event('B'), Event('A')]
            case.attributes['label'] = {'anomaly': 'Insert', 'attr': {'indices': [0]}}
        else:
            case.events = [Event('Random activity 1'), *case.events]
            case.attributes['label'] = {'anomaly': 'Insert', 'attr': {'indices': [0]}}


class AnomalyPoolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root/'raw'; self.raw.mkdir()
        self.xes = self.root/'xes'; self.xes.mkdir()
        self.splits = self.root/'splits'
        self.output = self.root/'outputs'
        self.config = self.root/'pool_config.json'
        self.config.write_text(json.dumps({'base_seed':42, 'capacities':{'BPIC12':8}}))

    def fixture(self):
        path = self.raw/'BPIC12.csv'
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f); w.writerow(['case_id','name','concept:name','timestamp','extra'])
            sequences=[('A+start','B')+('A+start',)*i for i in range(10)]
            sequences += [sequences[0]]*3
            for i, seq in enumerate(sequences):
                for activity in ['▶',*seq,'■']:
                    w.writerow([str(i),activity,activity.split('+')[0],'2026-01-01T00:00:00','quoted,"value"\nline'])
        (self.xes/'BPIC12.xes').write_text('<log><global scope="event"><string key="concept:name" value=""/></global><trace/></log>')
        split.main(['--datasets','BPIC12','--repeats','1','--input-dir',str(self.raw),'--output-dir',str(self.splits)])

    def args(self,*extras):
        return ['--datasets','BPIC12','--raw-dir',str(self.raw),'--xes-dir',str(self.xes),
                '--split-dir',str(self.splits),'--output-dir',str(self.output),
                '--pool-config',str(self.config),*extras]

    def digests(self):
        return {str(p.relative_to(self.output)):split.file_sha256(p) for p in self.output.rglob('*') if p.is_file()}

    def test_odd_even_counts_and_dataset_seeds(self):
        self.assertEqual([pool.anomaly_count(n) for n in (1,2,3,4,947)], [1,1,2,2,474])
        with self.assertRaises(ValueError): pool.anomaly_count(0)
        seeds={pool.derive_seed(d,42,p) for d in split.DATASETS for p in ('generation','order')}
        self.assertEqual(len(seeds),14)

    def test_reject_normal_empty_unchanged_and_other_normal_variant(self):
        mothers=[Case(id='mother',events=[Event('A'),Event('B')],label='normal')]
        original=mothers[0].json
        state=np.random.get_state()
        cases, provenance, report=pool.generate_pool(mothers,{('A','B'),('B','A')},[ScriptedInjector()],
                                                     'BPIC12',2,42)
        self.assertEqual(report['attempts'],6)
        self.assertEqual(report['rejections'],{'normal_label':1,'empty_sequence':1,'normal_variant_collision':2})
        self.assertEqual(report['duplicate_variant_cases'],1)
        self.assertEqual(len({c['id'] for c in cases}),2)
        self.assertEqual(mothers[0].json,original)
        self.assertTrue(all(p['source_case_id']=='mother' for p in provenance))
        self.assertTrue(np.array_equal(state[1],np.random.get_state()[1]))
        again=pool.generate_pool(mothers,{('A','B'),('B','A')},[ScriptedInjector()],'BPIC12',2,42)
        self.assertEqual((cases,provenance,report),again)

    def test_exhaustion_records_reasons(self):
        mothers=[Case(id='m',events=[Event('A'),Event('B')],label='normal')]
        with self.assertRaises(pool.PoolExhausted) as caught:
            pool.generate_pool(mothers,{('A','B')},[ScriptedInjector()],'BPIC12',2,42,max_attempts=2)
        self.assertEqual(caught.exception.report['attempts'],2)
        self.assertEqual(caught.exception.report['accepted'],0)

    def test_full_pipeline_prefix_labels_reproducibility_and_overwrite(self):
        self.fixture()
        before={str(p):split.file_sha256(p) for p in self.splits.rglob('*') if p.is_file()}
        pool.main(self.args())
        original=self.digests()
        pm=json.loads((self.output/'anomaly_pools/BPIC12/pool_manifest.json').read_text())
        self.assertEqual(pm['capacity'], 8)  # Keep the configured pool even when six suffice.
        ids=[c['pool_case_id'] for c in pm['ordered_cases']]
        manifests=list((self.output/'mixed_testsets').glob('BPIC12/seed_*/observed_*/test_manifest.json'))
        self.assertEqual(len(manifests),5)
        for path in manifests:
            m=json.loads(path.read_text())
            self.assertEqual(m['anomaly_case_ids'],ids[:pool.anomaly_count(m['normal_traces'])])
            self.assertEqual(m['unknown_normal_activities'],[])
            self.assertTrue(all(m['validation'][k] for k in ('read_back_verified','inference_reader_verified','event_labels_verified')))
            with gzip.open(path.parent/m['file']['filename'],'rt') as f: data=json.load(f)
            self.assertEqual(len(data['cases']),m['total_traces'])
            self.assertTrue(all(c['attributes']['label']=='normal' for c in data['cases'][:m['normal_traces']]))
            self.assertTrue(all(isinstance(c['attributes']['label'],dict) for c in data['cases'][m['normal_traces']:]))
            # Exercise the existing EventLog and Dataset label loading without writing caches.
            from dcad.dataset import Dataset
            log=pool.EventLog.from_json(path.parent/m['file']['filename'])
            targets,labels=Dataset._get_classes_and_labels_from_event_log(log)
            self.assertEqual(len(labels),m['total_traces'])
            self.assertEqual(int(np.any(targets[:,:,0],axis=1).sum()),m['anomalous_traces'])
        with self.assertRaises(FileExistsError): pool.main(self.args())
        pool.main(self.args('--overwrite'))
        self.assertEqual(original,self.digests())
        self.assertEqual(before,{str(p):split.file_sha256(Path(p)) for p in before})
        self.assertFalse(list(self.output.glob('.anomaly_stage_*')))
        self.assertFalse(list(self.output.glob('.anomaly_publish_*')))

    def test_input_hash_change_refused_before_output(self):
        self.fixture()
        path=next(self.splits.glob('BPIC12/seed_*/observed_*/*_variant_test.csv'))
        with path.open('a') as f:f.write('\n')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            pool.main(self.args())
        self.assertFalse(self.output.exists())

    def test_interrupted_stage_refused_and_recoverable(self):
        self.fixture()
        leftover=self.output/'.anomaly_stage_BPIC12_interrupted'
        leftover.mkdir(parents=True)
        with self.assertRaisesRegex(FileExistsError,'interrupted generation'):
            pool.main(self.args())
        pool.main(self.args('--overwrite'))
        self.assertFalse(leftover.exists())

    def test_failed_readback_never_publishes_log(self):
        folder=self.root/'staged';folder.mkdir()
        with patch.object(pool,'verify_log',side_effect=ValueError('read-back failed')):
            with self.assertRaises(ValueError):
                pool.publish_log(folder,'log.json.gz',{},lambda:iter([]))
        self.assertFalse((folder/'log.json.gz').exists())
        self.assertTrue((folder/'log.json.gz.tmp.json.gz').exists())


if __name__=='__main__':
    unittest.main()
