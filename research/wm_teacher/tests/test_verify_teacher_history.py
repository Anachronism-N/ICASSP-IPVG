"""CPU synthetic fixtures authenticate full old logs and preserve unknown costs."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import verify_teacher_history as history


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'eval').mkdir()
        self.old = self.root / 'old'
        self.old.mkdir()
        self.plan_path = self.old / 'plan.json'
        self.plan_path.write_bytes(history.encoded(dict(method='alg', nshards=16, video_count=188,
            seeds=[0, 1], phase_seconds=dict(generation=86400))))
        self.ledger = self.old / 'allocations.json'
        self.ledger.write_bytes(b'{damaged-old-ledger')
        self.audit_path = self.root / 'eval/audit.json'
        self.audit = dict(schema='teacher_devset_readonly_race_audit_v1', originals_modified=False,
            unknown_shards=[], counts=dict(mp4=0), plan_record=history.record(self.plan_path),
            ledger_record=history.record(self.ledger), shards=[dict(shard=s,node_rank=1 if s<8 else 2,
                cost_upper_bound_seconds=1.) for s in range(16)])
        self.audit_path.write_bytes(history.encoded(self.audit))
        streams, hashes = [], {}
        for rank, task in ((1, 'first'), (2, 'second')):
            for stream in ('stdout', 'stderr'):
                path = self.old / f'{task}.{stream}.log'
                path.write_bytes(history.encoded(dict(method='alg',rank=rank,success=False)) if stream=='stdout'
                                 else b'Original final failure retained')
                streams.append(dict(task_id=task,stream=stream,path=str(path),raw_log='Do not trust excerpts'))
                hashes[(task,stream)] = history.record(path)['sha256']
        self.index_path = self.root / 'eval/index.json'
        self.index_path.write_bytes(history.encoded(dict(schema='teacher_devset_original_tool_logs_readonly_v1',
            originals_modified=False, records=streams, task_summaries=[dict(task_id='first',rank=1,method='alg',reported_duration_seconds=2),
                                                                     dict(task_id='second',rank=2,method='alg',reported_duration_seconds=3)])))
        for obj, name, value in ((history,'AUDITS',{'alg':('audit.json',history.record(self.audit_path)['sha256'],16.,{1:('first',2),2:('second',3)})}),
                                (history,'INDEX','index.json'),(history,'INDEX_SHA',history.record(self.index_path)['sha256']),
                                (history,'LOG_SHAS',hashes)):
            patch = mock.patch.object(obj,name,value)
            patch.start(); self.addCleanup(patch.stop)
        self.inventories = {}
        for rank in (1,2):
            path = self.root / f'inventory{rank}.json'
            value = dict(schema='teacher_old_run_inventory_v1',node_rank=rank,hostname=f'actual-fixture-{rank}',
                         root=str(self.root),checked_epoch=time.time(),matches=[],old_scripts=sorted(history.OLD_SCRIPTS),
                         verifier_source=history.record(history.__file__),GPU_used=False,signals_sent=False)
            path.write_bytes(history.encoded(value)); self.inventories[rank]=history.record(path)
        self.output = self.root / 'retirement.json'

    def proof(self):
        row = history.prepare_retirement('alg',self.inventories,self.output,self.root)
        proof = json.loads(Path(row['path']).read_bytes())
        plan = dict(root=str(self.root),method='alg',prior_generation={k:proof[k] for k in ('audit','old_plan','old_ledger')})
        return proof,plan

    def test_full_raw_evidence_not_excerpts_and_damaged_ledger_preserved(self):
        before = self.ledger.read_bytes()
        proof,plan = self.proof()
        self.assertEqual(len(proof['shards']),16)
        self.assertEqual(proof['old_generation_upper_seconds'],16.)
        self.assertIsNone(proof['old_actual_seconds'])
        self.assertEqual(history.verify_retirement(proof,plan),proof)
        self.assertEqual(self.ledger.read_bytes(),before)
        self.assertEqual(len(proof['raw_tool_logs']),4)

    def test_cannot_zero_or_change_old_upper(self):
        proof,plan=self.proof()
        for field,value in (('old_actual_seconds',0),('old_generation_upper_seconds',0)):
            changed=copy.deepcopy(proof);changed[field]=value
            with self.assertRaises(ValueError):history.verify_retirement(changed,plan)

    def test_booleans_without_full_records_do_not_retire(self):
        proof,plan=self.proof()
        for field,value in (('raw_tool_logs',[]),('termination_inventory',[]),('source_files',[]),('shards',[])):
            changed=copy.deepcopy(proof);changed[field]=value
            with self.assertRaises(ValueError):history.verify_retirement(changed,plan)

    def test_same_size_log_changed_even_restored_mtime(self):
        proof,plan=self.proof()
        path=Path(proof['raw_tool_logs'][0]['file']['path'])
        stat=path.stat();raw=path.read_bytes();path.write_bytes(b'x'*len(raw));os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        with self.assertRaises(ValueError):history.verify_retirement(proof,plan)

    def test_missing_full_log_refuses_index_excerpt(self):
        path=Path(json.loads(self.index_path.read_bytes())['records'][0]['path'])
        path.unlink()
        with self.assertRaises(OSError):history.verified_old_evidence('alg',self.root)

    def test_live_old_process_evidence_refuses_retirement(self):
        row=self.inventories[1];path=Path(row['path']);value=json.loads(path.read_bytes())
        value['matches']=[dict(pid=12345,start_ticks=1)]
        path.write_bytes(history.encoded(value));self.inventories[1]=history.record(path)
        with self.assertRaises(ValueError):history.prepare_retirement('alg',self.inventories,self.output,self.root)

    def test_wrong_node_identity_and_duplicate_host_refused(self):
        row=self.inventories[2];path=Path(row['path']);value=json.loads(path.read_bytes());value['hostname']='actual-fixture-1'
        path.write_bytes(history.encoded(value));self.inventories[2]=history.record(path)
        with self.assertRaises(ValueError):history.prepare_retirement('alg',self.inventories,self.output,self.root)

    def test_no_overwrite_or_forged_audit_digest(self):
        self.proof()
        before=self.output.read_bytes()
        with self.assertRaises(ValueError):history.prepare_retirement('alg',self.inventories,self.output,self.root)
        self.assertEqual(self.output.read_bytes(),before)
        changed=json.loads(self.audit_path.read_bytes());changed['shards'][0]['cost_upper_bound_seconds']=0
        self.audit_path.write_bytes(history.encoded(changed))
        with self.assertRaises(ValueError):history.verified_old_evidence('alg',self.root)

    def test_relative_old_plan_is_resolved_against_process_cwd(self):
        proc=self.root/'fake_proc';entry=proc/'1234567';entry.mkdir(parents=True)
        args=['python','run_teacher_devset_node.py','--plan','eval/teacher_alg_devset_20261026/plan.json']
        (entry/'cmdline').write_bytes(bytes([0]).join(a.encode() for a in args)+bytes([0]))
        fields=['S','1','1234567','1234567']+['0']*15+['100']
        (entry/'stat').write_text('1234567 (python) '+' '.join(fields))
        (entry/'cwd').symlink_to(self.root)
        with mock.patch.object(history,'PROC_ROOT',proc):
            value=history.inventory(1,self.root)
        self.assertEqual(len(value['matches']),1)
        self.assertEqual(value['matches'][0]['pid'],1234567)

    def test_ancient_inventory_cannot_authorize_new_retirement(self):
        proof,plan=self.proof()
        path=Path(proof['termination_inventory'][0]['file']['path'])
        value=json.loads(path.read_bytes());value['checked_epoch']=1
        path.write_bytes(history.encoded(value))
        proof['termination_inventory'][0]['file']=history.record(path)
        with self.assertRaises(ValueError):history.verify_retirement(proof,plan)

    def test_inventory_is_cpu_readonly_and_source_bound(self):
        value=history.inventory(0,self.root)
        self.assertFalse(value['GPU_used']);self.assertFalse(value['signals_sent'])
        self.assertEqual(value['verifier_source'],history.record(history.__file__))
        self.assertEqual(value['node_rank'],0)


if __name__=='__main__':
    unittest.main()
