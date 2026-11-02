"""Synthetic CPU validation; never Qwen model loading or caption quality PASS."""
import copy
import hashlib
import json
import os
import signal
import time
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_teacher_label_pilot as pilot


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.local=self.root/'local'
        patch=mock.patch.object(pilot,'LOCAL',self.local);patch.start();self.addCleanup(patch.stop)
        self.auth=self.write('auth.json',dict(stage_caps_GPUh={'label_rebuild_and_cache':24,'single_gpu_training_resume':2,'multi_node_technical_validation':6},
                       total_cap_GPUh=32,formal_candidate_training_budget_approved=False,phase_borrowing_allowed=False))
        self.manifest=self.write('manifest.json',{'synthetic_only':True})
        self.sources=[]
        for name in pilot.SOURCE_NAMES:
            self.sources.append(pilot.record(ROOT/name,nonempty=(ROOT/name).stat().st_size>0))
        self.models=[]
        model_root=self.root/'model';model_root.mkdir()
        patch=mock.patch.object(pilot,'MODEL_ROOT',model_root);patch.start();self.addCleanup(patch.stop)
        files=[]
        for name in pilot.MODEL_FILES:
            model=model_root/name;model.write_bytes(b'original-model');files.append(dict(pilot.record(model),relative=name))
        identity=[{k:r[k] for k in ('relative','bytes','sha256')} for r in files]
        digest=hashlib.sha256(pilot.encoded(identity)).hexdigest()
        for rank in range(4):
            proof=self.write(f'modelproof{rank}.json',dict(schema='teacher_label_model_content_v1',rank=rank,
                hostname=os.uname().nodename if rank==0 else f'other-host-{rank}',models=files,model_root=str(model_root),content_sha256=digest,
                GPU_used=False,weight_shards=5,model_type='qwen2_5_vl'))
            self.models.append(dict(rank=rank,hostname=os.uname().nodename if rank==0 else f'other-host-{rank}',proof=proof,
                model_root=str(model_root),model_files=files))
        self.jobs=[]
        for i in range(16):
            path=self.root/f'source{i}.mp4';path.write_bytes(b'video-source-'+str(i).encode())
            self.jobs.append(dict(index=i,rank=i//4,sample_id=f'synthetic{i}',video=pilot.record(path),
                original_prompt_sha256='a'*64,local_video=str(self.local/f'video{i}.mp4')))
        self.runtime=dict(python=sys.executable,packages={'torch':'fixture'},tools={'timeout':'/usr/bin/timeout'},files=[],caveat='synthetic')
        prior_rows=[]
        for rank in range(4):
            prior_rows.append(self.write(f'prior{rank}.json',dict(cost={'charged_upper_seconds':pilot.PRIOR_COST_SECONDS[rank],'release_verified':True},
                parent={'released':True,'success':False},CUDA_initialized=False,labels_written=0)))
        self.plan=dict(attempt_id=pilot.ATTEMPT_ID,prior_attempt_proofs=prior_rows,
            prior_attempt_charged_seconds=sum(pilot.PRIOR_COST_SECONDS.values()),new_reserved_seconds=4*pilot.CAP,schema=pilot.SCHEMA,private=True,prompt_version=pilot.caption.PROMPT_VERSION,authorization=self.auth,manifest=self.manifest,
            source_files=self.sources,runtime=self.runtime,nodes=self.models,jobs=self.jobs,
            allocations=[dict(rank=r,gpu_index=0,allocated_gpus=1,cap_seconds=pilot.CAP,job_indices=list(range(r*4,r*4+4))) for r in range(4)],
            stage_cap_seconds=86400,pilot_reserved_seconds=3600,retry_allowed=False,independent_semantic_review_pass=False,formal_training_allowed=False)
        patch=mock.patch.object(pilot,'runtime',return_value=self.runtime);patch.start();self.addCleanup(patch.stop)

    def write(self,name,value):
        path=self.root/name;path.write_bytes(pilot.encoded(value));return pilot.record(path)

    def plan_row(self,value=None):
        row=self.write('plan.json',self.plan if value is None else value);return row

    def test_current_model_and_source_proofs_verify_without_cuda(self):
        row=self.plan_row();self.assertEqual(pilot.verify(row['path'],row['sha256'],0),self.plan)
        self.assertIsNotNone(self.plan['allocations'])
        self.assertFalse(self.plan['formal_training_allowed'])

    def test_wrong_rank_host_and_plan_sha_denied(self):
        row=self.plan_row()
        for rank,sha in ((4,row['sha256']),(1,row['sha256']),(0,'wrong')):
            with self.assertRaises(ValueError):pilot.verify(row['path'],sha,rank)

    def test_same_size_model_and_media_content_drift_refused(self):
        row=self.plan_row();path=Path(self.models[0]['model_files'][0]['path']);stat=path.stat()
        path.write_bytes(b'changed--model');os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        with self.assertRaises(ValueError):pilot.verify(row['path'],row['sha256'],0)

    def test_caps_duplicate_and_path_substitution_refused(self):
        for change in (lambda v:v['allocations'][0].update(cap_seconds=pilot.CAP+1),lambda v:v['jobs'][1].update(sample_id=v['jobs'][0]['sample_id']),
                       lambda v:v['jobs'][0].update(local_video=str(self.root/'outside.mp4')),
                       lambda v:v.update(independent_semantic_review_pass=True),lambda v:v.update(formal_training_allowed=True)):
            value=copy.deepcopy(self.plan);change(value);row=self.plan_row(value)
            with self.assertRaises(ValueError):pilot.verify(row['path'],row['sha256'],0)

    def test_auth_budget_cannot_be_borrowed(self):
        value=pilot.read(self.auth);value['stage_caps_GPUh']['label_rebuild_and_cache']=25
        auth=self.write('bad-auth.json',value);plan=copy.deepcopy(self.plan);plan['authorization']=auth;row=self.plan_row(plan)
        with self.assertRaises(ValueError):pilot.verify(row['path'],row['sha256'],0)

    def test_cpu_local_stage_exact_source_sha_and_no_overwrite(self):
        row=self.plan_row();result=pilot.stage(row['path'],row['sha256'],0)
        self.assertFalse(pilot.read(result)['GPU_used'])
        for j in self.jobs[:4]:self.assertEqual(pilot.record(j['local_video'])['sha256'],j['video']['sha256'])
        before=Path(self.jobs[0]['local_video']).read_bytes()
        with self.assertRaises(ValueError):pilot.stage(row['path'],row['sha256'],0)
        self.assertEqual(Path(self.jobs[0]['local_video']).read_bytes(),before)

    def test_stage_refuses_source_drift_without_cuda(self):
        row=self.plan_row();source=Path(self.jobs[0]['video']['path']);source.write_bytes(b'corrupt')
        with self.assertRaises(ValueError):pilot.stage(row['path'],row['sha256'],0)

    def test_model_root_missing_replaced_and_partial_asset_set_denied(self):
        proof=pilot.read(self.models[0]['proof'])
        for change in (lambda v:v.pop('model_root'),lambda v:v.update(model_root=str(self.root)),lambda v:v['models'].pop(),
                       lambda v:v['models'][0].update(path=str(self.root/'another-model.bin'))):
            bad=copy.deepcopy(proof);change(bad)
            with self.assertRaises(ValueError):pilot.validate_model_proof(bad,0)

    def ancestry_fixture(self):
        parent=dict(pid=12345,session=12345,start_ticks=777,argv=['python','approved-parent'])
        lease=dict(pid=12346,session=12346,start_ticks=888,argv=['python',str(ROOT/'scripts/gpu_hold.py'),
            '--action','exec','--gpus','0','--state-dir',str(self.root/'manager')])
        launch=dict(rank=0,cap_seconds=pilot.CAP,state_dir=str(self.root/'manager'),gpu_index=0,
            start_monotonic=100.,start_epoch=1000.,parent=parent)
        return launch,[parent,lease]

    def test_exact_ancestor_parent_and_gpu_lease_binding(self):
        launch,ancestors=self.ancestry_fixture()
        self.assertTrue(pilot.verify_worker_ancestry(launch,self.plan,0,self.local/'launch.json',self.root/'manager',ancestors))
        for change in (lambda a:a.pop(0),lambda a:a[0].update(start_ticks=999),
                       lambda a:a[1]['argv'].__setitem__(a[1]['argv'].index('--gpus')+1,'1')):
            bad=copy.deepcopy(ancestors);change(bad)
            with self.assertRaises(ValueError):pilot.verify_worker_ancestry(launch,self.plan,0,self.local/'launch.json',self.root/'manager',bad)
        with self.assertRaises(ValueError):pilot.verify_worker_ancestry(launch,self.plan,0,self.root/'other.json',self.root/'manager',ancestors)

    def test_real_exec_lease_command_with_repeated_child_flags_is_bound_to_prefix(self):
        launch,ancestors=self.ancestry_fixture()
        ancestors[1]['argv']+=['--','timeout','--signal=TERM','--kill-after=20s',str(pilot.CAP-pilot.TAIL)+'s',
            'python',str(ROOT/'scripts/run_teacher_label_pilot.py'),'worker','--rank','0','--state-dir',str(self.root/'manager')]
        self.assertTrue(pilot.verify_worker_ancestry(launch,self.plan,0,self.local/'launch.json',self.root/'manager',ancestors))
        ancestors[1]['argv'][ancestors[1]['argv'].index('--gpus')+1]='1'
        with self.assertRaises(ValueError):pilot.verify_worker_ancestry(launch,self.plan,0,self.local/'launch.json',self.root/'manager',ancestors)

    def test_previous_attempt_cost_and_remaining_cap_cannot_be_zeroed(self):
        value=copy.deepcopy(self.plan);value['prior_attempt_charged_seconds']=0;row=self.plan_row(value)
        with self.assertRaises(ValueError):pilot.verify(row['path'],row['sha256'],0)
        value=copy.deepcopy(self.plan);value['new_reserved_seconds']=3600;row=self.plan_row(value)
        with self.assertRaises(ValueError):pilot.verify(row['path'],row['sha256'],0)

    def test_run_without_explicit_authorization_cannot_start(self):
        with mock.patch.object(pilot.subprocess,'Popen') as popen, self.assertRaises(ValueError):
            pilot.run('missing','wrong',0,self.root)
        popen.assert_not_called()

    def test_repeated_term_during_cleanup_still_reaps_owned_child_and_saves_cost(self):
        row=self.plan_row();pilot.stage(row['path'],row['sha256'],0)
        real_popen=subprocess.Popen;real_cleanup=pilot.process.cleanup_owned
        children=[]
        def spawn(*args,**kwargs):
            child=real_popen([sys.executable,'-B','-c','import time;time.sleep(30)'],start_new_session=True)
            children.append(child);return child
        def cleanup(child,identity,deadline,tracked):
            os.kill(os.getpid(),signal.SIGTERM)
            os.kill(os.getpid(),signal.SIGTERM)
            return real_cleanup(child,identity,time.monotonic()+3,tracked)
        before={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
        with mock.patch.object(pilot,'TAIL',pilot.CAP-1),mock.patch.object(pilot.process,'manager_boundary',return_value={'gpu_uuid':'CPU-NOT-A-GPU'}), \
             mock.patch.object(pilot.subprocess,'Popen',side_effect=spawn), \
             mock.patch.object(pilot.process,'cleanup_owned',side_effect=cleanup), \
             mock.patch.object(pilot.process,'query_release',return_value={'pids':[]}):
            with self.assertRaisesRegex(ValueError,'Pilot failed'):
                pilot.run(row['path'],row['sha256'],0,self.root/'manager',True)
        for child in children:self.assertIsNotNone(child.poll())
        parent=pilot.read(pilot.record(self.local/'parent.json'))
        self.assertFalse(parent['success'])
        self.assertEqual(parent['interrupted_signals'],[signal.SIGTERM,signal.SIGTERM])
        self.assertTrue(parent['released'])
        self.assertTrue((self.local/'cost.json').exists())
        for sig,value in before.items():self.assertEqual(signal.getsignal(sig),value)

    def test_json_duplicate_and_nonfinite_denied(self):
        for raw in (b'{"x":1,"x":2}',b'{"x":NaN}'):
            path=self.root/'bad.json';path.write_bytes(raw)
            with self.assertRaises(ValueError):pilot.read(pilot.record(path))

    def test_import_cli_has_no_torch_or_model_load(self):
        r=subprocess.run([sys.executable,'-B','-c',
            "import sys;sys.path.insert(0,'scripts');import run_teacher_label_pilot;assert 'torch' not in sys.modules"],
            cwd=ROOT,capture_output=True,text=True,timeout=10,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        self.assertEqual(r.returncode,0,r.stderr)


if __name__=='__main__':unittest.main()
