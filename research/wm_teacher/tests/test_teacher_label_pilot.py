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
        self.auth=self.write('auth.json',dict(schema='teacher_training_admission_authorization_v1',
            stage_caps_GPUh={'label_rebuild_and_cache':24,'single_gpu_training_resume':2,'multi_node_technical_validation':6},
            total_cap_GPUh=32,formal_candidate_training_budget_approved=False,phase_borrowing_allowed=False,
            AR_allowed=False,student_or_distillation_allowed=False,previous_experiment_budgets_reused=False,
            training_started=False,failed_allocations_charged=True))
        self.patch('AUTH',Path(self.auth['path']));self.patch('AUTH_SHA256',self.auth['sha256'])
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
                original_prompt_sha256=hashlib.sha256(b'original synthetic prompt').hexdigest(),local_video=str(self.local/f'video{i}.mp4')))
        self.runtime=dict(python=sys.executable,packages={'torch':'fixture'},tools={'timeout':'/usr/bin/timeout'},files=[],caveat='synthetic')
        self.create_history()
        self.create_recovery_authorization()
        self.plan=dict(attempt_id=pilot.ATTEMPT_ID,local_root=str(self.local),**pilot.freeze_history(),
            new_reserved_seconds=4*pilot.CAP,schema=pilot.SCHEMA,private=True,prompt_version=pilot.caption.PROMPT_VERSION,
            authorization=self.auth,recovery_authorization=self.recovery_auth,manifest=self.manifest,
            source_files=self.sources,runtime=self.runtime,nodes=self.models,jobs=self.jobs,
            allocations=[dict(rank=r,gpu_index=0,allocated_gpus=1,cap_seconds=pilot.CAP,job_indices=list(range(r*4,r*4+4))) for r in range(4)],
            stage='label_rebuild_and_cache',stage_cap_seconds=86400,pilot_reserved_seconds=3600,retry_allowed=False,
            recovery_attempts_allowed=1,full_label_rebuild_allowed=False,AR_allowed=False,student_distillation_allowed=False,
            metadata_eligible_rows=1815,metadata_rejected_rows=41,selection_seed=20261101,safety_filter_before_selection=True,
            GPU_execution_started=False,schema_validation_is_semantic_review=False,
            independent_semantic_review_pass=False,formal_training_allowed=False)
        patch=mock.patch.object(pilot,'runtime',return_value=self.runtime);patch.start();self.addCleanup(patch.stop)

    def patch(self,name,value):
        patch=mock.patch.object(pilot,name,value);patch.start();self.addCleanup(patch.stop)

    def create_recovery_authorization(self):
        self.recovery_auth=self.write('recovery-auth.json',dict(
            schema='teacher_label_v3_one_recovery_authorization_v1',user_choice='批准一次v3恢复',authorized_at='2026-11-02T14:39:46.815Z',
            original_training_authorization_path=str(pilot.AUTH),original_training_authorization_sha256=pilot.AUTH_SHA256,
            prompt_version='e3-grounded-recaption-v3',fixed_sample_count=16,same_selection_seed=20261101,
            same_previously_selected_cases=True,allocated_gpus_per_node=1,gpu_index=0,nodes=4,cap_seconds_per_node=810,
            pilot_total_cap_seconds_including_all_prior=3600,previous_failed_seconds=70.82982997084036,
            previous_completed_seconds=259.7616256037727,all_previous_costs_preserved=True,
            charged_to_stage='label_rebuild_and_cache',stage_cap_GPUh=24,new_directory_required=True,
            new_source_and_publication_gate_required=True,independent_review_required=True,recovery_attempts_allowed=1,
            automatic_retry_allowed=False,full_label_rebuild_allowed=False,formal_training_allowed=False,
            AR_allowed=False,student_distillation_allowed=False))
        self.patch('RECOVERY_AUTH',Path(self.recovery_auth['path']));self.patch('RECOVERY_AUTH_SHA256',self.recovery_auth['sha256'])

    def create_history(self):
        self.patch('EVIDENCE_ROOT',self.root)
        first=[]
        for rank in range(4):
            first.append(self.write(f'attempt1_rank{rank}.json',dict(
                cost=dict(rank=rank,charged_upper_seconds=pilot.PRIOR_COST_SECONDS[rank],release_verified=True,budget_exceeded=False),
                parent=dict(rank=rank,released=True,success=False),CUDA_initialized=False,labels_written=0)))
        self.patch('PRIOR_FIRST_SHA256',tuple(r['sha256'] for r in first))
        old_jobs=copy.deepcopy(self.jobs)
        for j in old_jobs:j['local_video']=str(pilot.OLD_LOCAL_ROOTS[1]/('video%d.mp4' % j['index']))
        old=self.write('old-plan.json',dict(schema='teacher_label_pilot_execution_v2',attempt_id='label_pilot_attempt2',
            prompt_version='e3-grounded-recaption-v2',selection_seed=20261101,jobs=old_jobs,manifest=self.manifest,
            source_files=[dict(path='obsolete-source-not-to-reverify',bytes=1,sha256='0'*64)]))
        self.patch('OLD_PLAN',Path(old['path']));self.patch('OLD_PLAN_SHA256',old['sha256'])
        second_cost=[70.11004818044603,59.32944968063384,66.33598428685218,63.98614345584065]
        audit_nodes=[];audit_records=[]
        for rank in range(4):
            base=self.root/f'attempt2_collected/rank{rank}';base.mkdir(parents=True)
            labels=[]
            for j in self.jobs[rank*4:rank*4+4]:
                label=dict(sample_id=j['sample_id'],video_sha256=j['video']['sha256'],
                    original_prompt_sha256=j['original_prompt_sha256'],prompt_version='e3-grounded-recaption-v2',
                    semantic_review_pass=False,caption='Historical synthetic caption',facts_json={'motion':'at frame 9'},
                    frame_indices=list(range(8)))
                row=self.write(str(base.relative_to(self.root)/('label%d.json' % j['index'])),label);labels.append(label)
                audit_records.append(dict(index=j['index'],label=row,video=j['video'],semantic_review_pass=False))
            worker=self.write(str(base.relative_to(self.root)/'worker.json'),dict(plan=old,rank=rank,records=labels,
                success=True,semantic_review_pass=False,formal_training_allowed=False))
            parent=self.write(str(base.relative_to(self.root)/'parent.json'),dict(schema='teacher_label_pilot_parent_v1',
                plan=old,rank=rank,success=True,released=True,exit_code=0,error=None,interrupted_signals=[],cap_seconds=860,
                release_queries=[dict(pids=[],returncode=0),dict(pids=[],returncode=0)],semantic_review_pass=False,
                formal_training_allowed=False,worker=dict(worker,path=str(pilot.OLD_LOCAL_ROOTS[1]/'worker.json'))))
            cost=self.write(str(base.relative_to(self.root)/'cost.json'),dict(plan=old,rank=rank,release_verified=True,
                budget_exceeded=False,charged_upper_seconds=second_cost[rank],elapsed_seconds=second_cost[rank]))
            audit_nodes.append(dict(rank=rank,released=True,parent=parent,cost=cost,
                charged_seconds=second_cost[rank],actual_seconds=second_cost[rank]))
        audit=self.write('completed-audit.json',dict(schema='teacher_label_pilot_actual_completion_v1',plan=old,
            nodes=audit_nodes,records=audit_records,labels_written=16,error_records=0,label_stage_cap_GPUh=24,pilot_cap_GPUh=1,
            semantic_review_pass=False,formal_training_started=False,previous_failed_seconds=sum(pilot.PRIOR_COST_SECONDS.values()),
            new_allocated_seconds=sum(second_cost),cumulative_allocated_GPUh=(sum(pilot.PRIOR_COST_SECONDS.values())+sum(second_cost))/3600))
        self.patch('COMPLETED_AUDIT',Path(audit['path']));self.patch('COMPLETED_AUDIT_SHA256',audit['sha256'])

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
        with mock.patch.object(pilot,'TAIL',pilot.CAP-1),mock.patch.object(pilot,'verify',return_value=self.plan), \
             mock.patch.object(pilot.process,'manager_boundary',return_value={'gpu_uuid':'CPU-NOT-A-GPU'}), \
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

    def preparation_rows(self):
        import random
        rows=[dict(sample_id=f'pool{i}',video=str(self.root/f'pool{i}.mp4'),
            captions=dict(orig='original synthetic prompt',fixed='fixed synthetic prompt')) for i in range(1815)]
        selected=random.Random(20261101).sample(range(1815),16)
        for index,position in enumerate(selected):
            j=self.jobs[index];rows[position]=dict(sample_id=j['sample_id'],video=j['video']['path'],
                captions=dict(orig='original synthetic prompt',fixed='fixed synthetic prompt'))
        rows += [dict(sample_id=f'excluded{i}',video=str(self.root/f'excluded{i}.mp4'),
            captions=dict(orig='private',fixed='private')) for i in range(41)]
        return rows

    def test_prepare_recovery_with_all_actual_cpu_proof_files_and_same_selection(self):
        paths={n['rank']:n['proof']['path'] for n in self.models}
        with mock.patch.object(pilot.prep,'manifest_rows',return_value=(self.manifest,self.preparation_rows())):
            row=pilot.prepare(paths,self.root/'prepared-plan.json')
        value=pilot.verify(row['path'],row['sha256'],0)
        self.assertEqual(value['jobs'],self.jobs)
        self.assertEqual(value['prior_failed_charged_seconds'],70.82982997084036)
        self.assertEqual(value['prior_completed_charged_seconds'],259.7616256037727)
        self.assertEqual(value['prior_attempt_charged_seconds'],330.59145557461306)
        self.assertEqual(value['new_reserved_seconds'],3240)
        self.assertLessEqual(value['prior_attempt_charged_seconds']+value['new_reserved_seconds'],3600)
        self.assertEqual(value['schema'],'teacher_label_pilot_execution_v3')
        self.assertEqual(value['prompt_version'],'e3-grounded-recaption-v3')
        self.assertFalse(value['independent_semantic_review_pass'])
        self.assertFalse(self.local.exists())

    def test_prepare_rejects_pool_count_and_changed_selected_identity(self):
        paths={n['rank']:n['proof']['path'] for n in self.models}
        cases=[self.preparation_rows()[:-1],self.preparation_rows()]
        for r in cases[1]:
            if r['sample_id']==self.jobs[0]['sample_id']:r['captions']['orig']='changed synthetic prompt'
        for rows in cases:
            with mock.patch.object(pilot.prep,'manifest_rows',return_value=(self.manifest,rows)), self.assertRaises(ValueError):
                pilot.prepare(paths,self.root/'rejected-plan.json')
        self.assertFalse((self.root/'rejected-plan.json').exists())

    def test_recovery_authorization_raw_sha_path_and_schema_tampering_denied(self):
        for field,value in (('recovery_attempts_allowed',2),('recovery_attempts_allowed',True),
                            ('automatic_retry_allowed',True),('formal_training_allowed',True),
                            ('full_label_rebuild_allowed',True),('cap_seconds_per_node',860),
                            ('previous_completed_seconds',0.0),('independent_review_required',False),
                            ('new_source_and_publication_gate_required',False),('schema','wrong')):
            bad=pilot.read(self.recovery_auth);bad[field]=value
            bad_row=self.write('bad-recovery.json',bad);plan=copy.deepcopy(self.plan);plan['recovery_authorization']=bad_row
            row=self.plan_row(plan)
            with self.assertRaises(ValueError):pilot.verify(row['path'],row['sha256'],0)
            with mock.patch.object(pilot,'RECOVERY_AUTH',Path(bad_row['path'])), \
                 mock.patch.object(pilot,'RECOVERY_AUTH_SHA256',bad_row['sha256']), self.assertRaises(ValueError):
                pilot.validate_authorizations(self.auth,bad_row)
        path=Path(self.recovery_auth['path']);path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaises(ValueError):pilot.validate_authorizations(self.auth,self.recovery_auth)

    def test_original_job_sample_video_prompt_index_rank_exactly_bound(self):
        for change in (lambda v:v['jobs'][0].update(sample_id='replacement'),
                       lambda v:v['jobs'][0]['video'].update(sha256='b'*64),
                       lambda v:v['jobs'][0].update(original_prompt_sha256='c'*64),
                       lambda v:v['jobs'][0].update(index=1),lambda v:v['jobs'][0].update(rank=1),
                       lambda v:v['jobs'].__setitem__(slice(0,2),v['jobs'][1::-1])):
            bad=copy.deepcopy(self.plan);change(bad);row=self.plan_row(bad)
            with self.assertRaises(ValueError):pilot.verify(row['path'],row['sha256'],0)

    def test_prior_cost_parent_worker_label_and_audit_raw_mutation_denied(self):
        history=pilot.freeze_history()
        rows=[history['prior_completed_attempt_audit'],history['original_attempt2_plan'],*history['prior_failed_attempt_proofs']]
        for n in history['prior_completed_attempt_rank_proofs']:
            rows.extend([n['cost'],n['parent'],n['worker'],*n['labels']])
        for proof in rows:
            path=Path(proof['path']);raw=path.read_bytes()
            try:
                path.write_bytes(raw+b' ')
                with self.assertRaises(ValueError):pilot.freeze_history()
            finally:path.write_bytes(raw)

    def test_rebound_plan_proof_substitution_and_double_prior_budget_drift_denied(self):
        for change in (lambda v:v['prior_completed_attempt_rank_proofs'][0].update(cost=v['prior_completed_attempt_rank_proofs'][1]['cost']),
                       lambda v:v['prior_completed_attempt_rank_proofs'][0]['labels'].pop(),
                       lambda v:v['prior_failed_attempt_proofs'].pop(),lambda v:v.update(prior_failed_charged_seconds=0),
                       lambda v:v.update(prior_completed_charged_seconds=0),lambda v:v.update(prior_attempt_charged_seconds=70.82982997084036),
                       lambda v:v.update(prior_completed_attempt_audit=self.auth),lambda v:v.update(original_attempt2_plan=self.auth)):
            bad=copy.deepcopy(self.plan);change(bad);row=self.plan_row(bad)
            with self.assertRaises(ValueError):pilot.verify(row['path'],row['sha256'],0)

    def test_historical_v2_labels_are_only_raw_identity_evidence_never_v3_validation(self):
        with mock.patch.object(pilot.caption,'validate_facts',side_effect=AssertionError('Do not relabel historical facts')):
            h=pilot.freeze_history();row=self.plan_row();pilot.verify(row['path'],row['sha256'],0)
        label=pilot.read(h['prior_completed_attempt_rank_proofs'][0]['labels'][0])
        self.assertEqual(label['facts_json']['motion'],'at frame 9')
        self.assertFalse(self.plan['independent_semantic_review_pass'])

    def test_old_local_roots_and_existing_fresh_attempt_never_reset_or_reused(self):
        row=self.plan_row()
        for root in pilot.OLD_LOCAL_ROOTS:
            with mock.patch.object(pilot,'LOCAL',root),self.assertRaises(ValueError):
                pilot.verify(row['path'],row['sha256'],0)
        self.local.mkdir();marker=self.local/'old-work';marker.write_bytes(b'preserve')
        with self.assertRaises(ValueError):pilot.stage(row['path'],row['sha256'],0)
        self.assertEqual(marker.read_bytes(),b'preserve')
        pilot.write_new(self.local/'stage.json',dict(plan=row,rank=0))
        pilot.write_new(self.local/'launch.json',dict(previous_attempt=True))
        with mock.patch.object(pilot.subprocess,'Popen') as popen, self.assertRaises(ValueError):
            pilot.run(row['path'],row['sha256'],0,self.root/'manager',True)
        popen.assert_not_called()

    def test_recovery_cap_tail_scope_and_other_rank_model_proof_tampering_denied(self):
        row=self.plan_row()
        for name,value in (('CAP',860),('TAIL',89)):
            with mock.patch.object(pilot,name,value),self.assertRaises(ValueError):pilot.verify(row['path'],row['sha256'],0)
        for change in (lambda v:v.update(recovery_attempts_allowed=2),lambda v:v.update(full_label_rebuild_allowed=True),
                       lambda v:v.update(schema_validation_is_semantic_review=True),lambda v:v.update(stage='single_gpu_training_resume'),
                       lambda v:v.update(local_root=str(pilot.OLD_LOCAL_ROOTS[1])),lambda v:v['nodes'][3].update(model_root=str(self.root))):
            bad=copy.deepcopy(self.plan);change(bad);row=self.plan_row(bad)
            with self.assertRaises(ValueError):pilot.verify(row['path'],row['sha256'],0)

    def test_gpu_lease_flags_cannot_be_split_across_ancestors_or_dangle(self):
        launch,ancestors=self.ancestry_fixture();lease=ancestors[1]
        left=copy.deepcopy(lease);right=copy.deepcopy(lease)
        left['argv'][left['argv'].index('--state-dir')+1]='wrong-state'
        right['argv'][right['argv'].index('--gpus')+1]='1'
        with self.assertRaises(ValueError):pilot.verify_worker_ancestry(launch,self.plan,0,self.local/'launch.json',self.root/'manager',[ancestors[0],left,right])
        lease['argv']=lease['argv'][:lease['argv'].index('--state-dir')+1]
        with self.assertRaises(ValueError):pilot.verify_worker_ancestry(launch,self.plan,0,self.local/'launch.json',self.root/'manager',ancestors)

    def test_live_cpu_proc_nested_lease_prefix_and_repeated_worker_state_flags(self):
        launch,ancestors=self.ancestry_fixture();parent=pilot.process.identity(os.getpid());launch['parent']=parent
        args=[sys.executable,'-B','-c','import time;time.sleep(30)',str(ROOT/'scripts/gpu_hold.py'),
              '--action','exec','--gpus','0','--state-dir',str(self.root/'manager'),'--',
              '/usr/bin/timeout','--signal=TERM','--kill-after=20s','720s',sys.executable,
              str(ROOT/'scripts/run_teacher_label_pilot.py'),'worker','--state-dir','different-child-state']
        child=subprocess.Popen(args,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),start_new_session=True)
        try:
            actual=pilot.process.identity(child.pid)
            self.assertTrue(pilot.verify_worker_ancestry(launch,self.plan,0,self.local/'launch.json',self.root/'manager',[actual,parent]))
            with self.assertRaises(ValueError):pilot.verify_worker_ancestry(launch,self.plan,0,self.local/'launch.json',self.root/'different',[actual,parent])
        finally:child.terminate();child.wait(timeout=5)

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
