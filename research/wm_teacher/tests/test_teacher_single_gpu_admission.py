"""CPU-only admission regression: real harmless trees and fresh tiny CPU resume.

Mocks at the GPU boundary NEVER constitute GPU release/admission evidence.
No CUDA imports at module top, no real base weights and no test skips.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_teacher_single_gpu_admission as ad
import train_e3_lora as tr
import run_teacher_generation as process
import prepare_teacher_training_admission as admission

REAL_POPEN = subprocess.Popen
REAL_RUN = subprocess.run
UUID = 'GPU-cpu-test-boundary-only'


def gpu_boundary(*args, **kwargs):
    expected = args[2] if len(args)>2 else kwargs.get('expected_uuid')
    if expected is not None and expected != UUID: raise ValueError('Actual GPU UUID differs')
    return dict(gpu_uuid=UUID, manager=dict(pid=os.getpid(), start_ticks=process.identity(os.getpid())['start_ticks']),
                query=cpu_empty_query(UUID))


def cpu_empty_query(uuid):
    start = time.monotonic()
    result = REAL_RUN([sys.executable, '-B', '-c', 'pass'], capture_output=True, text=True, check=True)
    return dict(command=process.QUERY_COMMAND, gpu_uuid=uuid, returncode=result.returncode, stdout=result.stdout,
                stderr=result.stderr, pids=[], start_monotonic=start, end_monotonic=time.monotonic())


def deterministic_authorization_fixture(parent):
    return ad.write_new(Path(parent)/'det_auth.json', dict(
        schema='teacher_deterministic_recovery_authorization_v1', user_reply='CPU fixture only', in_response_to='CPU fixture only',
        assistant_bounded_scope=dict(new_GPUh_cap=2, allocated_cards=1, wall_cap_seconds=7200, one_attempt=True,
            reference_updates=4, prefix_updates=2, fresh_process_resume_updates=2, physical_optimizer_updates=8,
            selected_attention_backend='same native FA3; explicit deterministic backward',
            backend_capability_probe='First reference update is the actual capability gate, counted within four reference updates; no extra uncharged probe',
            no_silent_backend_fallback=True), prior_failed_technical_GPUh_upper=2, prior_costs_preserved=True,
        original_training_stage_caps_GPUh=dict(label_rebuild_and_cache=24,single_gpu_training_resume=2,multi_node_technical_validation=6),
        new_scope_additional_not_borrowed=True, labels_budget_borrowing_allowed=False, formal_training_allowed=False,
        AR_allowed=False, student_distillation_allowed=False, requires_new_source_runtime_publication_and_exact_resume_gate=True,
        automatic_retry_allowed=False))


def backend_fixture(parent):
    interface=Path(parent)/'fa3_interface.py'; interface.write_text('def flash_attn_func(*args, deterministic=False): pass\n')
    native=Path(parent)/'native_C.so'; native.write_bytes(b'CPU fixture native bytes, never GPU backend')
    return dict(backend='flash_attn_3', interface=ad.record(interface), native=ad.record(native), torch_version='CPU fixture')


def prior_fixture(parent, charged=7200, closure=False):
    old = Path(parent) / 'old_failed_allocation'
    old.mkdir()
    plan = ad.write_new(Path(parent)/'old_plan.json', dict(cpu_fixture_only=True))
    previous = ad.write_new(old/'parent.json', dict(schema='teacher_single_gpu_admission_v1', phase=ad.PHASE,
        execution_success=False, success=False, released=True, comparison=None,
        error='ValueError: Exact resumed trajectory differs: adapters', plan=plan,
        segments=[dict(steps=s,segment_updates=u) for s,u in ((4,4),(2,2),(4,2))]))
    failure = ad.write_new(old/'cost.failure.json', dict(schema='teacher_single_gpu_cost_failure_v1', phase=ad.PHASE,
        terminal=True, success=False, released_observed=True, parent_receipt=previous, plan=plan,
        charged_upper_seconds=charged, allocated_GPUh=charged/3600, cap_seconds=7200))
    provisional = ad.write_new(old/'cost.json', dict(parent_receipt=previous, success=False,
        charged_upper_seconds=charged, allocated_GPUh=charged/3600))
    if closure: ad.write_new(old/'cost.closure.json', dict(success=True))
    audit = ad.write_new(Path(parent)/'old_audit.json', dict(schema='teacher_single_gpu_actual_failure_audit_v1',
        stage=ad.PHASE, charged_upper_GPUh=2, physical_updates=8, closure_present=False,
        terminal_success=False, release_verified_at_audit=True, parent=previous, failure=failure,
        provisional_cost=provisional, plan=plan))
    return old, audit


def fake_plan(root):
    return dict(schema=ad.SCHEMA, phase=ad.PHASE, hostname=os.uname().nodename, output_root=str(root),
                gpu_uuid=UUID, state_dir=str(ad.STATE_DIR), config=ad.fixed_config(),
                prior_failed_allocation=dict(cpu_fixture_only=True),
                runtime=dict(python=sys.executable, tools=dict(timeout=shutil.which('timeout'))))


def tiny_child(out, steps, source=None):
    # Actual native trainer loss/run_loop/AdamW/checkpoint/export on a small CPU
    # fixture. This test entrypoint never invokes trainer.execute/CUDA setup.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('_admission_tiny_fixture', ROOT / 'tests/test_e3_train.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    tr.LORA_RANK = tr.LORA_ALPHA = fixture.TINY_RANK
    out = Path(out)
    out.mkdir()
    (out/'private').mkdir()
    (out/'export').mkdir()
    random.seed(tr.derive_seed('smoke_python', 20261102))
    pipe = fixture.build_tiny_pipe(seed=20261102, dtype=torch.float32)
    params = tr.adapter_list(pipe.dit)
    optimizer = torch.optim.AdamW(params, lr=5e-5, weight_decay=.01, betas=(.9,.999), eps=1e-8)
    cfg = ad.fixed_config()
    # Fixture LoRA rank is intentionally tiny and never native-base evidence.
    cfg.update(lora_rank=fixture.TINY_RANK, lora_alpha=fixture.TINY_RANK)
    offset = 0
    if source:
        record = json.loads((Path(source)/'checkpoint.json').read_text())
        offset = 2
        tr.load_checkpoint(record['path'], record, pipe.dit, optimizer, offset, expected_config=cfg)
    initial = tr.ts.state_hash(tr.ts.trainable_adapters(pipe.dit))
    before = tr.ts.frozen_state_hash(pipe.dit)
    allocation = tr.budget(2,1,60,240,120)
    rows = []
    done, records, stop = tr.run_loop(pipe,optimizer,params,fixture.tiny_sample_fp32,tr.DataOrder(20261102,64),
                                     cfg,tr.Comm(),allocation,lambda:0,rows.append,steps=steps,step_offset=offset)
    assert done == steps and stop is None and len(rows) == steps
    assert all(r['status']=='ok' and r['grad_norm_total_before_clip']>0 for r in rows)
    assert before == tr.ts.frozen_state_hash(pipe.dit)
    state = tr.checkpoint_state(pipe.dit,optimizer,offset+done,dict(cfg,completed_steps=offset+done))
    tr.validate_checkpoint(state,pipe.dit,optimizer,offset+done)
    record = tr.save_checkpoint(out/'private'/tr.CKPT_NAME,state)
    tr.export_lora(out/'export'/tr.EXPORT_NAME,state['adapters'])
    assert initial != tr.ts.state_hash(state['adapters'])
    ad.write_new(out/'checkpoint.json',record)
    ad.write_new(out/'evidence.json',dict(initial=initial,final=tr.ts.state_hash(state['adapters']),frozen=before,
                identity=tr.trainer_identity(),cpu_only=True,formal_training_allowed=False,rows=rows))
    assert not torch.cuda.is_initialized()


def cpu_harness(parent, mode):
    parent = Path(parent)
    root = parent/'allocation'
    plan = fake_plan(root)
    plan.update(source={},framework={},models={})
    plan['runtime']['files']=[]
    plan['authorization']=ad.write_new(parent/'auth.json',admission.AUTH_EXPECTED)
    plan['additional_deterministic_authorization']=deterministic_authorization_fixture(parent)
    path = parent/'plan.json'
    row = ad.write_new(path,plan)
    approval_path = parent/'approval.json'
    approval = ad.write_new(approval_path,dict(cpu_fixture_only=True))
    child_code = ("import subprocess,sys,time,os,json; "
                  "p=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(90)']); "
                  "print(json.dumps({'pid':os.getpid(),'grandchild':p.pid}),flush=True); "
                  + ("time.sleep(.2); sys.exit(3)" if mode in ('fail','third','savefail') else "time.sleep(90)"))
    def popen(command, **kwargs):
        if any(str(item).endswith('gpu_hold.py') for item in command):
            assert '--smoke-deterministic-fa3' in command
            assert kwargs['env']['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
            assert kwargs['env']['CUDA_VISIBLE_DEVICES']==UUID
            return REAL_POPEN([sys.executable,'-B','-c',child_code],**kwargs)
        return REAL_POPEN(command,**kwargs)
    real_write=ad.write_new
    def writer(path,value):
        if mode=='savefail' and Path(path).name=='parent.json': raise OSError('CPU injected parent save failure')
        return real_write(path,value)
    boundary_calls=[]
    def boundary(*args,**kwargs):
        boundary_calls.append(1)
        result=gpu_boundary(*args,**kwargs)
        if mode=='third' and len(boundary_calls)==3:
            result['query']['pids']=[42]
        return result
    with (
        patch.object(ad,'LOCAL',parent), patch.object(ad,'OUTPUT_ROOT',parent/'allocation'), patch.object(ad,'verify_plan',return_value=(plan,row)),
        patch.object(ad,'deterministic_authorization',return_value=plan['additional_deterministic_authorization']),
        patch.object(ad,'prior_failure',return_value=plan['prior_failed_allocation']),
        patch.object(ad,'verify_approval',return_value=approval),
        patch.object(process,'manager_boundary',side_effect=boundary),
        patch.object(process,'query_release',side_effect=cpu_empty_query),
        patch.object(ad.subprocess,'Popen',side_effect=popen),
        patch.object(ad,'write_new',side_effect=writer),
    ):
        try:
            ad.run(path,row['sha256'],approval_path,approval['sha256'],True)
        except ValueError:
            pass


def closure_harness(parent, mode):
    """CPU boundary fixtures exercise the real outer accounting and filesystem writes."""
    parent=Path(parent); root=parent/'allocation'; plan=fake_plan(root)
    plan.update(source={},framework={},models={})
    plan['runtime']['files']=[]
    plan['authorization']=ad.write_new(parent/'auth.json',admission.AUTH_EXPECTED)
    plan['additional_deterministic_authorization']=deterministic_authorization_fixture(parent)
    row=ad.write_new(parent/'plan.json',plan)
    approval=ad.write_new(parent/'approval.json',dict(cpu_only=True))
    real_write, real_clock, real_identity = ad.write_new, time.monotonic, process.identity
    real_mkdir=Path.mkdir
    def mkdir(path,*args,**kwargs):
        if mode=='mkdirfail' and path==root: raise OSError('CPU injected allocation mkdir failure')
        return real_mkdir(path,*args,**kwargs)
    offset=[0.0]
    def clock(): return real_clock()+offset[0]
    def writer(path,value):
        name=Path(path).name
        if name=='cost.json' and mode=='slowcost': offset[0]+=8000
        saved=real_write(path,value)
        if name=='cost.closure.json' and mode=='slowclosure': offset[0]+=8000
        if name=='cost.closure.json' and mode=='alarm': signal.raise_signal(signal.SIGALRM)
        return saved
    def identity(pid):
        if mode=='identityfail' and pid==os.getpid(): raise OSError('CPU injected parent identity failure')
        return real_identity(pid)
    def child(plan,name,updates,start,root,interrupted,proof=None):
        real_write(root/(name+'.launch.json'),dict(name=name,started_monotonic=clock()))
        receipt=dict(released=True)
        real_write(root/(name+'.parent.json'),receipt)
        return receipt
    def segment(plan,name,updates,cumulative,root,end):
        result=dict(trainer_identity=dict(pid=dict(reference=101,prefix=102,resume=103)[name],start_ticks=1),
                    frozen_base_sha256='base',initial_adapter_sha256='prefix' if name=='resume' else 'initial',
                    adapter_sha256='prefix' if name=='prefix' else 'final',steps=cumulative,segment_updates=updates,
                    backend_identity=dict(cpu_fixture=True), exact_trajectory=list(range(cumulative-updates,cumulative)))
        return result,dict(steps=cumulative)
    result,error=None,None
    with (
        patch.object(ad,'LOCAL',parent), patch.object(ad,'OUTPUT_ROOT',parent/'allocation'), patch.object(ad,'verify_plan',return_value=(plan,row)),
        patch.object(ad,'deterministic_authorization',return_value=plan['additional_deterministic_authorization']),
        patch.object(ad,'prior_failure',return_value=plan['prior_failed_allocation']),
        patch.object(ad,'verify_approval',return_value=approval),
        patch.object(process,'manager_boundary',side_effect=gpu_boundary),
        patch.object(process,'query_release',side_effect=cpu_empty_query),
        patch.object(process,'identity',side_effect=identity),
        patch.object(ad,'run_child',side_effect=child) as dispatch,
        patch.object(ad,'segment_state',side_effect=segment),
        patch.object(ad,'make_resume_proof',return_value=approval),
        patch.object(ad,'compare_states',return_value=dict(cpu_fixture=True)),
        patch.object(ad,'write_new',side_effect=writer),
        patch.object(ad.time,'monotonic',side_effect=clock),
        patch.object(Path,'mkdir',new=mkdir),
    ):
        try: result=ad.run(row['path'],row['sha256'],approval['path'],approval['sha256'],True)
        except ValueError as exc: error=str(exc)
        if mode in ('identityfail','mkdirfail'): assert dispatch.call_count==0
    return root,result,error


class AdmissionTests(unittest.TestCase):
    def test_help_is_stdlib_and_exposes_all_three_actions(self):
        code = "import sys;sys.path.insert(0,'scripts');import run_teacher_single_gpu_admission as a;assert 'torch' not in sys.modules;assert 'train_e3_lora' not in sys.modules;a.main(['--help'])"
        r = REAL_RUN([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('prepare,verify,run,verify-closure',r.stdout)
        self.assertIn('--approved-execution-approval-sha256',r.stdout)

    def test_authorization_complete_24_2_6_not_phase_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'auth.json'
            good=copy.deepcopy(admission.AUTH_EXPECTED)
            ad.write_new(path,good)
            self.assertEqual(admission.validate_authorization(path),ad.record(path))
            for field,value in [('total_cap_GPUh',2),('formal_candidate_training_budget_approved',True),
                                ('phase_borrowing_allowed',True),('stage_caps_GPUh',{'single_gpu_training_resume':2})]:
                bad=copy.deepcopy(good); bad[field]=value
                other=Path(tmp)/(field+'.json'); ad.write_new(other,bad)
                with self.assertRaises(ValueError): admission.validate_authorization(other)

    def test_actual_prepare_verify_freeze_and_source_model_gpu_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp); models=parent/'models'; models.mkdir(); framework=parent/'framework'; framework.mkdir()
            auth=parent/'auth.json'; ad.write_new(auth,admission.AUTH_EXPECTED)
            extra=deterministic_authorization_fixture(parent); old,audit=prior_fixture(parent)
            backend=backend_fixture(parent)
            rels=('base/shard1','base/shard2','base/shard3'); pins={}
            for i,rel in enumerate(rels):
                file=models/rel; file.parent.mkdir(exist_ok=True); file.write_bytes(('tiny CPU weight fixture '+str(i)).encode())
                pins[rel]=ad.record(file)['sha256']
            framework_names=tuple('module'+str(i)+'.py' for i in range(7))
            for name in framework_names: (framework/name).write_text('value = 1\n')
            source_names=('scripts/run_teacher_single_gpu_admission.py','tests/test_teacher_single_gpu_admission.py',
                          'scripts/train_e3_lora.py','scripts/run_teacher_training_smoke.py')
            files=[ad.asset_record(ROOT/name) for name in source_names]+[ad.asset_record(framework/name) for name in framework_names]
            files += [ad.asset_record(models/rel) for rel in rels]
            versions={name:ad.importlib.metadata.version(name) for name in ('torch','peft','numpy','safetensors','accelerate','transformers')}
            versions['flash_attn_3']='synthetic-test-only'  # backend_fixture is not the installed GPU backend.
            real_metadata_version=ad.importlib.metadata.version
            runtime=dict(python=str(Path(sys.executable).resolve()),packages=versions,
                         tools={name:sys.executable for name in ('timeout','nvidia-smi','git')},files=files,
                         actual_FA3=backend,required_flags=dict(tr.DET_FA3_FLAGS),required_environment=dict(tr.DET_FA3_ENV),
                         exhaustive_attestation=False,caveat='CPU fixture mocks selected attestation; not GPU permission')
            with patch.multiple(ad,LOCAL=parent,OUTPUT_ROOT=parent/'allocation',AUTH=auth), patch.object(ad,'DET_AUTH',Path(extra['path'])),                  patch.object(tr,'DET_FA3_AUTH',Path(extra['path'])),patch.object(tr,'DET_FA3_AUTH_SHA256',extra['sha256']),                  patch.object(ad,'PRIOR_ROOT',old),patch.object(ad,'PRIOR_AUDIT',Path(audit['path'])),                  patch.object(ad,'PRIOR_AUDIT_SHA256',audit['sha256']),patch.object(ad,'MODELS',models),                  patch.object(ad,'FRAMEWORK',framework),patch.object(ad,'SOURCE_NAMES',source_names),                  patch.object(tr,'SHARD_RELS',rels),patch.object(tr,'FRAMEWORK_FILES',framework_names),                  patch.object(tr.m1,'CACHE_PINS',pins),patch.object(ad,'runtime',return_value=runtime),                  patch.object(ad.shutil,'which',return_value=sys.executable),                  patch.object(ad.importlib.metadata,'version',side_effect=lambda n: versions['flash_attn_3'] if n=='flash_attn_3' else real_metadata_version(n)),                  patch.object(process,'manager_boundary',side_effect=gpu_boundary):
                plan=ad.prepare(parent/'allocation',ad.STATE_DIR)
                row=ad.write_new(parent/'plan.json',plan)
                verified,bound=ad.verify_plan(row['path'],row['sha256'])
                self.assertEqual(verified,plan); self.assertEqual(bound,row)
                self.assertFalse(plan['execution_allowed'] or plan['formal_training_allowed'])
                self.assertEqual(set(plan['trainer_source']),{'trainer','m1_helper','smoke_helper','admission_helper'})
                self.assertTrue(all(set(r)=={'path','bytes','sha256','mtime_ns','inode'} for r in plan['source'].values()))
                for field,value in [('phase','formal_training'),('cap_seconds',7201),('rank',1),('gpu_index',1),
                                    ('synthetic_samples',63),('formal_training_allowed',True),
                                    ('schema','teacher_single_gpu_admission_v1'),('one_attempt',True),
                                    ('prior_charged_seconds',0),('old_prior_upper_GPUh',0),('new_GPUh_cap',3),
                                    ('cumulative_upper_GPUh',2),('extra_GPU_probe_allowed',True),
                                    ('reference_updates',5),('prefix_updates',2.0),('restore_updates',3)]:
                    bad=dict(plan,**{field:value}); new=ad.write_new(parent/(field+'.json'),bad)
                    with self.assertRaises(ValueError): ad.verify_plan(new['path'],new['sha256'])
                self.assertEqual((plan['new_GPUh_cap'],plan['old_prior_upper_GPUh'],plan['cumulative_upper_GPUh']),(2,2,4))
                self.assertTrue(plan['GPU_capability_unverified'] and plan['first_reference_update_is_capability_probe'])
                for field in ('interface','native'):
                    bad=copy.deepcopy(plan);bad['runtime']['actual_FA3'][field]['sha256']='0'*64
                    new=ad.write_new(parent/('wrong_backend_'+field+'.json'),bad)
                    with self.assertRaises(ValueError):ad.verify_plan(new['path'],new['sha256'])
                bad=copy.deepcopy(plan);bad['runtime']['required_flags']['deterministic_algorithms']=1
                new=ad.write_new(parent/'wrongflagtype.json',bad)
                with self.assertRaises(ValueError):ad.verify_plan(new['path'],new['sha256'])
                bad=copy.deepcopy(plan);bad['prior_failed_allocation']['released_observed']=1
                new=ad.write_new(parent/'wrongpriortype.json',bad)
                with self.assertRaises(ValueError):ad.verify_plan(new['path'],new['sha256'])
                bad=copy.deepcopy(plan);bad['additional_deterministic_authorization']['sha256']='0'*64
                new=ad.write_new(parent/'wrongdet_auth.json',bad)
                with self.assertRaises(ValueError):ad.verify_plan(new['path'],new['sha256'])
                bad=copy.deepcopy(plan); bad['source'][source_names[0]]['sha256']='0'*64
                new=ad.write_new(parent/'source.json',bad)
                with self.assertRaises(ValueError): ad.verify_plan(new['path'],new['sha256'])
                bad=dict(plan,gpu_uuid='GPU-wrong-device'); new=ad.write_new(parent/'bad_gpu_uuid.json',bad)
                with self.assertRaises(ValueError): ad.verify_plan(new['path'],new['sha256'])
                with patch.object(process,'manager_boundary',side_effect=ValueError('GPU UUID changed')):
                    with self.assertRaises(ValueError): ad.verify_plan(row['path'],row['sha256'])
                (models/rels[0]).write_bytes(b'changed source fixture')
                with self.assertRaises(ValueError): ad.verify_plan(row['path'],row['sha256'])

    def test_fixed_config_windows_no_three_individual_budgets(self):
        cfg=ad.fixed_config()
        self.assertEqual((cfg['seed'],cfg['max_steps'],cfg['horizon'],cfg['warmup']), (20261102,4,4,1))
        self.assertEqual((cfg['anchor_lambda'],cfg['offload'],cfg['component_grad_every'],cfg['log_every']),(.5,True,10,1))
        self.assertEqual((ad.CAP,ad.TAIL,ad.CHILD_CAP),(7200,120,2000))
        with patch.object(ad.time,'monotonic',return_value=10): self.assertEqual(ad.child_window(0),2000)
        with patch.object(ad.time,'monotonic',return_value=6600): self.assertEqual(ad.child_window(0),480)
        with patch.object(ad.time,'monotonic',return_value=6700):
            with self.assertRaises(ValueError): ad.child_window(0)
        args=tr.parser().parse_args(ad.trainer_args(Path('/unused'),2,1000))
        self.assertLess(args.max_gpu_hours*3600,1000)
        self.assertEqual(args.smoke_offload,['on'])
        self.assertFalse(args.skip_model_hash or args.skip_frozen_hash)
        self.assertTrue(args.smoke_deterministic_fa3)
        self.assertTrue(all('--smoke-deterministic-fa3' in ad.trainer_args(Path('/unused'),u,1000)
                            for _,u,_ in ad.SEGMENTS))

    def test_evidence_writer_never_overwrites_existing_plan_or_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'receipt.json'; row=ad.write_new(path,dict(value=1))
            with self.assertRaises(ValueError): ad.write_new(path,dict(value=2))
            self.assertEqual(ad.record(path),row)
            self.assertEqual(ad.read(row),dict(value=1))

    def test_unknown_and_known_overrun_fees_never_clip(self):
        with patch.object(ad.time,'monotonic',return_value=100):
            self.assertEqual(ad.cost(0,False)['charged_upper_seconds'],7200)
            self.assertEqual(ad.cost(0,False)['prior_charged_seconds'],7200)
            self.assertEqual(ad.cost(0,False)['cumulative_upper_GPUh'],4)
            self.assertTrue(ad.cost(0,False)['budgets_independent'])
            self.assertEqual(ad.cost(0,True)['charged_upper_seconds'],100)
            known=ad.cost(0,True,publication_allowance=1)
            self.assertEqual(known['actual_elapsed_seconds'],100)
            self.assertEqual(known['elapsed_seconds'],100)
            self.assertEqual(known['charged_upper_seconds'],101)
            self.assertEqual(known['actual_allocated_GPUh'],100/3600)
            unknown=ad.cost(0,False,publication_allowance=1)
            self.assertEqual(unknown['charged_upper_seconds'],7200)
            self.assertIsNone(unknown['actual_allocated_GPUh'])
        with patch.object(ad.time,'monotonic',return_value=8000):
            for known in (False,True):
                c=ad.cost(0,known)
                self.assertEqual(c['charged_upper_seconds'],8000)
                self.assertTrue(c['budget_exceeded'])

    def test_wrong_phase_plan_rawsha_and_old_root_denied_before_cuda(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp); plan=fake_plan(parent/'allocation')
            plan['phase']='formal_training'
            row=ad.write_new(parent/'wrong.json',plan)
            with self.assertRaises(ValueError): ad.verify_plan(row['path'],row['sha256'])
            with self.assertRaises(ValueError): ad.verify_plan(row['path'],'0'*64)
            with patch.object(ad,'LOCAL',parent),patch.object(ad,'OUTPUT_ROOT',parent/'allocation'):
                ad.safe_root(parent/'allocation')
                with self.assertRaises(ValueError): ad.safe_root(parent/'fresh')
                with self.assertRaises(ValueError): ad.safe_root(parent/'teacher_single_gpu_admission_20261102_run01')
                (parent/'old').mkdir()
                for root in (parent,parent/'old',parent.parent/'foreign'):
                    with self.assertRaises(ValueError): ad.safe_root(root)
                (parent/'link').symlink_to(parent/'old',target_is_directory=True)
                with self.assertRaises(ValueError): ad.safe_root(parent/'link')

    def test_run_requires_both_authorization_and_main_thread(self):
        with self.assertRaises(ValueError): ad.run(None,None,None,None,False)
        import threading
        errors=[]
        def attempt():
            try: ad.run(None,None,None,None,True)
            except ValueError as e: errors.append(str(e))
        thread=threading.Thread(target=attempt); thread.start(); thread.join()
        self.assertEqual(len(errors),1)
        self.assertIn('main thread',errors[0])
        self.assertEqual(ad.main(['run','--plan','/unused','--approved-plan-sha256','0'*64,
                                 '--execution-approval','/unused','--gpu-authorized']),1)

    def test_raw_gpu_release_not_zero_self_claim(self):
        q=cpu_empty_query(UUID); q2=cpu_empty_query(UUID)
        self.assertTrue(ad.release_queries_valid([q,q2],UUID,q['start_monotonic']))
        missing_stderr=[{k:v for k,v in row.items() if k!='stderr'} for row in (q,q2)]
        self.assertTrue(ad.release_queries_valid(missing_stderr,UUID,q['start_monotonic']))
        with self.assertRaises(ValueError): ad.release_queries_valid([q,dict(q)],UUID,q['start_monotonic'])
        for mutation in (dict(stdout=UUID+', 42\n'),dict(command=['fake']),dict(pids=[42]),dict(returncode=1)):
            bad=dict(q,**mutation)
            with self.assertRaises(ValueError): ad.release_queries_valid([bad,bad],UUID,q['start_monotonic'])

    def test_external_compute_blocks_before_spawn(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); plan=fake_plan(root)
            with patch.object(process,'manager_boundary',side_effect=ValueError('Actual external compute')),                  patch.object(ad.subprocess,'Popen') as popen:
                with self.assertRaises(ValueError): ad.run_child(plan,'reference',4,time.monotonic(),root,ad.Interruptions())
                popen.assert_not_called()

    def test_real_harmless_owned_child_and_grandchild_cleanup(self):
        command=[sys.executable,'-B','-c',"import subprocess,sys,time; subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(90)']); time.sleep(90)"]
        child=REAL_POPEN(command,start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        owner=process.identity(child.pid); tracked=[]
        try:
            until=time.monotonic()+5
            while len(tracked)<2 and time.monotonic()<until:
                process.owned_members(owner,tracked); time.sleep(.02)
            self.assertGreaterEqual(len(tracked),2)
            signalled=process.cleanup_owned(child,owner,time.monotonic()+25,tracked)
            self.assertTrue(signalled)
            self.assertFalse(process.owned_members(owner,tracked))
            self.assertTrue(all(not process.same_process(r) for r in tracked))
            self.assertIsNotNone(child.returncode)
        finally:
            if child.poll() is None: process.cleanup_owned(child,owner,time.monotonic()+25,tracked)

    def harness(self, mode):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp)
            child=REAL_POPEN([sys.executable,'-B',str(Path(__file__)), '--cpu-harness',str(parent),mode],
                            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            if mode=='term':
                until=time.monotonic()+10
                log=parent/'allocation/reference.log'
                while (not log.is_file() or log.stat().st_size==0) and time.monotonic()<until: time.sleep(.02)
                self.assertTrue(log.is_file() and log.stat().st_size)
                for _ in range(12):
                    if child.poll() is None: os.kill(child.pid,signal.SIGTERM if _%2==0 else signal.SIGINT)
                    time.sleep(.01)
            stdout,stderr=child.communicate(timeout=35)
            self.assertEqual(child.returncode,0,stdout+stderr)
            root=parent/'allocation'
            receipt=ad.read(ad.record(root/'parent.json'))
            cost=ad.read(ad.record(root/'cost.json'))
            self.assertFalse(receipt['success'])
            if mode=='third':
                self.assertFalse(receipt['released'])
                self.assertGreaterEqual(cost['charged_upper_seconds'],ad.CAP)
            else:
                self.assertTrue(receipt['released'])
                self.assertGreaterEqual(cost['charged_upper_seconds'],ad.CAP)
            self.assertTrue(cost['provisional'])
            self.assertFalse(cost['success'])
            failure=ad.read(ad.record(root/'cost.failure.json'))
            self.assertFalse(failure['success'])
            self.assertGreaterEqual(failure['charged_upper_seconds'],ad.CAP)
            self.assertEqual(len(list(root.glob('*.launch.json'))),1)
            identity=json.loads((root/'reference.log').read_text().splitlines()[0])
            with self.assertRaises(OSError): process.identity(identity['pid'])
            try: grand=process.identity(identity['grandchild'])
            except OSError: grand=None
            self.assertTrue(grand is None or grand['state'] in ('Z','X'))
            if mode=='term': self.assertTrue(receipt['interrupted_signals'])

    def test_repeated_term_latches_and_cleans_real_tree(self): self.harness('term')
    def test_child_failure_stops_without_retry(self): self.harness('fail')
    def test_third_actual_release_query_occupied_retains_conservative_fee(self): self.harness('third')

    def test_parent_receipt_save_failure_still_preserves_conservative_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            result=REAL_RUN([sys.executable,'-B',str(Path(__file__)),'--cpu-harness',str(root),'savefail'],
                            capture_output=True,text=True,timeout=35)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            cost=ad.read(ad.record(root/'allocation/cost.json'))
            self.assertIsNone(cost['parent_receipt'])
            self.assertIn('CPU injected parent save failure',cost['parent_save_error'])
            self.assertFalse(cost['success'] or cost['released'])
            self.assertGreaterEqual(cost['charged_upper_seconds'],ad.CAP)
            parent=ad.read(ad.record(root/'allocation/reference.parent.json'))
            self.assertTrue(parent['released'])

    def test_parent_identity_failure_after_valid_plan_preserves_started_clock_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,result,error=closure_harness(tmp,'identityfail')
            self.assertIsNone(result); self.assertIn('Admission failed',error)
            parent=ad.read(ad.record(root/'parent.json'))
            self.assertIsNone(parent['owner']); self.assertTrue(parent['no_child_spawned'])
            self.assertIn('CPU injected parent identity failure',parent['error'])
            provisional=ad.read(ad.record(root/'cost.json'))
            failure=ad.read(ad.record(root/'cost.failure.json'))
            self.assertGreater(provisional['actual_elapsed_seconds'],0)
            self.assertGreaterEqual(failure['charged_upper_seconds'],ad.CAP)
            self.assertGreaterEqual(failure['charged_upper_seconds'],failure['actual_elapsed_seconds'])
            self.assertFalse(parent['success'] or provisional['success'] or failure['success'])

    def test_root_creation_failure_keeps_unique_parent_failure_cost_not_fake_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,result,error=closure_harness(tmp,'mkdirfail')
            self.assertIsNone(result); self.assertIn('Admission failed',error)
            self.assertFalse(root.exists())
            proofs=list(Path(tmp).glob('allocation.allocation-failure.*.json'))
            self.assertEqual(len(proofs),1)
            failure=ad.read(ad.record(proofs[0]))
            self.assertIn('CPU injected allocation mkdir failure',failure['error'])
            self.assertTrue(failure['no_child_spawned'])
            self.assertFalse(failure['success'])
            self.assertGreater(failure['actual_elapsed_seconds'],0)
            self.assertGreaterEqual(failure['charged_upper_seconds'],ad.CAP)
            with patch.object(ad,'LOCAL',Path(tmp)),patch.object(ad,'OUTPUT_ROOT',root),self.assertRaises(ValueError):
                ad.safe_root(root)

    def test_terminal_cost_save_stall_and_alarm_cannot_return_or_validate_success(self):
        for mode in ('slowcost','slowclosure','alarm'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root,result,error=closure_harness(tmp,mode)
                self.assertIsNone(result); self.assertIn('Admission failed',error)
                provisional=ad.read(ad.record(root/'cost.json'))
                failure=ad.read(ad.record(root/'cost.failure.json'))
                self.assertFalse(provisional['success'] or failure['success'])
                self.assertGreaterEqual(failure['charged_upper_seconds'],failure['actual_elapsed_seconds'])
                self.assertGreaterEqual(failure['charged_upper_seconds'],ad.CAP)
                if mode!='alarm':
                    self.assertGreater(failure['actual_elapsed_seconds'],8000)
                    self.assertTrue(failure['budget_exceeded'])
                if mode!='slowcost':
                    closure=ad.record(root/'cost.closure.json')
                    with self.assertRaisesRegex(ValueError,'Terminal publication failed'):
                        ad.verify_cost_closure(closure['path'],closure['sha256'])
                self.assertFalse(ad.read(ad.record(root/'parent.json'))['success'])

    def test_success_requires_bound_measured_terminal_closure_not_parent_bool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,result,error=closure_harness(tmp,'success')
            self.assertIsNone(error); self.assertEqual(Path(result['path']),root/'cost.closure.json')
            value=ad.verify_cost_closure(result['path'],result['sha256'])
            self.assertTrue(value['success'] and value['terminal'])
            self.assertEqual(value['charged_upper_seconds'],ad.CAP)
            self.assertLess(value['actual_elapsed_seconds']+value['publication_exit_allowance_seconds'],ad.CAP)
            self.assertFalse(ad.read(value['parent'])['success'])
            self.assertFalse(ad.read(value['cost'])['success'])
            self.assertEqual(ad.main(['verify-closure','--closure',result['path'],
                                     '--approved-closure-sha256',result['sha256']]),0)
            with self.assertRaises(ValueError): ad.verify_cost_closure(result['path'],'0'*64)

    def test_identity_failure_reaps_only_owned_popen_and_retains_unknown_fee(self):
        self.identity_failure(False)
        self.identity_failure(True)

    def identity_failure(self, malformed):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); plan=fake_plan(root)
            def popen(command,**kwargs):
                if any(str(item).endswith('gpu_hold.py') for item in command):
                    return REAL_POPEN([sys.executable,'-B','-c','import time;time.sleep(90)'],**kwargs)
                return REAL_POPEN(command,**kwargs)
            real_identity=process.identity
            def identity(pid):
                if pid!=os.getpid():
                    if malformed:
                        row=real_identity(pid); row['session']=-1
                        return row
                    raise OSError('unreadable original Popen identity')
                return real_identity(pid)
            with patch.object(process,'manager_boundary',side_effect=gpu_boundary),                  patch.object(process,'identity',side_effect=identity), patch.object(ad.subprocess,'Popen',side_effect=popen):
                with self.assertRaises(ValueError) as caught: ad.run_child(plan,'reference',4,time.monotonic(),root,ad.Interruptions())
                self.assertIn('Child failure/unknown release',str(caught.exception))
            receipt=ad.read(ad.record(root/'reference.parent.json'))
            self.assertFalse(receipt['released'])
            self.assertIsNotNone(receipt['exit_code'])
            self.assertGreaterEqual(ad.cost(time.monotonic(),receipt['released'])['allocated_GPUh'],2)

    def test_additional_authorization_raw_sha_and_exact_scope_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); good=deterministic_authorization_fixture(root)
            document=ad.read(good)
            def validate(row):
                with (patch.object(ad,'DET_AUTH',Path(row['path'])),
                      patch.object(tr,'DET_FA3_AUTH',Path(row['path'])),
                      patch.object(tr,'DET_FA3_AUTH_SHA256',row['sha256'])):
                    return ad.deterministic_authorization()
            self.assertEqual(validate(good),good)
            changes=[('new_GPUh_cap',True),('allocated_cards',2),('wall_cap_seconds',7200.0),
                     ('one_attempt',1),('reference_updates',5),('prefix_updates',True),
                     ('fresh_process_resume_updates',3),('physical_optimizer_updates',9),
                     ('no_silent_backend_fallback',1)]
            for i,(key,value) in enumerate(changes):
                bad=copy.deepcopy(document); bad['assistant_bounded_scope'][key]=value
                row=ad.write_new(root/('wrongscope'+str(i)+'.json'),bad)
                with self.subTest(key=key),self.assertRaises(ValueError): validate(row)
            for key,value in [('prior_failed_technical_GPUh_upper',0),('automatic_retry_allowed',True),
                              ('new_scope_additional_not_borrowed',False),('labels_budget_borrowing_allowed',True),
                              ('formal_training_allowed',True),('AR_allowed',True)]:
                row=ad.write_new(root/(key+'.json'),dict(document,**{key:value}))
                with self.subTest(key=key),self.assertRaises(ValueError): validate(row)
            with (patch.object(ad,'DET_AUTH',Path(good['path'])),patch.object(tr,'DET_FA3_AUTH',Path(good['path'])),
                  patch.object(tr,'DET_FA3_AUTH_SHA256','0'*64),self.assertRaises(ValueError)):
                ad.deterministic_authorization()

    def test_original_failed_allocation_raw_sha_fee_eight_calls_and_no_closure(self):
        for charged,closure in ((7200,False),(0,False),(7200.0,False),(7200,True)):
            with self.subTest(charged=charged,closure=closure),tempfile.TemporaryDirectory() as tmp:
                old,audit=prior_fixture(Path(tmp),charged,closure)
                with (patch.object(ad,'PRIOR_ROOT',old),patch.object(ad,'PRIOR_AUDIT',Path(audit['path'])),
                      patch.object(ad,'PRIOR_AUDIT_SHA256',audit['sha256'])):
                    if type(charged) is int and charged==7200 and not closure:
                        value=ad.prior_failure()
                        self.assertEqual((value['charged_upper_GPUh'],value['physical_optimizer_calls']),(2,8))
                        self.assertTrue(value['released_observed']);self.assertFalse(value['closure_present'])
                        with patch.object(ad,'PRIOR_AUDIT_SHA256','0'*64),self.assertRaises(ValueError): ad.prior_failure()
                        (old/'parent.json').write_bytes(b'changed actual original bytes')
                        with self.assertRaises(ValueError): ad.prior_failure()
                    else:
                        with self.assertRaises(ValueError): ad.prior_failure()

    def test_driver_fifteen_key_proof_is_accepted_by_real_cpu_trainer_and_exact_resume(self):
        import contextlib
        import torch
        spec=importlib.util.spec_from_file_location('_admission_det_fixture',ROOT/'tests/test_e3_train.py')
        fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        fixture.Base.setUpClass()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);extra=deterministic_authorization_fixture(root)
            auth=ad.write_new(root/'original_auth.json',admission.AUTH_EXPECTED)
            framework,wan,interface,native=fixture.fake_fa3_modules(root)
            runner=fixture.EndToEndTests()
            with (fixture.DeterministicFa3Tests().modules_patch(wan,interface,native),
                  patch.object(ad,'DET_AUTH',Path(extra['path'])),patch.object(tr,'DET_FA3_AUTH',Path(extra['path'])),
                  patch.object(tr,'DET_FA3_AUTH_SHA256',extra['sha256'])):
                for name,updates in (('reference',4),('prefix',2)):
                    argv=ad.trainer_args(root/name,updates,2000)+['--framework',str(framework)]
                    self.assertEqual(runner.run_main(root,argv),0)
                original=ad.read(ad.record(root/'prefix/run_manifest.json'))
                summary=ad.read(ad.record(root/'prefix/summary.json'))
                prefix=dict(steps=2,segment_updates=2,manifest=ad.record(root/'prefix/run_manifest.json'),
                    summary=ad.record(root/'prefix/summary.json'),checkpoint=summary['checkpoint'],
                    deterministic_backend=original['deterministic_backend'])
                proof=ad.make_resume_proof(dict(authorization=auth,additional_deterministic_authorization=extra),prefix,root)
                body=ad.read(proof)
                self.assertEqual(len(body),15)
                self.assertEqual(body['additional_deterministic_authorization'],extra)
                self.assertEqual(body['deterministic_backend'],ad.record(root/'prefix/deterministic_backend.json'))
                argv=ad.trainer_args(root/'resume',2,2000,proof)+['--framework',str(framework)]
                args=tr.parser().parse_args(argv)
                with contextlib.ExitStack() as stack:
                    for p in runner.patches(root):stack.enter_context(p)
                    self.assertEqual(tr.validate_smoke_resume(args)['proof'],proof)
                self.assertEqual(runner.run_main(root,argv),0)
                states={name:torch.load(root/name/'private'/tr.CKPT_NAME,map_location='cpu',weights_only=True)
                        for name in ('reference','resume')}
                self.assertTrue(ad.compare_states(states['reference'],states['resume'])['adapter_optimizer_rng_exact'])
                for key in ('adapters','optimizer','rng'):
                    self.assertEqual(tr.ts.state_hash(states['reference'][key]),tr.ts.state_hash(states['resume'][key]))
                self.assertFalse(torch.cuda.is_initialized())

    def test_actual_fresh_tiny_reference4_equals_prefix2_restore2(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name,steps,source in (('reference',4,None),('prefix',2,None),('restore',2,root/'prefix')):
                command=[sys.executable,'-B',str(Path(__file__)),'--tiny-child',str(root/name),str(steps)]
                if source: command.append(str(source))
                r=REAL_RUN(command,capture_output=True,text=True,timeout=90)
                self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            import torch
            states={}; evidence={}
            for name in ('reference','prefix','restore'):
                row=ad.read(ad.record(root/name/'checkpoint.json'))
                self.assertEqual(ad.record(row['path'])['sha256'],row['file_sha256'])
                states[name]=torch.load(row['path'],map_location='cpu',weights_only=True)
                self.assertEqual(tr.ts.state_hash(states[name]),row['state_sha256'])
                evidence[name]=ad.read(ad.record(root/name/'evidence.json'))
            self.assertEqual([states[n]['steps'] for n in states],[4,2,4])
            self.assertEqual([len(evidence[n]['rows']) for n in evidence],[4,2,2])
            self.assertEqual(evidence['prefix']['final'],evidence['restore']['initial'])
            self.assertEqual(evidence['reference']['initial'],evidence['prefix']['initial'])
            self.assertEqual(len({(e['identity']['pid'],e['identity']['start_ticks']) for e in evidence.values()}),3)
            self.assertEqual(len({e['frozen'] for e in evidence.values()}),1)
            self.assertTrue(ad.compare_states(states['reference'],states['restore'])['adapter_optimizer_rng_exact'])
            self.assertEqual(tr.ts.state_hash(states['reference']),tr.ts.state_hash(states['restore']))
            for state in states.values():
                self.assertEqual(set(state['rng']),{'python','torch_cpu','torch_cuda'})
                self.assertEqual(state['rng']['torch_cuda'],[])
                self.assertTrue(all(float(s['step'])==state['steps'] for s in state['optimizer']['state'].values()))
            bad=copy.deepcopy(states['restore']); key=next(iter(bad['adapters'])); bad['adapters'][key].add_(1)
            with self.assertRaises(ValueError): ad.compare_states(states['reference'],bad)
            bad=copy.deepcopy(states['restore']); bad['rng']['torch_cpu'][0] ^= 1
            with self.assertRaises(ValueError): ad.compare_states(states['reference'],bad)
            bad=copy.deepcopy(states['restore']); bad['rng']['python']=random.Random(7).getstate()
            with self.assertRaises(ValueError): ad.compare_states(states['reference'],bad)
            bad=copy.deepcopy(states['restore']); first=next(iter(bad['optimizer']['state'])); bad['optimizer']['state'][first]['step']+=1
            with self.assertRaises(ValueError): ad.compare_states(states['reference'],bad)
            self.assertFalse(torch.cuda.is_initialized())


class PublicationTests(unittest.TestCase):
    def test_actual_git_snapshot_can_differ_from_private_source_and_remote_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); repo=root/'repo'; repo.mkdir(); private=root/'private.py'; private.write_text('private host value\n')
            def git(*args):
                return REAL_RUN(['git','-C',str(repo),*args],capture_output=True,check=True).stdout
            git('init','-b',ad.BRANCH); git('config','user.email','cpu-test@example.invalid'); git('config','user.name','CPU test')
            snapshot=repo/ad.SNAPSHOT_PREFIX; (snapshot/'scripts').mkdir(parents=True)
            published=snapshot/'scripts/run_teacher_single_gpu_admission.py'; published.write_text('sanitized host value\n')
            source=ad.asset_record(private)
            manifest=snapshot/'PUBLICATION_MANIFEST.json'
            framework_source=ad.asset_record(ROOT/'DiffSynth-Studio/diffsynth/utils/lora/general.py')
            framework_path='DiffSynth-Studio/diffsynth/utils/lora/general.py'
            framework_snapshot=snapshot/framework_path; framework_snapshot.parent.mkdir(parents=True)
            framework_snapshot.write_text('sanitized framework fixture\n')
            extra=snapshot/'README.md'; extra.write_text('uncovered fixture snapshot\n')
            manifest_row=ad.write_new(manifest,dict(files=[
                dict(path='scripts/run_teacher_single_gpu_admission.py',source_sha256=source['sha256'],snapshot_sha256=ad.record(published)['sha256']),
                dict(path=framework_path,source_sha256=framework_source['sha256'],snapshot_sha256=ad.record(framework_snapshot)['sha256']),
                dict(path='README.md',source_sha256=ad.record(extra)['sha256'],snapshot_sha256=ad.record(extra)['sha256'])]))
            git('add','.'); git('commit','-m','CPU fixture publication')
            commit=git('rev-parse','HEAD').decode().strip()
            bare=root/'remote.git'; REAL_RUN(['git','clone','--bare',str(repo),str(bare)],check=True,capture_output=True)
            git('remote','add','origin',str(bare))
            receipt_content=dict(commit=commit,remote_equal=True,manifest_sha256=manifest_row['sha256'],
                                 changed_files_remote_bytes_verified=4,manifest_files_verified=3,quality_claim=False,sota_proven=False)
            receipt=ad.write_new(root/'remote_receipt.json',receipt_content)
            plan=dict(source={'scripts/run_teacher_single_gpu_admission.py':source},framework={'diffsynth/utils/lora/general.py':framework_source},
                      runtime=dict(tools=dict(git=shutil.which('git'))))
            plan_row=ad.write_new(root/'plan.json',plan)
            approval=dict(schema='teacher_single_gpu_execution_approval_v1',private=True,GPU_authorized=True,
                          plan=plan_row,source_git_root=str(repo),source_commit=commit,publication_manifest=manifest_row,
                          remote_receipt=receipt,phase=ad.PHASE,formal_training_allowed=False,
                          branch=ad.BRANCH,snapshot_prefix=ad.SNAPSHOT_PREFIX)
            approved=ad.write_new(root/'approval.json',approval)
            def actual_git(command,**kwargs):
                if command[-4:]==['remote','get-url','origin'] or command[-3:]==['remote','get-url','origin']:
                    return subprocess.CompletedProcess(command,0,(ad.ORIGIN+'\n').encode(),b'')
                return REAL_RUN(command,**kwargs)
            with patch.object(ad,'GIT_ROOT',repo),patch.object(ad.subprocess,'run',side_effect=actual_git):
                self.assertEqual(ad.verify_approval(approved['path'],approved['sha256'],plan_row,plan,time.monotonic()+60),approved)
                with self.assertRaises(ValueError): ad.verify_approval(approved['path'],'0'*64,plan_row,plan,time.monotonic()+60)
                for field,value in [('phase','formal_training'),('source_git_root',str(root)),('branch','other'),
                                    ('snapshot_prefix','wrong'),('source_commit','0'*40),('plan',dict(plan_row,sha256='0'*64))]:
                    bad=dict(approval,**{field:value}); row=ad.write_new(root/('bad_'+field+'.json'),bad)
                    with self.assertRaises(ValueError): ad.verify_approval(row['path'],row['sha256'],plan_row,plan,time.monotonic()+60)
                for field,value in [('commit','0'*40),('remote_equal',False),('manifest_sha256','0'*64),
                                    ('manifest_files_verified',2),('manifest_files_verified',3.0),
                                    ('changed_files_remote_bytes_verified',3)]:
                    bad_receipt=ad.write_new(root/('bad_receipt_'+field+str(value)+'.json'),dict(receipt_content) | {field:value})
                    bad_approval=ad.write_new(root/('bad_receipt_approval_'+field+str(value)+'.json'),dict(approval,remote_receipt=bad_receipt))
                    with self.assertRaises(ValueError): ad.verify_approval(bad_approval['path'],bad_approval['sha256'],plan_row,plan,time.monotonic()+60)
                bad_plan=copy.deepcopy(plan); bad_plan['framework']['diffsynth/utils/lora/general.py']['sha256']='0'*64
                with self.assertRaises(ValueError): ad.verify_approval(approved['path'],approved['sha256'],plan_row,bad_plan,time.monotonic()+60)
                def wrong_extra(command,**kwargs):
                    if command[-1]==commit+':'+ad.SNAPSHOT_PREFIX+'/README.md':
                        return subprocess.CompletedProcess(command,0,b'wrong unscoped snapshot bytes',b'')
                    return actual_git(command,**kwargs)
                with patch.object(ad.subprocess,'run',side_effect=wrong_extra):
                    with self.assertRaises(ValueError): ad.verify_approval(approved['path'],approved['sha256'],plan_row,plan,time.monotonic()+60)
                with patch.object(ad,'ORIGIN','https://wrong.invalid/repo.git'):
                    # Wrapper returns the previously pinned real origin, not the mutated constant.
                    def foreign(command,**kwargs):
                        if command[-3:]==['remote','get-url','origin']:
                            return subprocess.CompletedProcess(command,0,b'https://github.com/Anachronism-N/ICASSP-IPVG.git\n',b'')
                        return REAL_RUN(command,**kwargs)
                    with patch.object(ad.subprocess,'run',side_effect=foreign):
                        with self.assertRaises(ValueError): ad.verify_approval(approved['path'],approved['sha256'],plan_row,plan,time.monotonic()+60)
                # Real remote ref update to a different actual commit, no network.
                git('commit','--allow-empty','-m','new local state')
                other=git('rev-parse','HEAD').decode().strip()
                REAL_RUN(['git','--git-dir',str(bare),'fetch',str(repo),ad.BRANCH],check=True,capture_output=True)
                REAL_RUN(['git','--git-dir',str(bare),'update-ref','refs/heads/'+ad.BRANCH,other],check=True,capture_output=True)
                with self.assertRaises(ValueError): ad.verify_approval(approved['path'],approved['sha256'],plan_row,plan,time.monotonic()+60)
            self.assertNotEqual(source['sha256'],ad.record(published)['sha256'])


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--tiny-child':
        tiny_child(sys.argv[2],int(sys.argv[3]),sys.argv[4] if len(sys.argv)>4 else None)
    elif len(sys.argv)>1 and sys.argv[1]=='--cpu-harness':
        cpu_harness(sys.argv[2],sys.argv[3])
    else:
        unittest.main()
