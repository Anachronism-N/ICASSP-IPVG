import importlib.util,subprocess,sys
from pathlib import Path
from unittest.mock import patch
import pytest
S=Path(__file__).resolve().parents[1]/'scripts/run_teacher_a800_exact_resume.py'
spec=importlib.util.spec_from_file_location('admission',S);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def test_help_and_import_do_not_initialize_model():
    r=subprocess.run([sys.executable,str(S),'--help'],capture_output=True);assert r.returncode==0
    code='import runpy,sys;runpy.run_path('+repr(str(S))+',run_name="not_main");assert "torch" not in sys.modules'
    assert subprocess.run([sys.executable,'-c',code],capture_output=True).returncode==0

def test_exact_cfg_fixed_physical_and_logical_scope():
    cfg=a.cfg();assert cfg['max_steps']==cfg['horizon']==4 and cfg['warmup']==1
    assert cfg['offload'] is True and cfg['anchor_lambda']==.5 and cfg['lora_rank']==cfg['lora_alpha']==32
    assert cfg['per_rank_batch']==cfg['grad_accum']==1
    assert a.CAP==5040

def test_unsigned_allocation_rejected_before_claim(tmp_path):
    with patch.object(a,'E',tmp_path):
        a.save(tmp_path/'authorization.json',{'schema':'unsigned'})
        with pytest.raises(AssertionError):a.authorize()
        assert not (tmp_path/'attempt.claim').exists()

def test_checkpoint_evidence_writer_never_overwrites(tmp_path):
    row=a.save(tmp_path/'evidence.json',{'exact':True})
    with pytest.raises(FileExistsError):a.save(tmp_path/'evidence.json',{'exact':False})
    assert a.record(row['path'])==row


def test_revision002_is_separate_and_keeps_old_failure_costs():
    with patch.object(a,'E',a.E),patch.object(a,'AUTH_SHA',a.AUTH_SHA),          patch.object(a,'REVISION002',a.REVISION002),patch.object(a,'PRIOR_TECHNICAL_GPUH',a.PRIOR_TECHNICAL_GPUH):
        a.configure_revision002()
        assert a.E.name=='a800_exact_resume002' and a.REVISION002
        assert a.PRIOR_TECHNICAL_GPUH==2.0 and a.CAP==5040


def test_real_training_interfaces_fail_before_model_load():
    import types
    with pytest.raises(AttributeError):a.training_interfaces(types.SimpleNamespace(),'cpu')
    import ast
    worker=next(n for n in ast.parse(S.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='worker')
    calls=[n for n in ast.walk(worker) if isinstance(n,ast.Call)]
    contract=next(n.lineno for n in calls if isinstance(n.func,ast.Name) and n.func.id=='training_interfaces')
    load=next(n.lineno for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='build_pipe')
    assert contract<load


def test_real_API_CPU_reference_prefix_fresh_resume_exact(tmp_path):
    import random
    import torch
    path=S.parents[1]/'tests/test_e3_train.py'
    spec=importlib.util.spec_from_file_location('tiny_e3_contract',path)
    tiny=importlib.util.module_from_spec(spec);spec.loader.exec_module(tiny)
    tr=tiny.tr
    torch.set_num_threads(2)
    config=a.cfg()
    with patch.object(tr,'LORA_RANK',tiny.TINY_RANK),patch.object(tr,'LORA_ALPHA',tiny.TINY_RANK):
        def segment(name,offset,count,previous=None):
            random.seed(tr.derive_seed('smoke_python',20261104))
            pipe=tiny.build_tiny_pipe(seed=20261104)
            params=tr.adapter_list(pipe.dit)
            optimizer=torch.optim.AdamW(params,lr=config['lr'],weight_decay=config['weight_decay'])
            if previous is not None:
                tr.load_checkpoint(previous['path'],previous,pipe.dit,optimizer,offset,
                                   cuda=False,expected_config=config)
            order,comm=a.training_interfaces(tr,'cpu')
            rows=[]
            for step in range(offset,offset+count):
                index=order.index(step,0,1)
                row=tr.train_step(pipe,params,optimizer,tiny.tiny_sample(index),step,index,
                                  config,comm,autocast=tiny.autocast)
                assert row['status']=='ok' and row['grad_norm_total_before_clip']>0
                rows.append(row)
            state=tr.checkpoint_state(pipe.dit,optimizer,offset+count,
                                      dict(config,completed_steps=offset+count),cuda=False)
            tr.validate_checkpoint(state,pipe.dit,optimizer,offset+count)
            saved=tr.save_checkpoint(tmp_path/(name+'.pt'),state)
            return state,saved,rows
        reference,_,left=segment('reference',0,4)
        _,prefix,_=segment('prefix',0,2)
        resumed,_,right=segment('resume',2,2,prefix)
        for key in ('adapters','optimizer','rng'):
            assert tr.ts.state_hash(reference[key])==tr.ts.state_hash(resumed[key])
        for x,y in zip(left[2:],right):
            for key in ('step','lr','grad_norm_total_before_clip','loss_fm_mean',
                        'loss_anchor_mean','loss_total_mean','per_rank'):
                assert x[key]==y[key]


def test_revision003_preserves_both_failed_full_resume_allocations(monkeypatch):
    for k in ('E','AUTH_SHA','REVISION002','REVISION003','PRIOR_TECHNICAL_GPUH'):
        monkeypatch.setattr(a,k,getattr(a,k))
    a.configure_revision003()
    assert a.E.name=='a800_exact_resume003' and a.PRIOR_TECHNICAL_GPUH==3.4
    assert a.CAP==5040 and a.REVISION003 and a.PRIOR_TECHNICAL_GPUH+1.4<=8


def test_revision003_controller_drift_is_rejected_before_claim(tmp_path,monkeypatch):
    import types
    import teacher_gpu_exclusive as guard
    monkeypatch.setattr(a,'E',tmp_path);monkeypatch.setattr(a,'REVISION003',True)
    monkeypatch.setattr(a,'authorize',lambda:None)
    monkeypatch.setattr(sys,'argv',[str(S)])
    row=a.save(tmp_path/'plan.json',{'resource_control_runtime':{'pinned':True}})
    args=types.SimpleNamespace(plan=row['path'],sha=row['sha256'])
    monkeypatch.setattr(guard,'control_runtime',lambda:{'pinned':False})
    with patch.object(a.subprocess,'Popen') as spawn,pytest.raises(AssertionError):a.run(args)
    spawn.assert_not_called();assert not (tmp_path/'attempt.claim').exists()
