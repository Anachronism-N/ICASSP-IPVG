#!/usr/bin/env python3
"""Independent full-DiT FA2 exact-resume admission. Synthetic only, not formal training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'eval/teacher_progress_campaign_20261104/a800_exact_resume001'
AUTH_SHA='2253a35b0f1a65dcf64fc2601071c7306e9e4f4a730867edac6a131635f17e5c'
CAMPAIGN_SHA='ffa19129fb3256e482553bb1b90a7f09edd95eed4391687e6b45426d0706b8fe'
PYTHON='/dockerdata/venv_qwen32b_20261103/bin/python3.12'
SITE='/dockerdata/wm_a800_fa2_runtime_20261103/site'
CAP=5040
REVISION002=False
PRIOR_TECHNICAL_GPUH=.6

def configure_revision002():
    global E,AUTH_SHA,REVISION002,PRIOR_TECHNICAL_GPUH
    E=ROOT/'eval/teacher_progress_campaign_20261104/a800_exact_resume002'
    AUTH_SHA='bfe1a2cabb914250c50599a5050a40b157fcce3ae18dc4a210027fdb103da6c9'
    REVISION002=True;PRIOR_TECHNICAL_GPUH=2.0

def training_interfaces(tr,device):
    for name in ('DataOrder','Comm','synthetic_sample','train_step','checkpoint_state',
                 'validate_checkpoint','save_checkpoint','load_checkpoint'):
        assert callable(getattr(tr,name)), 'Real trainer API missing: '+name
    return tr.DataOrder(20261104,4),tr.Comm(None,0,1,device)

def record(p):
    p=Path(p).resolve();before=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    after=p.stat();assert (before.st_size,before.st_ino,before.st_mtime_ns)==(after.st_size,after.st_ino,after.st_mtime_ns)
    return dict(path=str(p),bytes=after.st_size,sha256=h.hexdigest())

def read(row):
    assert record(row['path'])==row;return json.loads(Path(row['path']).read_bytes())

def save(p,v):
    b=(json.dumps(v,sort_keys=True,allow_nan=False)+chr(10)).encode()
    with Path(p).open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    Path(p).chmod(0o600);return record(p)

def authorize():
    row=record(E/'authorization.json');assert row['sha256']==AUTH_SHA
    a=read(row);assert record(E.parent/'campaign.authorization.json')['sha256']==CAMPAIGN_SHA
    assert a['wall_cap_seconds']==CAP and a['upper_GPUh']==1.4 and a['phase']=='technical_admission'
    assert a['formal_training_allowed'] is False and a['automatic_retry_allowed'] is False
    assert a['prior_new_technical_GPUh']==PRIOR_TECHNICAL_GPUH and PRIOR_TECHNICAL_GPUH+1.4<=8
    if REVISION002:
        assert a['revision']==2 and a['real_training_API_CPU_test_required']
        previous=E.parent/'a800_exact_resume001/terminal.json'
        assert record(previous)['sha256']==a['previous_failed_terminal_sha256']
        assert not json.loads(previous.read_bytes())['success']
    return a,row

def prepare():
    authorize()
    base_path=Path('/dockerdata/wm_models_teacher_base_20261103/base_content.proof.json');base=read(record(base_path))
    assert base['all_three_original_pins_matched'] and len(base['files'])==3
    models=[record(r['local_path']) for r in base['files']]
    assert all(m['sha256']==b['sha256'] for m,b in zip(models,base['files']))
    runtime=read(record(ROOT/'eval/teacher_label_constrained_decode_20261103/A800_runtime_prepared.private.json'))
    source_names=['scripts/run_teacher_a800_exact_resume.py','tests/test_teacher_a800_exact_resume.py',
        'scripts/train_e3_lora.py','scripts/run_teacher_training_smoke.py','scripts/run_teacher_mechanism_probe.py',
        'scripts/prepare_teacher_training_admission.py','scripts/teacher_gpu_exclusive.py']
    result=dict(schema='a800_exact_resume_plan_v1',source={n:record(ROOT/n) for n in source_names},
        runtime_files=runtime['files'],models=models,framework=str(ROOT/'DiffSynth-Studio'),
        backend='native_FA2_explicit_deterministic_true',model_updates=8,logical_updates=4,
        allocated_cards=1,cap_seconds=CAP,upper_GPUh=1.4,automatic_retry_allowed=False,formal_training_allowed=False)
    return save(E/'plan.json',result)

def verify_plan(path,digest):
    row=record(path);assert row['sha256']==digest;p=read(row)
    assert p['cap_seconds']==CAP and p['allocated_cards']==1 and p['formal_training_allowed'] is False
    for r in [*p['source'].values(),*p['runtime_files'],*p['models']]:assert record(r['path'])==r
    return p,row

def cfg():
    return dict(seed=20261104,lr=5e-5,weight_decay=0.01,warmup=1,horizon=4,clip=1.0,
        anchor_lambda=0.5,anchor_weighting='training_weight',component_grad_every=10,log_every=1,
        offload=True,max_steps=4,caption_version='synthetic_FA2_exact_resume',lora_rank=32,lora_alpha=32,
        targets='q,k,v,o,ffn.0,ffn.2',per_rank_batch=1,grad_accum=1)

def worker(a):
    authorize();launch=read(record(E/'launch.json'))
    assert os.getppid()==launch['parent_pid']
    stat=Path('/proc') .joinpath(str(os.getppid()),'stat').read_text().split(') ',1)[1].split()
    assert int(stat[19])==launch['parent_start_ticks']
    assert os.environ['CUDA_VISIBLE_DEVICES']==launch['gpu_uuid'] and os.environ['PYTHONPATH']==SITE
    assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    p,row=verify_plan(a.plan,a.sha)
    sys.path.insert(0,str(ROOT/'scripts'))
    import train_e3_lora as tr
    order,comm=training_interfaces(tr,'cuda:0')
    import torch
    import flash_attn
    import flash_attn.flash_attn_interface as interface
    sys.path.insert(0,p['framework']);sys.path.insert(0,str(ROOT/'scripts'))
    import diffsynth.models.wan_video_dit as wan
    import train_e3_lora as tr
    assert not wan.FLASH_ATTN_3_AVAILABLE and wan.FLASH_ATTN_2_AVAILABLE and wan.flash_attn is flash_attn
    assert torch.cuda.device_count()==1 and torch.cuda.get_device_capability(0)==(8,0)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    original=flash_attn.flash_attn_func;calls=[]
    def deterministic(*args,**kwargs):
        assert wan.flash_attn.flash_attn_func is deterministic and not wan.FLASH_ATTN_3_AVAILABLE
        assert torch.are_deterministic_algorithms_enabled() and kwargs.get('dropout_p',0)==0
        assert kwargs.get('deterministic',True) is True
        calls.append(1);return original(*args,**dict(kwargs,deterministic=True,dropout_p=0.0))
    flash_attn.flash_attn_func=deterministic
    import random
    random.seed(tr.derive_seed('smoke_python',20261104))
    pipe=tr.build_pipe(p['framework'],[r['path'] for r in p['models']],20261104,device='cuda:0')
    assert all(v.device.type=='cuda' for v in pipe.dit.parameters())
    frozen_before=tr.ts.frozen_state_hash(pipe.dit);params=tr.adapter_list(pipe.dit)
    optimizer=torch.optim.AdamW(params,lr=5e-5,weight_decay=.01)
    config=cfg();offset=0;prefix_record=None
    output=E/a.segment;output.mkdir(mode=0o700)
    if a.segment=='resume':
        prefix=read(record(E/'prefix/summary.json'));prefix_record=prefix['checkpoint']
        tr.load_checkpoint(prefix_record['path'],prefix_record,pipe.dit,optimizer,2,cuda=True,expected_config=config)
        offset=2
    elif a.segment not in ('reference','prefix'):raise AssertionError('Unknown segment')
    initial=tr.ts.state_hash(tr.ts.trainable_adapters(pipe.dit))
    records=[]
    for step in range(offset,offset+(4 if a.segment=='reference' else 2)):
        sample_index=order.index(step,0,1);sample=tr.synthetic_sample(sample_index)
        row=tr.train_step(pipe,params,optimizer,sample,step,sample_index,config,comm,
            autocast=lambda:torch.autocast(device_type='cuda',dtype=torch.bfloat16))
        assert row['status']=='ok' and row['grad_norm_total_before_clip']>0
        records.append(row)
    completed=4 if a.segment in ('reference','resume') else 2
    state=tr.checkpoint_state(pipe.dit,optimizer,completed,dict(config,completed_steps=completed),cuda=True)
    tr.validate_checkpoint(state,pipe.dit,optimizer,completed)
    checkpoint=tr.save_checkpoint(output/'checkpoint.pt',state)
    frozen_after=tr.ts.frozen_state_hash(pipe.dit);assert frozen_before==frozen_after
    save(output/'summary.json',dict(segment=a.segment,checkpoint=checkpoint,steps=completed,records=records,
        initial_adapters_sha256=initial,final_adapters_sha256=tr.ts.state_hash(state['adapters']),
        optimizer_sha256=tr.ts.state_hash(state['optimizer']),RNG_sha256=tr.ts.state_hash(state['rng']),
        frozen_base_sha256=frozen_after,attention_calls=len(calls),backend='FA2_deterministic_true',
        prior_prefix=prefix_record,formal_training_allowed=False))
    assert calls
    flash_attn.flash_attn_func=original

def owners(uuid):
    import csv,io
    s=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    return [int(p) for u,p in csv.reader(io.StringIO(s)) if u.strip()==uuid]

def run(a):
    authorize();start=time.monotonic()
    assert Path(sys.argv[0]).is_absolute()
    save(E/'attempt.claim',dict(start_monotonic=start,pid=os.getpid()))
    info=subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=uuid,compute_mode','--format=csv,noheader'],text=True).strip().split(',')
    uuid,mode=info[0].strip(),info[1].strip();child=None;error=None;released=False;restored=False
    lease_dir=Path('/dockerdata/wm_gpu_exclusivity');lease_dir.mkdir(mode=0o700,exist_ok=True);lease=lease_dir/(uuid+'.json')
    assert not lease.exists()
    lease.write_text(json.dumps(dict(gpu_uuid=uuid,expires_epoch=time.time()+CAP,owner_pid=os.getpid(),owner_start=start)));lease.chmod(0o600)
    def alarm(*_):raise TimeoutError('Full exact-resume technical cap reached')
    old={s:signal.signal(s,alarm) for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    signal.setitimer(signal.ITIMER_REAL,CAP-60)
    try:
        p,b=verify_plan(a.plan,a.sha)
        receipt=read(record(E/'publication.remote_bytes.json'))
        assert receipt['remote_equal'] and receipt['public_tests_passed']>0
        manifest_path=Path('/dockerdata/icaspp_ipvg_publication_20261018/research/wm_teacher/PUBLICATION_MANIFEST.json')
        manifest=json.loads(manifest_path.read_bytes());assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==receipt['manifest_sha256']
        for n,r in p['source'].items():assert next(v for v in manifest['files'] if v['path']==n)['source_sha256']==r['sha256']
        for item in manifest['files']:
            data=subprocess.check_output(['git','-C','/dockerdata/icaspp_ipvg_publication_20261018','show',receipt['commit']+':research/wm_teacher/'+item['path']])
            assert hashlib.sha256(data).hexdigest()==item['snapshot_sha256']
        subprocess.run(['nvidia-smi','-i',uuid,'-c','3'],capture_output=True,check=True)
        import teacher_gpu_exclusive as guard
        for pid in owners(uuid):assert guard.signal_holder(pid,4), 'Unknown GPU owner blocks exact admission'
        assert not owners(uuid)
        stat=Path('/proc/self/stat').read_text().split(') ',1)[1].split()
        save(E/'launch.json',dict(parent_pid=os.getpid(),parent_start_ticks=int(stat[19]),gpu_uuid=uuid,plan=b))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,PYTHONPATH=SITE,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1')
        for segment in ('reference','prefix','resume'):
            assert time.monotonic()-start<CAP-120
            with (E/(segment+'.log')).open('xb') as log:
                command=[PYTHON,str(Path(__file__).resolve()),'worker','--segment',segment,'--plan',a.plan,'--sha',a.sha]
                if REVISION002:command.insert(2,'--revision002')
                child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                child.wait(timeout=max(1,CAP-120-(time.monotonic()-start)))
            assert child.returncode==0, 'A segment failed; no automatic retries'
            assert not owners(uuid)
        reference=read(record(E/'reference/summary.json'));resume=read(record(E/'resume/summary.json'))
        assert reference['steps']==resume['steps']==4
        for key in ('final_adapters_sha256','optimizer_sha256','RNG_sha256','frozen_base_sha256'):
            assert reference[key]==resume[key], 'Exact resumed state differs: '+key
        for left,right in zip(reference['records'][2:],resume['records']):
            for key in ('step','lr','grad_norm_total_before_clip','loss_fm_mean','loss_anchor_mean','loss_total_mean','per_rank'):
                assert left[key]==right[key], 'Exact next-update trajectory differs'
        save(E/'comparison.json',dict(exact=True,adapter_optimizer_RNG_exact=True,next_two_update_records_exact=True,
            reference=record(E/'reference/summary.json'),resume=record(E/'resume/summary.json'),
            formal_training_allowed=False,model_quality_claim=False))
    except BaseException as exc:error=type(exc).__name__
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
        queries=[owners(uuid),owners(uuid)];released=not any(queries)
        subprocess.run(['nvidia-smi','-i',uuid,'-c','0' if mode=='Default' else '3'],check=True,capture_output=True)
        restored=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=compute_mode','--format=csv,noheader'],text=True).strip()==mode
        value=json.loads(lease.read_text());assert value['owner_pid']==os.getpid();lease.unlink()
        elapsed=time.monotonic()-start
        save(E/'terminal.json',dict(success=error is None and released and restored and elapsed<=CAP,terminal=True,
            error_type=error,elapsed_seconds=elapsed,allocated_cards=1,charged_upper_GPUh=max(1.4,elapsed/3600),
            cumulative_new_technical_upper_GPUh=PRIOR_TECHNICAL_GPUH+max(1.4,elapsed/3600),released=released,modes_restored=restored,
            formal_training_allowed=False,goal_complete=False))
        for s,h in old.items():signal.signal(s,h)
    assert error is None and released and restored

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--revision002',action='store_true')
    p.add_argument('command',choices=['prepare','run','worker'])
    p.add_argument('--plan');p.add_argument('--sha');p.add_argument('--segment');a=p.parse_args()
    if a.revision002:configure_revision002()
    if a.command=='prepare':print(json.dumps(prepare(),sort_keys=True))
    elif a.command=='worker':worker(a)
    else:run(a)
