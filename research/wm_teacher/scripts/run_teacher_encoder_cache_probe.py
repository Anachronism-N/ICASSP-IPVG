#!/usr/bin/env python3
"""Finite real encoder/cache API probe with synthetic input; never supervised training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'eval/teacher_progress_campaign_20261104/encoder_cache_probe001'
AUTH_SHA='40cfc1c9effc89d64e6ead8e2c1a5644df04c7f5f6169638552a56a14f81cd97'
PYTHON='/dockerdata/venv_qwen32b_20261103/bin/python3.12'
SITE='/dockerdata/wm_a800_fa2_runtime_20261103/site'
CAP=720


def helpers():
    sys.path.insert(0,str(ROOT/'scripts'))
    import run_teacher_qwen3vl32b_recovery as r
    import teacher_gpu_exclusive as g
    return r,g


def record(path):return helpers()[0].record(path)
def read(row):return helpers()[0].read(row)
def save(path,value):return helpers()[0].write_new(path,value)


def authorize():
    row=record(E/'authorization.json');assert row['sha256']==AUTH_SHA;a=read(row)
    assert record(E.parent/'campaign.authorization.json')['sha256']==a['campaign_authorization_sha256']
    assert a['phase']=='technical_admission' and a['prior_phase_GPUh']==4.8 and a['phase_cap_GPUh']==8
    assert a['allocated_cards']==1 and a['GPU_index']==4 and a['wall_cap_seconds']==CAP and a['upper_GPUh']==0.2
    assert a['prior_phase_GPUh']+a['upper_GPUh']<=a['phase_cap_GPUh']
    assert a['synthetic_source_frames']==121 and a['synthetic_size']==[1280,704]
    assert a['automatic_retry_allowed'] is False and a['formal_training_allowed'] is False
    assert a['actual_supervised_data_used'] is False
    return a,row


def prepare(output=None):
    r,g=helpers();a,auth=authorize()
    audit=record(E.parent/'encoder_originalpins.independent_audit.json');proof=read(audit)
    assert audit['sha256']==a['encoder_original_pin_audit_sha256'] and proof['success']
    assert proof['all_six_original_pins_independently_rehashed'] and len(proof['files'])==6
    for f in proof['files']:r.check({k:f[k] for k in ('path','bytes','sha256')})
    import cache_e3_inputs as cache
    assert cache.X0_SHAPE==(1,48,31,44,80) and cache.FIRST_SHAPE==(1,48,1,44,80) and cache.CTX_SHAPE==(1,512,4096)
    modes=g.query_modes();uuid=next(u for u,v in modes.items() if v['index']==4)
    source_names=('scripts/run_teacher_encoder_cache_probe.py','tests/test_teacher_encoder_cache_probe.py',
                  'scripts/cache_e3_inputs.py','scripts/run_teacher_mechanism_probe.py','scripts/teacher_gpu_exclusive.py')
    plan=dict(schema='teacher_encoder_cache_technical_plan_v1',authorization=auth,encoder_audit=audit,
        encoders=proof['files'],hostname=os.uname().nodename,gpu_uuid=uuid,cap_seconds=CAP,
        source={n:record(ROOT/n) for n in source_names},runtime=r.runtime(),
        framework=record(ROOT/'DiffSynth-Studio/diffsynth/pipelines/wan_video.py'),
        resource_control_runtime=g.control_runtime(),formal_training_allowed=False,
        actual_supervised_data_used=False,automatic_retry_allowed=False)
    return save(Path(output) if output else E/'plan.json',plan)


def verify(path,digest):
    r,g=helpers();a,auth=authorize();bound=record(path);assert bound['sha256']==digest;p=read(bound)
    assert p['authorization']==auth and p['hostname']==os.uname().nodename and p['cap_seconds']==CAP
    assert p['resource_control_runtime']==g.control_runtime() and p['runtime']==r.runtime()
    assert p['formal_training_allowed'] is False and p['actual_supervised_data_used'] is False and p['automatic_retry_allowed'] is False
    assert p['encoder_audit']['sha256']==a['encoder_original_pin_audit_sha256']
    for row in [p['encoder_audit'],p['framework'],*p['source'].values(),*p['encoders']]:
        r.check({k:row[k] for k in ('path','bytes','sha256')})
    return p,bound


def worker(args):
    r,g=helpers();p,b=verify(args.plan,args.sha);launch=read(record(E/'launch.json'))
    parent=r.helpers()[1].identity(os.getppid());assert r.same_identity(parent,launch['parent']) and launch['plan']==b
    assert os.environ['CUDA_VISIBLE_DEVICES']==p['gpu_uuid'] and os.environ['PYTHONPATH']==SITE
    assert os.environ['HF_HUB_OFFLINE']==os.environ['TRANSFORMERS_OFFLINE']=='1'
    import torch
    from PIL import Image
    import cache_e3_inputs as cache
    assert torch.cuda.device_count()==1 and torch.cuda.get_device_capability(0)==(8,0)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    models=Path(p['encoders'][0]['path'])
    for _ in Path(p['encoders'][0]['relative']).parts:models=models.parent
    began=time.monotonic();pipe=cache.build_encoders(models,ROOT/'DiffSynth-Studio',hash_models=True)
    assert pipe.dit is None and pipe.vae is not None and pipe.text_encoder is not None
    frames=[Image.new('RGB',(1280,704),(80,100,120)) for _ in range(121)]
    x0,first=cache.encode_video(pipe,frames)
    prompt='A red circular disk rests on a plain white table without movement.'
    context=cache.encode_prompt(pipe,prompt);again=cache.encode_prompt(pipe,prompt)
    assert torch.equal(context,again)
    for value,shape in ((x0,cache.X0_SHAPE),(first,cache.FIRST_SHAPE),(context,cache.CTX_SHAPE)):
        assert value.device.type=='cpu' and value.dtype==torch.bfloat16 and tuple(value.shape)==shape and bool(torch.isfinite(value).all())
    cache_file=cache.m1.save_tensors(E/'synthetic_encoder_outputs.pt',dict(x0=x0,first_frame_latents=first,context=context))
    torch.cuda.synchronize()
    save(E/'worker.json',dict(success=True,plan=b,synthetic_source_frames=121,
        x0_shape=list(x0.shape),first_shape=list(first.shape),context_shape=list(context.shape),
        repeated_T5_context_exact=True,outputs_finite=True,DiT_loaded=False,model_updates=0,
        actual_supervised_data_used=False,supervised_cache_admitted=False,formal_training_allowed=False,
        outputs=cache_file,compute_seconds=time.monotonic()-began,peak_allocated_bytes=torch.cuda.max_memory_allocated()))


def run(args):
    r,g=helpers();assert args.gpu_authorized and Path(sys.argv[0]).is_absolute()
    started=time.monotonic();p,b=verify(args.plan,args.sha)
    receipt=read(record(E/'publication.remote_bytes.json'))
    import run_teacher_label_semantic_review as review
    review.publication(p,receipt['commit'],str(E/'publication.remote_bytes.json'),time.monotonic()+120)
    initial=g.query_modes()[p['gpu_uuid']];assert initial['index']==4 and initial['mode'] in ('Default','Exclusive_Process')
    lease=Path('/dockerdata/wm_gpu_exclusivity')/(p['gpu_uuid']+'.json');assert not lease.exists()
    assert all(g.holder_identity(pid,4) is not None for pid in g.apps().get(p['gpu_uuid'],set()))
    assert time.monotonic()-started<CAP-120
    save(E/'attempt.claim',dict(started_monotonic=started,plan=b));created=False;child=None;error=None;signals=[]
    def timeout(*_):raise TimeoutError('Encoder technical allocation cap reached')
    old={s:signal.signal(s,timeout) for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    signal.setitimer(signal.ITIMER_REAL,max(.01,CAP-60-(time.monotonic()-started)))
    try:
        with lease.open('x') as f:json.dump(dict(gpu_uuid=p['gpu_uuid'],expires_epoch=time.time()+CAP,owner_pid=os.getpid()),f)
        lease.chmod(0o600);created=True
        subprocess.run(['nvidia-smi','-i',p['gpu_uuid'],'-c','3'],check=True,capture_output=True,timeout=10)
        for pid in g.apps().get(p['gpu_uuid'],set()):
            identity=g.signal_holder(pid,4);assert identity is not None;signals.append(identity)
        assert not g.apps().get(p['gpu_uuid'])
        save(E/'launch.json',dict(parent=r.helpers()[1].identity(os.getpid()),plan=b))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=p['gpu_uuid'],PYTHONPATH=SITE,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1')
        with (E/'worker.log').open('xb') as log:
            child=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'worker','--plan',args.plan,'--sha',args.sha],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            child.wait(timeout=max(1,CAP-120-(time.monotonic()-started)))
        assert child.returncode==0 and read(record(E/'worker.json'))['success']
    except BaseException as exc:error=type(exc).__name__
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
        queries=[g.apps(),g.apps()];released=all(not q.get(p['gpu_uuid']) for q in queries)
        subprocess.run(['nvidia-smi','-i',p['gpu_uuid'],'-c','0' if initial['mode']=='Default' else '3'],check=True,capture_output=True,timeout=10)
        restored=g.query_modes()[p['gpu_uuid']]['mode']==initial['mode']
        if created:
            assert read(record(lease))['owner_pid']==os.getpid();lease.unlink()
        elapsed=time.monotonic()-started
        terminal=dict(success=error is None and released and restored and elapsed<=CAP,terminal=True,error_type=error,
            elapsed_seconds=elapsed,actual_GPUh=elapsed/3600,charged_upper_GPUh=max(.2,elapsed/3600),
            cumulative_new_technical_upper_GPUh=4.8+max(.2,elapsed/3600),released=released,modes_restored=restored,
            holder_signals=signals,model_updates=0,actual_supervised_data_used=False,supervised_cache_admitted=False,
            formal_training_allowed=False,sota_proven=False)
        save(E/'terminal.json',terminal)
        for s,h in old.items():signal.signal(s,h)
    assert terminal['success']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','verify','run','worker'])
    parser.add_argument('--output');parser.add_argument('--plan');parser.add_argument('--sha');parser.add_argument('--gpu-authorized',action='store_true');a=parser.parse_args()
    if a.command=='prepare':print(json.dumps(prepare(a.output),sort_keys=True))
    elif a.command=='verify':verify(a.plan,a.sha)
    elif a.command=='worker':worker(a)
    else:run(a)
