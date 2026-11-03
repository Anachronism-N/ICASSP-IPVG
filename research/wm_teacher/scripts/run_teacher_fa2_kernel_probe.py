#!/usr/bin/env python3
"""One finite native FA2 CUDA admission probe; not model resume, training or quality."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'eval/teacher_progress_campaign_20261104/technical_kernel_probe001'
AUTH_SHA='e7b595010a29b903b23a6b8e326777ea2d0f899338ef342b7bde92f645a93905'
CAMPAIGN_SHA='ffa19129fb3256e482553bb1b90a7f09edd95eed4391687e6b45426d0706b8fe'
PYTHON='/dockerdata/venv_qwen32b_20261103/bin/python3.12'
SITE='/dockerdata/wm_a800_fa2_runtime_20261103/site'
CAP=720
REVISION2=False
REVISION3=False

def bound(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def write(p,v):
    b=(json.dumps(v,sort_keys=True,allow_nan=False)+chr(10)).encode()
    with Path(p).open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    Path(p).chmod(0o600);return bound(p)

def authorize():
    row=bound(E/'authorization.json');assert row['sha256']==AUTH_SHA
    a=json.loads(Path(row['path']).read_bytes());c=bound(E.parent/'campaign.authorization.json')
    assert c['sha256']==CAMPAIGN_SHA==a['campaign_authorization_sha256']
    assert a['phase']=='technical_admission' and a['phase_cap_GPUh']==8 and a['wall_cap_seconds']==CAP
    assert a['GPU_index']==4 and a['GPUh_upper_cap']==0.2 and a['automatic_retry_allowed'] is False
    return a,row

def device():
    r=subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=uuid,compute_mode','--format=csv,noheader,nounits'],text=True)
    u,m=next(csv.reader(io.StringIO(r)));return u.strip(),m.strip()

def owners(uuid):
    r=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    return [int(p) for u,p in csv.reader(io.StringIO(r)) if u.strip()==uuid]

def native_id():
    r=json.loads(Path('/dockerdata/wm_a800_fa2_runtime_20261103/runtime_cpu_identity.json').read_bytes())
    return next(f for f in r['selected_files'] if Path(f['path']).name.startswith('flash_attn_2_cuda.'))

def worker():
    authorize();launch=json.loads((E/'launch.json').read_bytes())
    assert os.getppid()==launch['child_parent_pid']
    parent_stat=Path('/proc').joinpath(str(os.getppid()),'stat').read_text().split(') ',1)[1].split()
    assert int(parent_stat[19])==launch['parent_start_ticks']
    assert os.environ['CUDA_VISIBLE_DEVICES']==launch['gpu_uuid'] and os.environ['PYTHONPATH']==SITE
    assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    assert bound(Path(__file__).resolve())==launch['source']
    assert bound(ROOT/'scripts/run_teacher_qwen3vl32b_recovery.py')==launch['decoder_source']
    assert bound(E/'runtime.native.json')==launch['native_record']
    frozen=json.loads((E/'runtime.native.json').read_bytes());assert bound(frozen['path'])==frozen
    import torch
    import flash_attn
    import flash_attn_2_cuda
    assert str(Path(flash_attn_2_cuda.__file__).resolve())==frozen['path']
    assert torch.cuda.device_count()==1 and torch.cuda.get_device_capability(0)==(8,0)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    from transformers import NoBadWordsLogitsProcessor
    sys.path.insert(0,str(ROOT/'scripts'))
    from run_teacher_qwen3vl32b_recovery import vectorized_bad_words
    bad=[[2],[3,4],[5,4],[7,8,9]]
    vector=vectorized_bad_words(bad,[11]);native=NoBadWordsLogitsProcessor(bad,eos_token_id=[11])
    mask_checks=0
    for length in (1,3,8):
        ids=torch.randint(0,12,(2,length),device='cpu');scores=torch.randn(2,20,device='cuda')
        left=vector(ids,scores.clone());right=native(ids.to(scores.device),scores.clone())
        assert torch.equal(left,right);mask_checks+=1
    records=[];torch.cuda.reset_peak_memory_stats()
    for length in (128,1024):
        generator=torch.Generator(device='cuda').manual_seed(20261104+length)
        base=[torch.randn((1,length,24,128),device='cuda',dtype=torch.bfloat16,generator=generator) for _ in range(3)]
        outputs=[];gradients=[];began=time.monotonic()
        for repeat in range(2):
            q,k,v=[t.detach().clone().requires_grad_(True) for t in base]
            result=flash_attn.flash_attn_func(q,k,v,dropout_p=0.0,causal=False,deterministic=True)
            result.float().square().mean().backward();torch.cuda.synchronize()
            assert all(torch.isfinite(g.grad).all() for g in (q,k,v)) and torch.isfinite(result).all()
            outputs.append(result.detach().cpu());gradients.append([t.grad.detach().cpu() for t in (q,k,v)])
        forward=torch.equal(outputs[0],outputs[1]);backward=all(torch.equal(a,b) for a,b in zip(*gradients))
        records.append(dict(sequence_length=length,forward_exact=forward,backward_exact=backward,seconds=time.monotonic()-began,
            dtype='bfloat16',head_dimension=128,heads=24))
        assert forward and backward
    write(E/'worker.json',dict(success=True,records=records,native=frozen,forward_calls=4,backward_calls=4,
        CUDA_vectorized_mask_equivalence_checks=mask_checks,CPU_input_GPU_scores_alignment_verified=True,
        deterministic=True,dropout_p=0.0,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        model_loaded=False,model_updates=0,model_exact_resume_verified=False,formal_training_allowed=False))

def run():
    a,row=authorize();start=time.monotonic();uuid,mode=device();assert mode in ('Default','Exclusive_Process')
    write(E/'attempt.claim',dict(start_monotonic=start,pid=os.getpid(),auth=row,uuid=uuid))
    child=None;released=False;restored=False;error=None;signals=[];source=bound(Path(__file__).resolve())
    def alarm(*_):raise TimeoutError('Native FA2 technical allocation cap reached')
    old={s:signal.signal(s,alarm) for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    signal.setitimer(signal.ITIMER_REAL,CAP-30)
    try:
        # The caller must provide a fresh independently verified publication receipt.
        receipt=json.loads((E/'publication.remote_bytes.json').read_bytes())
        assert receipt['remote_equal'] is True and receipt['public_tests_passed']>0
        manifest=json.loads(Path('/dockerdata/icaspp_ipvg_publication_20261018/research/wm_teacher/PUBLICATION_MANIFEST.json').read_bytes())
        published=next(r for r in manifest['files'] if r['path']=='scripts/run_teacher_fa2_kernel_probe.py')
        assert published['source_sha256']==source['sha256']
        assert receipt['manifest_sha256']==hashlib.sha256(Path('/dockerdata/icaspp_ipvg_publication_20261018/research/wm_teacher/PUBLICATION_MANIFEST.json').read_bytes()).hexdigest()
        G='/dockerdata/icaspp_ipvg_publication_20261018';prefix='research/wm_teacher/'
        assert receipt['manifest_files_verified']==len(manifest['files'])
        for file in manifest['files']:
            content=subprocess.check_output(['git','-C',G,'show',receipt['commit']+':'+prefix+file['path']])
            assert hashlib.sha256(content).hexdigest()==file['snapshot_sha256']
        remote=subprocess.check_output(['git','-C',G,'ls-remote','origin','refs/heads/wm-teacher-experiments-20261018'],text=True,timeout=30).split()
        assert remote[0]==receipt['commit']
        lease_dir=Path('/dockerdata/wm_gpu_exclusivity');lease_dir.mkdir(mode=0o700,exist_ok=True)
        lease=lease_dir/(uuid+'.json');assert not lease.exists(), 'Unknown existing lease blocks admission'
        lease.write_text(json.dumps(dict(gpu_uuid=uuid,expires_epoch=time.time()+CAP,owner_pid=os.getpid(),owner_start=start)))
        lease.chmod(0o600)
        subprocess.run(['nvidia-smi','-i',uuid,'-c','3'],check=True,capture_output=True)
        for pid in owners(uuid):
            fd=os.pidfd_open(pid)
            try:
                p=Path('/proc')/str(pid);argv=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
                allowed=[['/opt/venv_eval/bin/python','-u','/root/planner_two_node_tasks_20261029_v4/code/hold_gpu.py'],
                    ['/srv/PRIVATE_STORAGE',
                     '/srv/PRIVATE_STORAGE']]
                assert argv in allowed, 'Unknown owner blocks admission'
                assert p.stat().st_uid==os.getuid()
                signal.pidfd_send_signal(fd,signal.SIGTERM);assert select.select([fd],[],[],20)[0]
                signals.append(pid)
            finally:os.close(fd)
        assert not owners(uuid)
        native=write(E/'runtime.native.json',native_id())
        parent_stat=Path('/proc/self/stat').read_text().split(') ',1)[1].split()
        write(E/'launch.json',dict(gpu_uuid=uuid,child_parent_pid=os.getpid(),parent_start_ticks=int(parent_stat[19]),source=source,
            decoder_source=bound(ROOT/'scripts/run_teacher_qwen3vl32b_recovery.py'),native_record=native))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,PYTHONPATH=SITE,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1')
        with (E/'worker.log').open('xb') as log:
            command=[PYTHON,str(Path(__file__).resolve())]+(['--revision3'] if REVISION3 else ['--revision2'] if REVISION2 else [])+['worker']
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            child.wait(timeout=max(1,CAP-50-(time.monotonic()-start)))
        assert child.returncode==0 and json.loads((E/'worker.json').read_bytes())['success']
    except BaseException as exc:error=type(exc).__name__
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
        queries=[owners(uuid),owners(uuid)];released=not any(queries)
        subprocess.run(['nvidia-smi','-i',uuid,'-c','0' if mode=='Default' else '3'],check=True,capture_output=True)
        restored=device()[1]==mode;elapsed=time.monotonic()-start
        lease_file=Path('/dockerdata/wm_gpu_exclusivity')/(uuid+'.json')
        if lease_file.is_file():
            lease_value=json.loads(lease_file.read_text())
            if lease_value.get('owner_pid')==os.getpid() and lease_value.get('owner_start')==start:lease_file.unlink()
        write(E/'terminal.json',dict(terminal=True,success=error is None and released and restored and elapsed<=CAP,
            phase='technical_admission',elapsed_seconds=elapsed,allocated_cards=1,
            charged_upper_GPUh=max(0.2,elapsed/3600),actual_allocated_GPUh=elapsed/3600,
            cumulative_technical_upper_GPUh=a['prior_phase_used_GPUh']+max(0.2,elapsed/3600),
            released=released,modes_restored=restored,holder_pids_signaled=signals,error_type=error,
            model_exact_resume_verified=False,formal_training_allowed=False,goal_complete=False))
        for s,h in old.items():signal.signal(s,h)
    assert error is None and released and restored

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--revision2',action='store_true');p.add_argument('--revision3',action='store_true')
    p.add_argument('command',choices=['run','worker'])
    a=p.parse_args()
    if a.revision2:
        REVISION2=True
        E=ROOT/'eval/teacher_progress_campaign_20261104/technical_kernel_probe002'
        AUTH_SHA='b3a0675dd046beeac7a9ab828381a5ab720a2d4a7fa9aa92ca7f6db8a0d30bfc'
    if a.revision3:
        REVISION3=True
        E=ROOT/'eval/teacher_progress_campaign_20261104/technical_kernel_probe003'
        AUTH_SHA='d83036ae1ec59eda9e6b3031c637c019f90e6659417723e45ba4e9bba3caf3b8'
    if a.command=='worker':worker()
    else:run()
