#!/usr/bin/env python3
"""Finite new-cluster resource guard; only verified holders are signaled, modes restored."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time

HOLDER='/srv/PRIVATE_STORAGE'
HOLDER_PYTHON='/srv/PRIVATE_STORAGE'

def query_modes():
    r=subprocess.run(['nvidia-smi','--query-gpu=uuid,index,compute_mode','--format=csv,noheader,nounits'],
        capture_output=True,text=True,check=True,timeout=10)
    return {u.strip():dict(index=int(i),mode=m.strip()) for u,i,m in csv.reader(io.StringIO(r.stdout))}

def apps():
    r=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],
        capture_output=True,text=True,check=True,timeout=10)
    result={}
    for u,p in csv.reader(io.StringIO(r.stdout)):result.setdefault(u.strip(),set()).add(int(p))
    return result

def holder_identity(pid,gpu):
    p=Path('/proc')/str(pid)
    args=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
    if args!=[HOLDER_PYTHON,HOLDER]:return None
    env=dict(s.split(b'=',1) for s in (p/'environ').read_bytes().split(bytes([0])) if b'=' in s)
    placement=env.get(b'CUDA_VISIBLE_DEVICES',env.get(b'OCCUPY_GPU'))
    if placement!=str(gpu).encode() or p.stat().st_uid!=os.getuid():return None
    stat=(p/'stat').read_text().split(') ',1)[1].split()
    return dict(pid=pid,start_ticks=int(stat[19]),argv=args,gpu=gpu)

def signal_holder(pid,gpu):
    fd=os.pidfd_open(pid)
    try:
        identity=holder_identity(pid,gpu)
        if identity is None:return None
        signal.pidfd_send_signal(fd,signal.SIGTERM)
        if not select.select([fd],[],[],20)[0]:raise RuntimeError('Known holder did not exit; no global kill')
        return identity
    finally:os.close(fd)

def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,sort_keys=True,indent=2)+chr(10))
    tmp.chmod(0o600);os.replace(tmp,path)

def run(a):
    import hashlib
    authorization=json.loads(Path(a.authorization).read_bytes())
    assert hashlib.sha256(Path(a.authorization).read_bytes()).hexdigest()==a.auth_sha
    assert authorization['schema']=='teacher_extended_stage_authorization_v1' and authorization['new_campaign_GPUh_cap']==168
    uuids=a.uuids.split(',');assert len(uuids)==len(set(uuids))==4
    out=Path(a.state);out.mkdir(mode=0o700)
    (out/'once.claim').open('x').close()
    mapping=query_modes();initial={u:mapping[u] for u in uuids};assert all(initial[u]['index'] in range(4) for u in uuids)
    assert all(x['mode'] in ('Default','Exclusive_Process') for x in initial.values())
    signals=[];start=time.monotonic();stop=False;error=None;restored=False
    def interrupted(*_):
        nonlocal stop
        stop=True
    for s in (signal.SIGTERM,signal.SIGINT):signal.signal(s,interrupted)
    try:
        for u in uuids:
            subprocess.run(['nvidia-smi','-i',u,'-c','3'],check=True,capture_output=True,timeout=10)
        assert all(query_modes()[u]['mode']=='Exclusive_Process' for u in uuids)
        while not stop and time.monotonic()-start<5400:
            current=apps()
            for u in uuids:
                for pid in current.get(u,set()):
                    try:
                        row=signal_holder(pid,initial[u]['index'])
                        if row:signals.append(row)
                    except ProcessLookupError:pass
            save(out/'status.json',dict(running=True,start_monotonic=start,observed_monotonic=time.monotonic(),
                initial=initial,signals=signals,exclusive_process_verified=True))
            if Path(a.stop_file).exists():break
            time.sleep(1)
    except BaseException as exc:error=type(exc).__name__
    finally:
        for u,row in initial.items():
            mode='0' if row['mode']=='Default' else '3'
            subprocess.run(['nvidia-smi','-i',u,'-c',mode],check=True,capture_output=True,timeout=10)
        restored=all(query_modes()[u]['mode']==initial[u]['mode'] for u in uuids)
        save(out/'terminal.json',dict(running=False,initial=initial,signals=signals,error_type=error,
            elapsed_seconds=time.monotonic()-start,modes_restored=restored,unknown_processes_signaled=False,
            source_files_modified=False,old_cluster_touched=False))
    assert restored and error is None

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--uuids',required=True)
    p.add_argument('--authorization',required=True);p.add_argument('--auth-sha',required=True)
    p.add_argument('--state',required=True);p.add_argument('--stop-file',required=True)
    run(p.parse_args())
