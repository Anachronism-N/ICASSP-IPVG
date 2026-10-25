"""One explicitly budgeted recovery attempt; preserves original node receipts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import run_e3_subset_node as n


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--node-rank',type=int,choices=(1,2),required=True)
    ap.add_argument('--approved-plan-sha256',required=True)
    a=ap.parse_args()
    plan_path=n.ROOT/'eval/e3_subset_20261025/plan.json'
    plan,sources,cache=n.verify_inputs(plan_path,a.approved_plan_sha256)
    peers=[json.loads(n.node_path(plan_path,i,'_preflight.json').read_bytes()) for i in range(4)]
    assert len({p['model_content_sha256'] for p in peers})==1
    assert all(p['plan_sha256']==a.approved_plan_sha256 and p['checkpoint_cache']['sha256']==cache['sha256'] for p in peers)
    assert peers[a.node_rank]['hostname']==os.uname().nodename, 'wrong recovery node'
    assert peers[a.node_rank]['checkpoint_cache']==cache, 'cache changed after preflight'
    for asset in peers[a.node_rank]['models']:
        stat=Path(asset['path']).stat()
        assert (stat.st_size,stat.st_mtime_ns)==(asset['bytes'],asset['mtime_ns'])
    previous=[json.loads(n.node_path(plan_path,i,'.finished.json').read_bytes()) for i in (1,2)]
    assert all(p['exit_code']!=0 and p['owned_tree_released'] and not p['cap_exceeded'] for p in previous)
    previous_hours=sum(p['gpu_hours'] for p in previous)
    reserved=4*8*3000/3600
    assert previous_hours+reserved<=28,'recovery exceeds original generation cap'
    folder=plan_path.parent
    launch=folder/f'node_{a.node_rank}_recovery_a1.launch.json'
    finish=folder/f'node_{a.node_rank}_recovery_a1.finished.json'
    assert not launch.exists() and not finish.exists(),'one recovery only; preserve receipts'
    command,env=n.command_for(plan,a.node_rank)
    env['OMP_NUM_THREADS']='4';env['MKL_NUM_THREADS']='4'
    n.exclusive(launch,{'rank':a.node_rank,'run_id':'recovery_a1','plan_sha256':a.approved_plan_sha256,
                      'prior_failed_gpu_hours':previous_hours,'aggregate_reserved_gpu_hours':reserved+previous_hours,
                      'max_seconds':3000,'command':command,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    flags=[];child=None;error=None;interrupted=None;started=time.monotonic();start=time.time()
    def interrupt(signum,frame):
        nonlocal interrupted
        interrupted=signum
    for signum in (signal.SIGTERM,signal.SIGINT):signal.signal(signum,interrupt)
    try:
        for gpu in range(8):
            flag=n.STATE/f'pause.{gpu}'
            try:fd=os.open(flag,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            except FileExistsError:continue
            os.close(fd);flags.append(flag)
        child=subprocess.Popen(command,env=env,start_new_session=True)
        while child.poll() is None and not interrupted and time.monotonic()-started<2940:
            time.sleep(.2)
    except BaseException as exc:error=type(exc).__name__+': '+str(exc)
    finally:
        released=child is None
        if child is not None:
            try:
                n.stop_group(child,started+3000)
                released=not n.group_alive(child.pid)
            except BaseException as exc:
                error=type(exc).__name__+': '+str(exc)
                released=False
        elapsed=time.monotonic()-started
        if released:
            for flag in flags:flag.unlink(missing_ok=True)
        receipt={'rank':a.node_rank,'run_id':'recovery_a1','start':start,'end':time.time(),
                 'elapsed_seconds':elapsed,'gpu_hours':8*elapsed/3600,'exit_code':child.returncode if child else None,
                 'error':error,'interrupted_signal':interrupted,'owned_tree_released':released,'cap_exceeded':elapsed>3000,
                 'complete_not_proven':True,'plan_sha256':a.approved_plan_sha256}
        n.exclusive(finish,receipt);print(json.dumps(receipt),flush=True)
    raise SystemExit(0 if child and child.returncode==0 and released and not error and elapsed<=3000 else 1)

if __name__=='__main__':main()
