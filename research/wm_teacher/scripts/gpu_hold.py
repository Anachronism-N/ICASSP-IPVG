#!/usr/bin/env python3
"""Local GPU reservation: matrix multiplication, yield to external processes, explicit experiment pause.

Run one supervisor per node. Use pause before CUDA initialization when reserving 30 GiB.
  gpu_hold.py --action run --gpus 0,1,2,3,4,5,6,7
  gpu_hold.py --action pause --gpus 0,1
  gpu_hold.py --action resume --gpus 0,1
  gpu_hold.py --action exec --gpus 0 -- python experiment.py
  gpu_hold.py --action stop
"""
import argparse
import contextlib
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

DEFAULT_STATE = '/dockerdata/wm_logs/gpu_hold'


def query_gpus():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True,
                            text=True, timeout=10)
    return {int(i): {'uuid': u.strip(), 'total_mib': int(t), 'used_mib': int(m)}
            for i, u, t, m in csv.reader(io.StringIO(result.stdout))}


def query_processes():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True,
                            text=True, timeout=10)
    processes = {}
    for row in csv.reader(io.StringIO(result.stdout)):
        if len(row) == 2:
            processes.setdefault(row[0].strip(), set()).add(int(row[1]))
    return processes


def write_state(folder, value):
    tmp = folder / 'status.json.tmp'
    tmp.write_text(json.dumps(value, indent=2))
    os.replace(tmp, folder / 'status.json')


def read_state(folder):
    path = folder / 'status.json'
    return json.loads(path.read_text()) if path.exists() else {}


def matching_process(pid, action=None):
    try:
        args = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
        if not any(os.path.basename(x) == b'gpu_hold.py' for x in args):
            return False
        if action is None:
            return True
        wanted = action.encode()
        return (b'--action=' + wanted in args
                or any(flag == b'--action' and value == wanted for flag, value in zip(args, args[1:])))
    except (OSError, ValueError):
        return False


def stop_child(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=60)


def worker(args):
    import torch
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    target = int(args.memory_gib * 1024 ** 3)
    n = 8192
    a = torch.empty((n, n), device='cuda', dtype=torch.bfloat16).fill_(0.01)
    b = torch.empty_like(a).fill_(0.01)
    c = torch.empty_like(a)
    matrix_bytes = a.numel() * a.element_size() * 3
    if target < matrix_bytes:
        raise ValueError('Requested memory is smaller than matrix buffers')
    reserve = torch.empty(target - matrix_bytes, device='cuda', dtype=torch.uint8)
    reserve.zero_()
    torch.cuda.synchronize()
    print(f'holder GPU {args.gpu}: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GiB allocated', flush=True)
    with torch.no_grad():
        while not stop.is_set():
            torch.mm(a, b, out=c)
            torch.cuda.synchronize()
            stop.wait(args.interval)
    del a, b, c, reserve
    torch.cuda.empty_cache()


def supervisor(args, folder, gpus):
    lock = (folder / 'supervisor.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('A GPU holder supervisor already runs on this node')
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    mapping = query_gpus()
    for gpu in gpus:
        if gpu not in mapping or args.memory_gib * 1024 > mapping[gpu]['total_mib'] - 1024:
            raise ValueError(f'GPU {gpu} unavailable or insufficient capacity')
    children, idle_since, retry_at = {}, {}, {}
    try:
        while not stopping.is_set():
            try:
                processes = query_processes()
                mapping = query_gpus()
                query_ok, error = True, None
            except (subprocess.SubprocessError, OSError, ValueError) as exc:
                processes, query_ok, error = {}, False, str(exc)
            own = {p.pid for p in children.values()}
            status = {}
            for gpu in gpus:
                child = children.get(gpu)
                if child is not None and child.poll() is not None:
                    print(f'GPU {gpu} holder exited rc={child.returncode}', flush=True)
                    del children[gpu]
                    retry_at[gpu] = time.monotonic() + 30
                    child = None
                paused = (folder / f'pause.{gpu}').exists()
                external = processes.get(mapping[gpu]['uuid'], set()) - own
                blocked = paused or bool(external) or not query_ok
                if blocked:
                    idle_since.pop(gpu, None)
                    if child is not None:
                        print(f'GPU {gpu}: releasing reservation (paused={paused}, external={sorted(external)})', flush=True)
                        stop_child(child)
                        del children[gpu]
                        child = None
                else:
                    now = time.monotonic()
                    idle_since.setdefault(gpu, now)
                    if child is None and now - idle_since[gpu] >= args.idle_seconds and now >= retry_at.get(gpu, 0):
                        available = mapping[gpu]['total_mib'] - mapping[gpu]['used_mib']
                        if available > args.memory_gib * 1024 + 512:
                            log = (folder / f'worker.{gpu}.log').open('a')
                            env = dict(os.environ, CUDA_VISIBLE_DEVICES=mapping[gpu]['uuid'], OMP_NUM_THREADS='1')
                            command = [sys.executable, str(Path(__file__).resolve()), '--action', 'worker', '--gpu', str(gpu),
                                       '--memory-gib', str(args.memory_gib), '--interval', str(args.interval)]
                            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                            log.close()
                            children[gpu] = child
                            print(f'GPU {gpu}: holder started PID {child.pid}', flush=True)
                status[str(gpu)] = {'paused': paused, 'external_pids': sorted(external),
                                    'holder_pid': child.pid if child else None,
                                    'gpu_uuid': mapping[gpu]['uuid'], 'memory_gib_target': args.memory_gib}
            write_state(folder, {'pid': os.getpid(), 'running': True, 'updated_at': time.time(), 'query_ok': query_ok,
                                 'error': error, 'gpus': status})
            stopping.wait(1)
    finally:
        for child in list(children.values()):
            stop_child(child)
        write_state(folder, {'pid': os.getpid(), 'running': False, 'updated_at': time.time(), 'gpus': {}})
        lock.close()


def pause(folder, gpus):
    for gpu in gpus:
        (folder / f'pause.{gpu}').touch(mode=0o600)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        state = read_state(folder)
        if not state.get('running') or not matching_process(state.get('pid', 0), 'run'):
            print('Pause flags saved; supervisor is not running', flush=True)
            return
        selected = [state.get('gpus', {}).get(str(g)) for g in gpus]
        if all(s and s['paused'] and s['holder_pid'] is None for s in selected):
            print(f'Reservation released on GPUs {gpus}; external workloads were not stopped', flush=True)
            return
        time.sleep(0.2)
    raise RuntimeError('Pause not acknowledged; do not initialize an experiment on these GPUs yet')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=['run', 'worker', 'pause', 'resume', 'stop', 'status', 'exec'], required=True)
    parser.add_argument('--gpus')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--state-dir', default=DEFAULT_STATE)
    parser.add_argument('--memory-gib', type=float, default=30)
    parser.add_argument('--interval', type=float, default=0.0)
    parser.add_argument('--idle-seconds', type=float, default=5)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.memory_gib <= 0 or args.interval < 0 or args.idle_seconds < 0:
        parser.error('Invalid memory, interval or idle settings')
    if args.action == 'worker':
        worker(args)
        return
    folder = Path(args.state_dir)
    os.umask(0o077)
    folder.mkdir(mode=0o700, exist_ok=True)
    gpus = list(dict.fromkeys(int(g) for g in args.gpus.split(','))) if args.gpus else []
    if args.action in ('run', 'pause', 'resume', 'exec') and not gpus:
        parser.error('--gpus is required')
    if args.action == 'run':
        supervisor(args, folder, gpus)
    elif args.action == 'status':
        print(json.dumps(read_state(folder), indent=2))
    elif args.action == 'pause':
        pause(folder, gpus)
    elif args.action == 'resume':
        for gpu in gpus:
            (folder / f'pause.{gpu}').unlink(missing_ok=True)
        print(f'Reservation may resume when GPUs {gpus} are idle')
    elif args.action == 'stop':
        state = read_state(folder)
        pid = state.get('pid', 0)
        if state.get('running') and matching_process(pid, 'run'):
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if not matching_process(pid, 'run'):
                    print('Supervisor and holder workers stopped')
                    return
                time.sleep(0.2)
            raise RuntimeError('Supervisor teardown still pending')
        print('No matching supervisor is running')
    elif args.action == 'exec':
        command = args.command[1:] if args.command[:1] == ['--'] else args.command
        if not command:
            parser.error('An experiment command is required after --')
        # Serialize overlapping wrapped experiments so one cannot clear another's pause.
        with contextlib.ExitStack() as stack:
            for gpu in sorted(gpus):
                lock = stack.enter_context((folder / f'experiment.{gpu}.lock').open('a'))
                fcntl.flock(lock, fcntl.LOCK_EX)
            already_paused = {g for g in gpus if (folder / f'pause.{g}').exists()}
            pause(folder, gpus)
            child = None
            def interrupted(*_):
                raise KeyboardInterrupt
            old_handler = signal.signal(signal.SIGTERM, interrupted)
            try:
                child = subprocess.Popen(command)
                rc = child.wait()
            except KeyboardInterrupt:
                rc = 130
            finally:
                if child is not None and child.poll() is None:
                    stop_child(child)
                for gpu in gpus:
                    if gpu not in already_paused:
                        (folder / f'pause.{gpu}').unlink(missing_ok=True)
                signal.signal(signal.SIGTERM, old_handler)
            raise SystemExit(rc)


if __name__ == '__main__':
    main()
