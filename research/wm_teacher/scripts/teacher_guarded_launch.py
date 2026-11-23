#!/usr/bin/env python3
"""Generic finite GPU launcher for already-frozen Teacher commands (smoke, cache, training, generation).

Guards only: node-local UUID leases (known holders wait), PIDFD TERM of exactly recognised holders,
optional EXCLUSIVE_PROCESS, foreign-PID intrusion stop, hard wall cap, release/mode restore, and a
terminal cost record charged as world x wall. It grants no data, semantic or training approval.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'eval/teacher_progress_campaign_20261104'
LEASE_DIR = Path('/dockerdata/wm_gpu_exclusivity')
PHASES = {'technical_admission': ('current_technical_reserved_upper_GPUh', 'technical_admission'),
          'label_and_cache': ('current_label_reserved_upper_GPUh', 'label_and_cache'),
          'gate_conditioned_teacher_training_and_evaluation': ('current_training_eval_reserved_upper_GPUh',
                                                               'gate_conditioned_teacher_training_and_evaluation'),
          'official_protocol_evaluation': ('current_official_eval_reserved_upper_GPUh', 'official_protocol_evaluation'),
          'training_optimization': ('current_training_opt_reserved_upper_GPUh', 'training_optimization')}
BUDGET = CAMPAIGN / 'budget.live.json'


def update_budget(fn):
    """Locked read-modify-write of the shared ledger (several nodes may finish concurrently)."""
    import fcntl
    with open(str(BUDGET) + '.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        b = json.loads(BUDGET.read_text())
        out = fn(b)
        tmp = BUDGET.with_suffix('.json.tmp.%d' % os.getpid())
        tmp.write_text(json.dumps(b, indent=2, sort_keys=True) + '\n')
        os.replace(tmp, BUDGET)
        return out


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha_record(path):
    data = Path(path).read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def save_new(path, value):
    with open(path, 'x') as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write('\n')
    Path(path).chmod(0o600)
    return sha_record(path)


def guard():
    sys.path.insert(0, str(ROOT / 'scripts'))
    import teacher_gpu_exclusive as g
    return g


def ledger_prior(budget, phase):
    key, _ = PHASES[phase]
    return float(budget.get(key, 0.0))


def prepare(a):
    g = guard()
    stage = Path(a.stage_dir)
    require(stage.is_absolute() and stage.parent == CAMPAIGN and not stage.exists(), 'New campaign stage dir required')
    spec = json.loads(Path(a.spec).read_text())
    require(type(spec['command']) is list and spec['command'] and all(type(x) is str for x in spec['command']), 'argv list required')
    require(spec['phase'] in PHASES, 'Known budget phase required')
    gpus = spec['gpu_indices']
    require(len(gpus) == len(set(gpus)) and all(type(i) is int and 0 <= i < 8 for i in gpus), 'Distinct GPU indices required')
    uuids = {int(v['index']): u for u, v in g.query_modes().items()}
    upper = len(gpus) * spec['cap_seconds'] / 3600.0
    reserve = bool(spec.get('reserve_in_ledger', False))

    def take(b):
        prior = ledger_prior(b, spec['phase'])
        cap = b['stage_caps_GPUh'][PHASES[spec['phase']][1]]
        require(prior + upper <= cap, 'Stage cap exceeded: %.3f + %.3f > %s' % (prior, upper, cap))
        if reserve:
            require(not any(a['id'] == spec['name'] for a in b['allocations']), 'Allocation id already in ledger')
            b['allocations'].append(dict(id=spec['name'], phase=spec['phase'], charged_upper_GPUh=upper, terminal=False,
                                         reserved_at_prepare=True, host=os.uname().nodename))
            b[PHASES[spec['phase']][0]] = prior + upper
        return prior, cap
    budget = json.loads(BUDGET.read_text())
    prior, cap = take(json.loads(json.dumps(budget)))  # dry check before creating anything
    stage.mkdir(mode=0o700)
    if reserve:
        prior, cap = update_budget(take)
    sources = {n: sha_record(ROOT / n) for n in spec.get('sources', [])}
    snap = stage / 'executed_sources'
    for n in spec.get('sources', []):
        dst = snap / n; dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / n).read_bytes()); dst.chmod(0o600)
    plan = dict(schema='teacher_guarded_launch_plan_v1', name=spec['name'], phase=spec['phase'],
                purpose=spec['purpose'], command=spec['command'], env=spec.get('env', {}),
                exclusive_process=bool(spec.get('exclusive_process', False)), gpu_indices=gpus,
                gpu_uuids=[uuids[i] for i in gpus], cap_seconds=spec['cap_seconds'], upper_GPUh=upper,
                prior_phase_reserved_upper_GPUh=prior, stage_cap_GPUh=cap, hostname=os.uname().nodename,
                sources=sources, launcher=sha_record(Path(__file__).resolve()),
                resource_control_runtime=g.control_runtime(), output_root=str(stage / 'run01'),
                success_files=spec.get('success_files', []), automatic_retry_allowed=False, reserved_in_ledger=reserve,
                formal_training_allowed=bool(spec.get('formal_training_allowed', False)), sota_proven=False)
    row = save_new(stage / 'plan.json', plan)
    print(json.dumps(dict(plan=row, upper_GPUh=upper, prior=prior)))


def descendant_pids(root):
    """root plus all live descendants by parent links (torchrun ranks may start new sessions)."""
    children = {}
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                children.setdefault(int((p / 'stat').read_text().rsplit(')', 1)[1].split()[1]), []).append(int(p.name))
            except (OSError, IndexError, ValueError):
                pass
    out, stack = set(), [root]
    while stack:
        pid = stack.pop()
        if pid not in out:
            out.add(pid); stack.extend(children.get(pid, ()))
    return out


def terminate_tree(pids, grace=60.0):
    """TERM every pid of a snapshotted owned tree (ranks in their own sessions are outside the child's process
    group and would otherwise be orphaned holding GPUs), wait up to grace seconds, then KILL the rest.
    Returns the pids still alive afterwards (e.g. stuck in uninterruptible I/O)."""
    def alive():
        return [q for q in pids if Path('/proc/%d' % q).exists()
                and Path('/proc/%d/stat' % q).read_text().rsplit(')', 1)[1].split()[0] != 'Z']
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 10.0)):
        for q in alive():
            try:
                os.kill(q, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + wait
        while alive() and time.monotonic() < deadline:
            time.sleep(0.2)
    return alive()


RESERVE = '/tmp/od_reserve_%d'  # hold.sh idle skips GPUs whose reserve marker is younger than 1200 s


MS12K_LOCK = '/tmp/ms12k_gpu_%d.lock'  # omnidance resource-manager GPU lock (user-approved lock protocol)


def hold_foreign_locks(indices):
    """flock every existing ms12k GPU lock for the run; a lock held by someone else means the GPU is in use."""
    import fcntl
    fds = []
    try:
        for i in indices:
            p = MS12K_LOCK % i
            if not os.path.exists(p):
                continue
            fd = os.open(p, os.O_RDWR | os.O_NOFOLLOW)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                raise RuntimeError('Foreign ms12k lock held on GPU %d; never override' % i)
            fds.append(fd)
    except BaseException:
        for fd in fds:
            os.close(fd)
        raise
    return fds


def touch_reserve(indices):
    for i in indices:
        Path(RESERVE % i).touch()


def pending_holders(g, gpu):
    """Recognised holder processes bound to this GPU, including ones still importing torch (no CUDA context yet)."""
    out = []
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                if g.holder_identity(int(p.name), gpu):
                    out.append(int(p.name))
            except (OSError, ValueError):
                pass
    return out


def run(a):
    g = guard()
    require(a.gpu_authorized and Path(sys.argv[0]).is_absolute(), 'Explicit GPU authorization and absolute path required')
    started = time.monotonic(); started_wall = time.time()
    raw = Path(a.plan).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == a.sha, 'Plan SHA mismatch')
    p = json.loads(raw)
    require(p['hostname'] == os.uname().nodename and p['launcher'] == sha_record(Path(__file__).resolve()), 'Host/launcher drift')
    require(p['resource_control_runtime'] == g.control_runtime(), 'PIDFD controller drift')
    for n, row in p['sources'].items():
        require(sha_record(ROOT / n) == row, 'Source drift: ' + n)
    budget = json.loads(BUDGET.read_text())
    if p.get('reserved_in_ledger'):
        row = [a for a in budget['allocations'] if a['id'] == p['name']]
        require(len(row) == 1 and row[0]['terminal'] is False and row[0]['charged_upper_GPUh'] == p['upper_GPUh'],
                'Prepared ledger reservation missing or already terminal')
    else:
        require(ledger_prior(budget, p['phase']) == p['prior_phase_reserved_upper_GPUh'], 'Budget ledger moved since plan')
    run_dir = Path(p['output_root']); run_dir.mkdir(mode=0o700)
    save_new(run_dir / 'attempt.claim', dict(plan_sha256=a.sha, started_wall=started_wall))
    modes = g.query_modes(); initial = {u: modes[u] for u in p['gpu_uuids']}
    require([initial[u]['index'] for u in p['gpu_uuids']] == p['gpu_indices'], 'GPU index/UUID drift')
    leases, signals, intrusions, lock_fds = [], [], [], []
    child = None; error = None; stop = None
    def interrupted(*_):
        raise TimeoutError('Finite cap or owned termination')
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(s, interrupted)
    signal.setitimer(signal.ITIMER_REAL, max(.01, p['cap_seconds'] - 60 - (time.monotonic() - started)))
    try:
        LEASE_DIR.mkdir(mode=0o700, exist_ok=True)
        for u in p['gpu_uuids']:
            f = LEASE_DIR / (u + '.json')
            with f.open('x') as out:
                json.dump(dict(gpu_uuid=u, expires_epoch=started_wall + p['cap_seconds'], owner_pid=os.getpid()), out)
            f.chmod(0o600); leases.append(f)
        touch_reserve(p['gpu_indices'])
        lock_fds = hold_foreign_locks(p['gpu_indices'])
        for u in p['gpu_uuids']:
            if p['exclusive_process']:
                subprocess.run(['nvidia-smi', '-i', u, '-c', '3'], check=True, capture_output=True, timeout=10)
            for pid in g.apps().get(u, set()):
                v = g.signal_holder(pid, initial[u]['index'])
                require(v is not None, 'Unknown GPU owner; never signal')
                signals.append(v)
            for pid in pending_holders(g, initial[u]['index']):
                v = g.signal_holder(pid, initial[u]['index'])
                if v is not None:
                    signals.append(dict(v, pending_import=True))
        require(all(not g.apps().get(u) for u in p['gpu_uuids']), 'GPUs not free after known-holder release')
        env = dict(os.environ, **p['env'], CUDA_VISIBLE_DEVICES=','.join(p['gpu_uuids']))
        with (run_dir / 'command.log').open('xb') as log:
            child = subprocess.Popen(p['command'], stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True,
                                     cwd=str(ROOT))
            save_new(run_dir / 'launch.json', dict(parent_pid=os.getpid(), child_pid=child.pid, started_wall=started_wall))
            last_touch = time.monotonic()
            while child.poll() is None:
                time.sleep(5)
                if time.monotonic() - last_touch > 60:
                    touch_reserve(p['gpu_indices']); last_touch = time.monotonic()
                mine = descendant_pids(child.pid)
                for u in p['gpu_uuids']:
                    for pid in g.apps().get(u, set()) - mine:
                        intrusions.append(dict(gpu_uuid=u, pid=pid, observed=time.time()))
                if intrusions:
                    stop = 'external_gpu_process_observed'
                    raise RuntimeError(stop)
        require(child.returncode == 0, 'Command failed with exit %s; no retry' % child.returncode)
        for f in p['success_files']:
            require(Path(f).is_file(), 'Declared success file missing: ' + f)
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)[:300]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if child is not None and child.poll() is None:
            tree = descendant_pids(child.pid) - {os.getpid(), child.pid}  # snapshot before ranks get orphaned
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=10)
            stuck = terminate_tree(tree)
            if stuck:
                error = (error or '') + ' | descendants still alive after KILL: %s' % stuck[:16]
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and any(g.apps().get(u) for u in p['gpu_uuids']):
            time.sleep(1)
        queries = [g.apps(), g.apps()]
        released = all(not q.get(u) for q in queries for u in p['gpu_uuids'])
        for u, v in initial.items():
            subprocess.run(['nvidia-smi', '-i', u, '-c', '0' if v['mode'] == 'Default' else '3'], check=True, capture_output=True, timeout=10)
        restored = all(g.query_modes()[u]['mode'] == v['mode'] for u, v in initial.items())
        for f in leases:
            if json.loads(f.read_text())['owner_pid'] == os.getpid():
                f.unlink()
        for fd in lock_fds:
            os.close(fd)
        elapsed = time.monotonic() - started
        terminal = dict(terminal=True, name=p['name'], foreign_locks_held=len(lock_fds), phase=p['phase'], success=error is None and released and restored,
                        error=error, stop_reason=stop, returncode=None if child is None else child.returncode,
                        released=released, modes_restored=restored, holder_signals=signals, intrusions=intrusions,
                        elapsed_seconds=elapsed, actual_GPUh=elapsed * len(p['gpu_uuids']) / 3600.0,
                        charged_upper_GPUh=p['upper_GPUh'], prior_phase_reserved_upper_GPUh=p['prior_phase_reserved_upper_GPUh'],
                        cumulative_phase_reserved_upper_GPUh=p['prior_phase_reserved_upper_GPUh'] + p['upper_GPUh'],
                        formal_training_allowed=p['formal_training_allowed'], sota_proven=False)
        save_new(run_dir / 'terminal.json', terminal)
        if p.get('reserved_in_ledger'):
            def close(b):
                for a in b['allocations']:
                    if a['id'] == p['name'] and a['terminal'] is False:
                        a.update(terminal=True, success=terminal['success'], actual_GPUh=terminal['actual_GPUh'],
                                 error=terminal['error'])
            update_budget(close)
        print(json.dumps({k: terminal[k] for k in ('success', 'error', 'released', 'modes_restored', 'actual_GPUh')}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('prepare'); s.add_argument('--spec', required=True); s.add_argument('--stage-dir', required=True)
    s = sub.add_parser('run'); s.add_argument('--plan', required=True); s.add_argument('--sha', required=True)
    s.add_argument('--gpu-authorized', action='store_true')
    a = ap.parse_args()
    prepare(a) if a.cmd == 'prepare' else run(a)


if __name__ == '__main__':
    main()
