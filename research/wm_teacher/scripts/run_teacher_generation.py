#!/usr/bin/env python3
"""One-shot generation supervision; frozen CPU plans never authorize a GPU.

Execution is a separate externally approved sidecar. Allocation clocks begin
before gpu_hold exec and finish only after owned-session and GPU release.
"""
import argparse
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import prepare_teacher_recovery as recovery

require, encoded, sha, record = recovery.require, recovery.encoded, recovery.sha, recovery.record
dev, smoke = recovery.dev, recovery.dev.smoke
SCHEMA = 'teacher_generation_execution_v1'
RECEIPT_SCHEMA = 'teacher_generation_parent_receipt_v1'
METHODS = ('base', 'alg', 'dymos')
# Keep termination + several ten-second device queries within every hard cap.
CLEANUP_SECONDS = 120
WORKER_MARGIN_SECONDS = 10
KILL_SECONDS = 20
QUERY_COMMAND = ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits']


def read_bound(row):
    path = Path(row['path']).absolute()
    require(path.resolve() == path, 'Evidence symlink forbidden')
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            and len(raw) == row['bytes'] and sha(raw) == row['sha256'],
            'Evidence content drift during single-pass read')
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate evidence JSON key')
            result[key] = value
        return result
    def constant(_value):
        raise ValueError('Nonfinite evidence JSON constant')
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    import prepare_teacher_comparison as comparison
    comparison.finite_evidence(value)
    return value

def identity(pid):
    fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    args = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
    return dict(pid=pid, ppid=int(fields[1]), pgid=int(fields[2]), session=int(fields[3]),
                start_ticks=int(fields[19]), state=fields[0], argv=[os.fsdecode(a) for a in args if a])


def same_process(row):
    try:
        current = identity(row['pid'])
        return all(current[k] == row[k] for k in ('pid', 'session', 'start_ticks')) and current['state'] not in ('Z', 'X')
    except (OSError, ValueError, KeyError):
        return False


def session_members(owner):
    require(owner['session'] == owner['pid'] and owner['start_ticks'] > 0,
            'Cleanup requires a fresh owned session identity')
    try:
        current_leader = identity(owner['pid'])
    except (OSError, ValueError):
        current_leader = None
    require(current_leader is None or current_leader['start_ticks'] == owner['start_ticks'],
            'Owned session leader PID/start identity was reused; never signal new owner')
    result = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            row = identity(int(entry.name))
            if row['session'] == owner['session'] and row['state'] not in ('Z', 'X'):
                # A newer session with a reused leader PID must never be signalled.
                require(row['start_ticks'] >= owner['start_ticks'], 'Session identity reuse')
                result.append(row)
        except (OSError, ValueError):
            continue
    return result


def owned_members(owner, tracked):
    rows = session_members(owner)
    known = {r['pid']: r for r in tracked if same_process(r)}
    known.update({r['pid']: r for r in rows})
    snapshot = []
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                snapshot.append(identity(int(entry.name)))
            except (OSError, ValueError):
                pass
    while True:
        children = [r for r in snapshot if r['pid'] not in known and r['ppid'] in known
                    and r['start_ticks'] >= known[r['ppid']]['start_ticks'] and r['state'] not in ('Z', 'X')]
        if not children:
            break
        known.update({r['pid']: r for r in children})
    for row in known.values():
        if not any(r['pid'] == row['pid'] and r['start_ticks'] == row['start_ticks'] for r in tracked):
            tracked.append(row)
    return list(known.values())


def cleanup_owned(child, owner, deadline, tracked=None):
    """Signal individually rechecked owned identities, including observed escaped children."""
    tracked = [] if tracked is None else tracked
    signalled = []
    for signum, allowance in ((signal.SIGTERM, KILL_SECONDS), (signal.SIGKILL, 2)):
        for row in owned_members(owner, tracked):
            if same_process(row):
                os.kill(row['pid'], signum)
                signalled.append(dict(identity=row, signal=signum))
        until = min(deadline, time.monotonic() + allowance)
        while owned_members(owner, tracked) and time.monotonic() < until:
            child.poll()
            time.sleep(.02)
    child.wait(timeout=max(.01, deadline - time.monotonic()))
    require(not owned_members(owner, tracked), 'Owned tree remains alive; quota stays reserved')
    return signalled


def fixed_allocations(plan):
    if plan['method'] != 'base':
        return recovery.allocations(plan['method'])
    require('allocations' in plan, 'Native base needs its explicitly frozen physical node mapping')
    rows = plan['allocations']
    require(len(rows) == 16 and [r['shard'] for r in rows] == list(range(16))
            and all(r['cap_seconds'] == 4500 and r['allocated_gpus'] == 1 for r in rows), 'Native all16 cap drift')
    require(len({r['node_rank'] for r in rows}) == 2
            and all(len([a for a in rows if a['node_rank'] == rank]) == 8 for rank in {r['node_rank'] for r in rows})
            and all({a['gpu_index'] for a in rows if a['node_rank'] == rank} == set(range(8))
                    for rank in {r['node_rank'] for r in rows}), 'Native explicit two-owner/eight-GPU mapping required')
    return rows


def sources(root):
    return [Path(root) / p for p in ('scripts/run_teacher_generation.py', 'tests/test_teacher_generation.py',
                                    'scripts/verify_teacher_history.py', 'tests/test_verify_teacher_history.py')]


def validate_sidecar(value):
    require(value.get('schema') == SCHEMA and value.get('private') is True
            and value.get('execution_id') == recovery.EXECUTION_ID and value.get('GPU_authorized') is True
            and value.get('GPU_used') is False and value.get('quality_pass') is False
            and value.get('sota_claim') is False, 'Explicit independent execution authorization required')
    require(set(value['plans']) == set(METHODS) and set(value['allocations']) == set(METHODS)
            and set(value['owners']) == set(METHODS), 'Freeze all three methods/all16 before any GPU')
    require(value['cleanup_seconds'] == CLEANUP_SECONDS and value['worker_margin_seconds'] == WORKER_MARGIN_SECONDS
            and value['kill_seconds'] == KILL_SECONDS
            and value['canary_shard'] == 0 and value['canary_job_count'] == 12
            and value['retry_allowed'] is False and value['borrow_allowed'] is False, 'One-shot fixed formal canary drift')
    for method in METHODS:
        plan = read_bound(value['plans'][method])
        if method == 'base':
            import run_teacher_base as base
            base.validate_contract(plan)
        else:
            recovery.validate_plan(plan)
        require(plan.get('common_contract_frozen', method == 'base') is True, 'Unfrozen CPU common contract')
        require(value['allocations'][method] == fixed_allocations(plan), 'All16 immutable caps/owner drift')
        require(sum(a['cap_seconds'] for a in value['allocations'][method])
                + (0 if method == 'base' else recovery.PRIOR[method][2]) <= plan['phase_seconds']['generation'],
                'Prior upper charged exactly once; no borrowed budget')
        owners = value['owners'][method]
        require(len(owners) == 2 and {r['node_rank'] for r in owners} == {a['node_rank'] for a in value['allocations'][method]}
                and len({r['hostname'] for r in owners}) == 2, 'Exactly two distinct node-owner CPU proofs required')
        for owner in owners:
            proof = read_bound(owner['proof'])
            require(proof.get('schema') == 'teacher_generation_owner_cpu_proof_v1'
                    and proof.get('GPU_used') is False and proof['method'] == method
                    and proof['plan_sha256'] == value['plans'][method]['sha256']
                    and proof['node_rank'] == owner['node_rank'] and proof['hostname'] == owner['hostname'],
                    'CPU proof owner/content/plan mismatch')
            expected = [a for a in value['allocations'][method] if a['node_rank'] == owner['node_rank']]
            require(proof['allocations'] == expected and proof['source_files'] == plan['source_files']
                    and proof['model_content_sha256'] == plan['model_content_sha256'], 'CPU proof common/source/model drift')
            reservation = proof['reservation_snapshot']
            require(reservation == dict(schema='teacher_generation_owner_cpu_allocations_v1',
                    plan_sha256=value['plans'][method]['sha256'], method=method, node_rank=owner['node_rank'],
                    hostname=owner['hostname'], GPU_used=False,
                    allocations=[dict(a, status='reserved', GPU_used=False) for a in expected]),
                    'Both real node-local CPU owner reservation snapshots are required')
            expected_local = recovery.LOCAL_ROOT / f'teacher_{method}_owner_cpu_{recovery.EXECUTION_ID}/reservations.json'
            require(proof['local_reservations']['path'] == str(expected_local)
                    and proof['local_reservations']['sha256'] == sha(encoded(reservation))
                    and proof['local_reservations']['bytes'] == len(encoded(reservation))
                    and type(proof['lock_identity']['inode']) is int and proof['lock_identity']['inode'] > 0,
                    'Real local CPU reservation path/stable lock inode required')
            if owner['hostname'] == os.uname().nodename:
                require(read_bound(proof['local_reservations']) == reservation, 'Actual own8 immutable CPU reservation drift')
                lock = expected_local.with_name(expected_local.name + '.lock').stat()
                require(proof['lock_identity'] == dict(device=lock.st_dev, inode=lock.st_ino), 'Actual own CPU lock inode drift')
            import run_teacher_base as base
            require(base.model_identity(proof['models']) == base.model_identity(plan['models'])
                    and base.runtime_identity(proof['runtime']) == base.runtime_identity(plan['runtime']), 'CPU proof runtime/model tuple mismatch')
            probe = proof['probe_snapshot']
            if owner['hostname'] == os.uname().nodename:
                require(read_bound(proof['probe']) == probe, 'Actual own CPU probe snapshot drift')
            require(probe.get('schema') == 'teacher_real_local_cpu_probe_v1' and probe.get('GPU_used') is False
                    and probe['hostname'] == owner['hostname'] and probe['counter'] == 40
                    and probe['unique_entries'] == 40 and probe['errors'] == [] and probe['exit_codes'] == [0] * 4
                    and probe['read_checks'] > 0, 'Actual isolated local CPU lock/write evidence required')
            filesystem = proof['filesystem']
            require(proof['local_root'] == str(recovery.LOCAL_ROOT) and filesystem['node_local'] is True
                    and filesystem['type'] not in ('nfs', 'nfs4', 'ceph', 'lustre', 'fuse.ceph')
                    and filesystem['command'] == ['stat', '-f', '-c', '%T %t', str(recovery.LOCAL_ROOT)]
                    and filesystem['returncode'] == 0 and filesystem['stderr'] == ''
                    and filesystem['stdout'].strip().split() == [filesystem['type'], filesystem['magic']]
                    and type(filesystem['device']) is int and filesystem['device'] > 0
                    and proof['lock_identity']['device'] == filesystem['device'],
                    'Raw node-local filesystem evidence required; shared lock failure is not reusable')
            pf = proof['node_preflight_snapshot']
            if owner['hostname'] == os.uname().nodename:
                require(read_bound(proof['node_preflight']) == pf, 'Actual own preflight snapshot drift')
            require(pf['schema'] == 'e3_subset_node_preflight_v1' and pf['GPU_used'] is False
                    and pf['hostname'] == owner['hostname'] and pf['models'] == proof['models']
                    and pf['model_content_sha256'] == plan['model_content_sha256'], 'Owner actual preflight content tuple drift')
            frozen_sources = read_bound(value['plans']['alg'])['source_files']
            require(probe['source'] in frozen_sources and probe['test_source'] in frozen_sources,
                    'CPU probe must bind exactly frozen protocol/test sources')
    return value


def verify_retirement(row, method, plan):
    proof = read_bound(row)
    require(method == plan['method'], 'Retirement method mismatch')
    import verify_teacher_history as history
    return history.verify_retirement(proof, plan)

def validate_authorization(value):
    authorization = read_bound(value['authorization'])
    require(authorization.get('schema') == 'teacher_generation_authorization_v1'
            and authorization.get('GPU_authorized') is True and authorization['execution_id'] == recovery.EXECUTION_ID
            and authorization['methods'] == list(METHODS) and authorization['comparison'] == value['comparison']
            and authorization['allocations'] == value['allocations'] and authorization['canary_shard'] == 0
            and authorization['retry_allowed'] is False and authorization['borrow_allowed'] is False,
            'CLI boolean is not approved user execution authorization')
    return authorization


def scoring_owner_content(bundle):
    """Content tuple from approved bundle bytes; no peer-local filesystem reads."""
    import score_teacher_devset as score
    execution = bundle.get('execution')
    require(bundle.get('schema') == score.BUNDLE_SCHEMA and bundle.get('private') is True
            and isinstance(execution, dict) and set(execution) == {'python', 'vbench_repo', 'tools', 'offline', 'gpu_dimensions', 'gt_K', 'vbench_local', 'cache_dir'}
            and execution['offline'] is True and execution['gt_K'] == 12
            and execution['vbench_local'] is True and execution['cache_dir'] == bundle['model_roots']['vbench']
            and execution['gpu_dimensions'] == score.e1.DIMS, 'Actual scoring execution bundle required before generation')
    runtime = bundle['runtime']
    require(set(execution['tools']) == {'python', 'ffmpeg', 'ffprobe', 'timeout', 'nvidia_smi'}
            and 'observation' in runtime and runtime['observation']['versions'] == runtime['versions']
            and runtime['observation']['dependency_roots'] == bundle['dependency_roots']
            and runtime['observation']['library_roots'] == runtime['library_roots'], 'Actual scoring observation tuple required')
    require(set(execution['tools'].values()) <= {r['path'] for r in runtime['files']}, 'Unbound scoring executable content')
    digests = {}
    for name, rows in (('sources', bundle['sources']), ('models', bundle['models']),
                       ('dependencies', bundle['dependencies']), ('runtime', runtime['files']),
                       ('libraries', runtime['library_files'])):
        require(isinstance(rows, list) and rows and len({r['path'] for r in rows}) == len(rows), 'Missing/duplicate owner content inventory')
        content = sorted([{k: r[k] for k in ('path', 'bytes', 'sha256')} for r in rows], key=lambda r: r['path'])
        require(all(type(r['bytes']) is int and r['bytes'] >= 0 and score.is_sha(r['sha256'])
                    and Path(r['path']).is_absolute() for r in content), 'Malformed owner content record')
        digests[name] = sha(encoded(content))
    require({r['path'] for r in bundle['sources']} == set(score.required_sources()), 'Owner scoring source set drift')
    return dict(content_sha256=digests, observation=runtime['observation'], execution=execution,
                dependency_roots=bundle['dependency_roots'], model_roots=bundle['model_roots'],
                library_roots=runtime['library_roots'])


def scoring_owner_sources():
    return [Path(__file__).resolve(), Path(__file__).resolve().parents[1] / 'tests/test_teacher_generation.py',
            Path(__file__).resolve().with_name('run_teacher_scoring.py'),
            Path(__file__).resolve().with_name('score_teacher_devset.py')]


def prepare_scoring_owner_proof(bundle_path, approved_bundle_sha, output, *, node_rank=0):
    require(node_rank == 0, 'The actual scoring environment is owned only by rank0')
    bundle_row = record(bundle_path, nonempty=True)
    require(bundle_row['sha256'] == approved_bundle_sha, 'Approved actual scoring bundle SHA required')
    import score_teacher_devset as score
    import run_teacher_scoring as scoring
    bundle = score.verify_bundle(bundle_path, approved_bundle_sha, verify_content=True)
    require('execution' in bundle, 'Legacy CPU supplier inventory cannot authorize generation')
    scoring.validate_execution_bundle(bundle, observe=False, verify_content=False)
    proof = dict(schema='teacher_generation_scoring_owner_proof_v1', private=True, GPU_used=False,
        node_rank=0, hostname=os.uname().nodename, bundle=bundle_row, content=scoring_owner_content(bundle),
        observed_epoch=time.time(), observed_monotonic=time.monotonic(),
        verifier_sources=[record(p, nonempty=True) for p in scoring_owner_sources()])
    require(record(bundle_path, nonempty=True) == bundle_row, 'Scoring bundle changed during owner validation')
    dev.exclusive(output, proof)
    return Path(output), sha(Path(output).read_bytes())


def verify_scoring_owner(value, *, full_content=False):
    owner = value.get('scoring_owner')
    require(isinstance(owner, dict) and owner.get('node_rank') == 0 and isinstance(owner.get('hostname'), str)
            and owner['hostname'] and isinstance(owner.get('proof'), dict), 'Approved rank0 scoring owner proof required')
    proof = read_bound(owner['proof'])
    bundle = read_bound(value['score_bundle'])
    require(proof.get('schema') == 'teacher_generation_scoring_owner_proof_v1' and proof.get('private') is True
            and proof.get('GPU_used') is False and proof.get('node_rank') == 0 and proof.get('hostname') == owner['hostname']
            and proof['bundle'] == value['score_bundle'] and proof['content'] == scoring_owner_content(bundle)
            and recovery.finite(proof.get('observed_epoch')) and proof['observed_epoch'] > 0
            and recovery.finite(proof.get('observed_monotonic')) and proof['observed_monotonic'] > 0,
            'Approved owner proof bundle/source/model/runtime/observation content drift')
    expected = {str(p) for p in scoring_owner_sources()}
    require(len(proof['verifier_sources']) == len(expected)
            and {r['path'] for r in proof['verifier_sources']} == expected, 'Scoring owner verifier source freeze required')
    for row in proof['verifier_sources'] + bundle['sources']:
        recovery.check_content(row)
    if full_content:
        require(owner['hostname'] == os.uname().nodename, 'Only the actual scoring host may verify local metric assets')
        import score_teacher_devset as score
        import run_teacher_scoring as scoring
        checked = score.verify_bundle(value['score_bundle']['path'], value['score_bundle']['sha256'], verify_content=True)
        scoring.validate_execution_bundle(checked, observe=False, verify_content=False)
        require(scoring_owner_content(checked) == proof['content'], 'Actual scoring owner environment drift')
    return proof


def verify_execution(path, approved_sha, verify_content=True):
    row = record(path, nonempty=True)
    require(row['sha256'] == approved_sha, 'Approved execution SHA mismatch')
    value = validate_sidecar(read_bound(row))
    require(Path(path).absolute().resolve() == Path(path).absolute()
            and Path(path).stat().st_mode & 0o077 == 0, 'Private immutable execution sidecar required')
    validate_authorization(value)
    import prepare_teacher_comparison as comparison
    proof = verify_scoring_owner(value)
    scoring_host = proof['hostname'] == os.uname().nodename
    common = (comparison.verify if scoring_host else comparison.verify_metadata)(
        value['comparison']['path'], value['comparison']['sha256'])
    if scoring_host:
        verify_scoring_owner(value, full_content=True)
    require(common['plans'] == value['plans'] and common['common_execution_sha256'] == value['common_execution_sha256']
            and common['score_bundle'] == value['score_bundle'], 'Execution common manifest drift')
    require(set(value['retirement']) == {'alg', 'dymos'}, 'Both old reservation retirement proofs required')
    for method in ('alg', 'dymos'):
        verify_retirement(value['retirement'][method], method, read_bound(value['plans'][method]))
    if verify_content:
        for source in value['source_files']:
            recovery.check_content(source)
        expected = sources(read_bound(value['plans']['base'])['root'])
        require(len(value['source_files']) == len(expected)
                and {r['path'] for r in value['source_files']} == {str(p) for p in expected}, 'Driver/test source freeze required')
    require(record(path, nonempty=True) == row, 'Execution changed during verification')
    return value


def verify_worker_execution(path, approved_sha, launch):
    """Generation-only validation under a live, approved full-preflight parent.

    Scoring-host assets are not used during generation. The worker separately
    rehashes its own generation models, runtime, sources and inputs before CUDA.
    """
    row = record(path, nonempty=True)
    require(row['sha256'] == approved_sha and launch['execution'] == row
            and launch['execution_sha256'] == approved_sha, 'Worker approved execution binding drift')
    value = validate_sidecar(read_bound(row))
    require(Path(path).resolve() == Path(path).absolute() and Path(path).stat().st_mode & 0o077 == 0,
            'Worker execution must be private and immutable')
    validate_authorization(value)
    verify_scoring_owner(value)
    import prepare_teacher_comparison as comparison
    common = comparison.verify_metadata(value['comparison']['path'], value['comparison']['sha256'])
    require(common['plans'] == value['plans'] and common['common_execution_sha256'] == value['common_execution_sha256']
            and common['score_bundle'] == value['score_bundle'], 'Worker common contract drift')
    method, shard, rank = launch['method'], launch['shard'], launch['allocation']['node_rank']
    require(method in METHODS and type(shard) is int and 0 <= shard < 16
            and value['allocations'][method][shard] == launch['allocation']
            and launch['plan'] == value['plans'][method], 'Worker method/shard/plan drift')
    require(same_process(launch['owner']), 'Live full-preflight parent is missing or reused')
    parent = identity(launch['owner']['pid'])
    require(all(parent[k] == launch['owner'][k] for k in ('pid', 'ppid', 'pgid', 'session', 'start_ticks', 'argv')),
            'Worker parent identity changed after full preflight')
    args = parent['argv']
    require(str(Path(__file__).resolve()) in args and '--run' in args and '--gpu-authorized' in args,
            'Worker parent is not the approved generation supervisor')
    for flag, expected in (('--execution', str(Path(path).absolute())), ('--approved-execution-sha256', approved_sha),
                           ('--method', method), ('--shard', str(shard)), ('--rank', str(rank))):
        require(args.count(flag) == 1 and args[args.index(flag) + 1] == expected,
                'Worker parent CLI does not match the immutable launch')
    require(set(value['retirement']) == {'alg', 'dymos'}, 'Both historical retirement proofs required')
    for candidate in ('alg', 'dymos'):
        verify_retirement(value['retirement'][candidate], candidate, read_bound(value['plans'][candidate]))
    require({r['path'] for r in value['source_files']} == {str(p) for p in sources(read_bound(value['plans']['base'])['root'])},
            'Worker supervisor source set drift')
    for source in value['source_files']:
        recovery.check_content(source)
    require(record(path, nonempty=True) == row, 'Worker execution changed during verification')
    return value


def reserve_owner_cpu(plan, digest, rank, local_root=recovery.LOCAL_ROOT):
    require(sha(encoded(plan)) == digest, 'CPU owner reservation plan digest drift')
    fixed = [a for a in fixed_allocations(plan) if a['node_rank'] == rank]
    require(len(fixed) == 8, 'Only own8 CPU reservations may be created')
    path = recovery.local_path(Path(local_root) / f'teacher_{plan["method"]}_owner_cpu_{recovery.EXECUTION_ID}' / 'reservations.json', local_root)
    require(not path.parent.exists(), 'CPU owner reservation exists; never reset or retry')
    path.parent.mkdir(mode=0o700)
    value = dict(schema='teacher_generation_owner_cpu_allocations_v1', plan_sha256=digest, method=plan['method'],
        node_rank=rank, hostname=os.uname().nodename, GPU_used=False,
        allocations=[dict(a, status='reserved', GPU_used=False) for a in fixed])
    with recovery.local_locked(path, local_root):
        dev.exclusive(path, value)
        lock = path.with_name(path.name + '.lock').stat()
    return path, value, dict(device=lock.st_dev, inode=lock.st_ino)


def prepare_owner_proof(plan_path, approved_plan_sha, method, rank, node_preflight, approved_node_sha,
                        probe_path, approved_probe_sha, output):
    """Run only during authorized CPU preflight, including real content hashes."""
    plan_row = record(plan_path, nonempty=True)
    require(plan_row['sha256'] == approved_plan_sha, 'Approved owner plan SHA required')
    plan = read_bound(plan_row)
    require(plan['method'] == method, 'Wrong owner method')
    fixed = [a for a in fixed_allocations(plan) if a['node_rank'] == rank]
    require(len(fixed) == 8, 'Explicit own-eight rank required')
    pf_row, probe_row = record(node_preflight, nonempty=True), record(probe_path, nonempty=True)
    require(pf_row['sha256'] == approved_node_sha and probe_row['sha256'] == approved_probe_sha, 'Approved preflight/probe SHAs required')
    pf = read_bound(pf_row)
    models = smoke.models_from_preflight(pf)
    runtime = smoke.runtime_records()
    import run_teacher_base as base
    require(base.model_identity(models) == base.model_identity(plan['models'])
            and base.runtime_identity(runtime) == base.runtime_identity(plan['runtime']), 'Real owner current model/runtime content drift')
    for row in plan['source_files']:
        recovery.check_content(row)
    _, cases, inputs, manifest = dev.frozen_cases(Path(plan['root']))
    require((cases, inputs, manifest) == (plan['cases'], plan['inputs'], plan['cases_file']), 'Real owner current input drift')
    probe = read_bound(probe_row)
    require(probe.get('schema') == 'teacher_real_local_cpu_probe_v1' and probe['hostname'] == os.uname().nodename
            and probe['GPU_used'] is False and probe['counter'] == 40 and probe['unique_entries'] == 40
            and probe['read_checks'] > 0 and probe['errors'] == [] and probe['exit_codes'] == [0] * 4,
            'Actual local CPU proof required')
    local_root = recovery.local_path(recovery.LOCAL_ROOT, recovery.LOCAL_ROOT)
    command = ['stat', '-f', '-c', '%T %t', str(local_root)]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    filesystem_type, magic = result.stdout.strip().split()
    require(filesystem_type not in ('nfs', 'nfs4', 'ceph', 'lustre', 'fuse.ceph'), 'Shared filesystem cannot replace local lock proof')
    reservation_path, reservation_snapshot, lock_identity = reserve_owner_cpu(plan, approved_plan_sha, rank)
    value = dict(schema='teacher_generation_owner_cpu_proof_v1', GPU_used=False, method=method, node_rank=rank,
        hostname=os.uname().nodename, plan=plan_row, plan_sha256=approved_plan_sha, allocations=fixed,
        node_preflight=pf_row, node_preflight_snapshot=pf, models=models, model_content_sha256=plan['model_content_sha256'], runtime=runtime,
        source_files=plan['source_files'], probe=probe_row, probe_snapshot=probe, local_root=str(local_root),
        local_reservations=record(reservation_path, nonempty=True), reservation_snapshot=reservation_snapshot,
        lock_identity=lock_identity,
        filesystem=dict(node_local=True, type=filesystem_type, magic=magic, command=command, stdout=result.stdout,
                        stderr=result.stderr, returncode=result.returncode, device=local_root.stat().st_dev),
        observed_monotonic=time.monotonic(), observed_epoch=time.time(), owner=identity(os.getpid()))
    dev.exclusive(output, value)
    return Path(output), sha(Path(output).read_bytes())


def prepare_execution(comparison_path, approved_comparison_sha, authorization, approved_authorization_sha,
                      owners, retirement, output, *, scoring_owner_proof=None, approved_scoring_owner_sha=None):
    """CPU content gate; caller must independently approve the resulting sidecar SHA."""
    import prepare_teacher_comparison as comparison
    require(scoring_owner_proof and approved_scoring_owner_sha, 'Explicit approved scoring-owner proof SHA required')
    owner_row = record(scoring_owner_proof, nonempty=True)
    require(owner_row['sha256'] == approved_scoring_owner_sha, 'Approved scoring-owner proof SHA mismatch')
    owner = read_bound(owner_row)
    require(owner['hostname'] == os.uname().nodename and owner['node_rank'] == 0, 'Full execution preparation belongs to rank0 scoring host')
    common = comparison.verify(comparison_path, approved_comparison_sha)
    plans = {m: read_bound(common['plans'][m]) for m in METHODS}
    value = dict(schema=SCHEMA, private=True, execution_id=recovery.EXECUTION_ID, GPU_authorized=True,
        GPU_used=False, quality_pass=False, sota_claim=False, comparison=record(comparison_path, nonempty=True),
        plans=common['plans'], allocations={m: fixed_allocations(p) for m, p in plans.items()}, owners=owners,
        authorization=record(authorization, nonempty=True), retirement=retirement,
        common_execution_sha256=common['common_execution_sha256'], score_bundle=common['score_bundle'],
        scoring_owner=dict(node_rank=0, hostname=owner['hostname'], proof=owner_row),
        source_files=[record(p, nonempty=True) for p in sources(plans['base']['root'])],
        cleanup_seconds=CLEANUP_SECONDS, worker_margin_seconds=WORKER_MARGIN_SECONDS, kill_seconds=KILL_SECONDS,
        canary_shard=0, canary_job_count=12,
        retry_allowed=False, borrow_allowed=False)
    require(value['authorization']['sha256'] == approved_authorization_sha, 'Explicit approved authorization SHA required')
    validate_sidecar(value)
    verify_scoring_owner(value, full_content=True)
    output = Path(output).absolute()
    require(output.resolve() == output and output.parent.stat().st_mode & 0o077 == 0 and not output.exists(),
            'New private execution sidecar only; no retry/overwrite')
    # Validate the full chain before publishing any execution preparation.
    validate_authorization(value)
    for method in ('alg', 'dymos'):
        verify_retirement(retirement[method], method, plans[method])
    dev.exclusive(output, value)
    return output, sha(output.read_bytes())


def ledger_path(execution, method, local_root):
    return recovery.local_path(Path(local_root) / f'teacher_{method}_generation_{execution["execution_id"]}' / 'allocations.json', local_root)


def validate_local(value, execution, digest, method, rank):
    require(value['schema'] == 'teacher_generation_local_ledger_v1' and value['execution_sha256'] == digest
            and value['method'] == method and value['node_rank'] == rank and value['hostname'] == os.uname().nodename,
            'Wrong exclusive local owner ledger')
    fixed = [a for a in execution['allocations'][method] if a['node_rank'] == rank]
    require(len(value['allocations']) == 8 and len(fixed) == 8, 'Only own8 mutable allocations allowed')
    for item, allocation in zip(value['allocations'], fixed):
        require(all(item.get(k) == v for k, v in allocation.items())
                and item['status'] in ('reserved', 'running', 'finished', 'failed'), 'Immutable quota drift')
    return value


def reserve_local(execution, digest, method, rank, local_root=recovery.LOCAL_ROOT):
    require(method in METHODS and sha(encoded(execution)) == digest, 'Explicit approved execution digest/method required')
    owner = next((o for o in execution['owners'][method] if o['node_rank'] == rank), None)
    require(owner is not None and owner['hostname'] == os.uname().nodename, 'Foreign node cannot own allocations')
    path = ledger_path(execution, method, local_root)
    require(not path.parent.exists(), 'Local execution already exists; never reset/retry')
    path.parent.mkdir(mode=0o700)
    value = dict(schema='teacher_generation_local_ledger_v1', execution_sha256=digest, method=method,
        node_rank=rank, hostname=os.uname().nodename,
        allocations=[dict(a, status='reserved') for a in execution['allocations'][method] if a['node_rank'] == rank])
    validate_local(value, execution, digest, method, rank)
    with recovery.local_locked(path, local_root):
        dev.exclusive(path, value)
    return path


@contextlib.contextmanager
def local_transaction(execution, digest, method, rank, local_root):
    path = ledger_path(execution, method, local_root)
    with recovery.local_locked(path, local_root):
        value = validate_local(json.loads(path.read_bytes()), execution, digest, method, rank)
        yield value
        validate_local(value, execution, digest, method, rank)
        recovery.durable_write(path, value)


def manager_boundary(state_dir, gpu, expected_uuid=None, require_idle=True):
    import gpu_hold
    state = gpu_hold.read_state(Path(state_dir))
    manager = identity(state.get('pid', 0))
    require(state.get('running') is True and state.get('monitor_only') is True and state.get('query_ok') is True
            and 0 <= time.time() - state['updated_at'] < 5 and gpu_hold.matching_process(manager['pid'], 'run')
            and '--monitor-only' in manager['argv'], 'Live no-hold manager PID/argv/fresh status required')
    item = state['gpus'][str(gpu)]
    mapping = gpu_hold.query_gpus()
    uuid = mapping[gpu]['uuid']
    require(item['gpu_uuid'] == uuid and item['holder_pid'] is None and item['memory_gib_target'] == 0
            and (expected_uuid is None or uuid == expected_uuid), 'Physical manager/device UUID or zero-holder mismatch')
    query = query_release(uuid)
    if require_idle:
        require(item['external_pids'] == [] and query['pids'] == [], 'Actual external compute blocks launch; never kill it')
    return dict(manager=manager, state=state, gpu_uuid=uuid, query=query)


def query_release(uuid):
    begin, epoch = time.monotonic(), time.time()
    result = subprocess.run(QUERY_COMMAND, check=True, capture_output=True, text=True, timeout=10)
    pids = []
    for row in csv.reader(io.StringIO(result.stdout)):
        if not row:
            continue
        require(len(row) == 2, 'Malformed raw device process query')
        if row[0].strip() == uuid:
            pids.append(int(row[1]))
    return dict(command=QUERY_COMMAND, gpu_uuid=uuid, start_monotonic=begin, start_epoch=epoch,
                end_monotonic=time.monotonic(), returncode=result.returncode, stdout=result.stdout,
                stderr=result.stderr, pids=sorted(pids))


def shard_folder(plan, shard):
    return Path(plan['output_root']) / f'generation_shard{shard}'


def remaining_admission(execution, method):
    """No quality-based selection; native canary must pass before remaining15."""
    import run_teacher_base as base
    base_plan = read_bound(execution['plans']['base'])
    path = shard_folder(base_plan, 0) / 'parent.json'
    receipt = json.loads(path.read_bytes())
    verify_parent_release(receipt, base_plan)
    require(receipt['success'] is True, 'Native formal canary did not fully complete/release')
    require(base.verify_native_canary_admission(receipt, base_plan) is True, 'Native fixed canary throughput failed')
    if method != 'base':
        plan = read_bound(execution['plans'][method])
        candidate = json.loads((shard_folder(plan, 0) / 'parent.json').read_bytes())
        verify_parent_release(candidate, plan)
        require(candidate['success'] is True and candidate['exit_code'] == 0 and candidate['error'] is None
                and candidate['interrupted'] is None and candidate['timed_out'] is False
                and candidate['elapsed_seconds'] <= execution['allocations'][method][0]['cap_seconds'],
                'Candidate fixed canary did not fully complete/release within original stage cap')
        worker = read_bound(candidate['worker'])
        require(worker['success'] is True and worker['error'] is None
                and [r['job'] for r in worker['cases']] == plan['jobs'][0::16], 'Canary is exactly12 formal jobs, not extra smoke')
        baseline, case_times = None, []
        for row in worker['cases']:
            verify_case(row, plan, execution['plans'][method]['sha256'])
            require(baseline is None or row['parameters_before'] == baseline, 'Canary parameter content differs across formal jobs')
            baseline = row['parameters_before']
            seconds = read_bound(row['case_receipt'])['elapsed_seconds']
            require(recovery.finite(seconds) and seconds > 0, 'Invalid actual canary case timing')
            case_times.append(seconds)
        load = worker.get('load_seconds')
        require(recovery.finite(load) and load > 0
                and load + sum(case_times) <= candidate['elapsed_seconds'] + 1e-6, 'Canary true load/case clocks exceed allocation')
        tail = CLEANUP_SECONDS + WORKER_MARGIN_SECONDS
        residual = max(tail, candidate['elapsed_seconds'] - load - sum(case_times))
        projected = []
        for allocation in execution['allocations'][method]:
            body = load + len(plan['jobs'][allocation['shard']::16]) * max(case_times)
            full = body + residual
            require(body < allocation['cap_seconds'] - tail and full < allocation['cap_seconds'],
                    'Candidate worst-case projection cannot preserve the full cleanup/worker tail within shard cap')
            projected.append(full)
        require(plan['prior_generation']['upper_seconds'] + sum(projected) <= plan['phase_seconds']['generation'],
                'Candidate worst-case projection plus old upper exceeds the independent generation stage')


def current_local_content(execution, method, rank):
    import verify_teacher_history as history
    require(history.inventory(rank)['matches'] == [], 'Old generation is still live; never overlap its reservations')
    plan = read_bound(execution['plans'][method])
    owner = next(o for o in execution['owners'][method] if o['node_rank'] == rank)
    require(owner['hostname'] == os.uname().nodename, 'Foreign owner launch denied')
    proof = read_bound(owner['proof'])
    for source in plan['source_files'] + execution['source_files'] + proof['runtime']['files']:
        recovery.check_content(source)
    local_models = smoke.models_from_preflight(read_bound(proof['node_preflight']))
    import run_teacher_base as base
    require(base.model_identity(local_models) == base.model_identity(plan['models'])
            and base.runtime_identity(smoke.runtime_records()) == base.runtime_identity(plan['runtime']), 'Current local model/runtime drift')
    _, cases, inputs, manifest = dev.frozen_cases(Path(plan['root']))
    require((cases, inputs, manifest) == (plan['cases'], plan['inputs'], plan['cases_file']), 'Current local inputs drift')
    return plan, local_models


def command_for(path, digest, method, shard, rank, state_dir, folder, cap):
    launch = read_bound(record(Path(folder) / 'launch.json', nonempty=True))
    plan = read_bound(launch['plan'])
    timeout_binary = plan['runtime']['tools']['timeout']
    version = subprocess.run([timeout_binary, '--version'], check=True, capture_output=True, text=True, timeout=10)
    require('GNU coreutils' in version.stdout, 'Authenticated real GNU timeout required')
    return [sys.executable, str(Path(__file__).resolve().with_name('gpu_hold.py')), '--action', 'exec',
        '--gpus', str(shard_allocation_from_folder(folder)['gpu_index']), '--state-dir', str(state_dir), '--',
        timeout_binary, '--signal=TERM', f'--kill-after={KILL_SECONDS}s', f'{cap - CLEANUP_SECONDS}s',
        sys.executable, str(Path(__file__).resolve()), '--worker', '--execution', str(path),
        '--approved-execution-sha256', digest, '--method', method, '--shard', str(shard), '--rank', str(rank),
        '--state-dir', str(state_dir), '--launch', str(folder / 'launch.json')]


def shard_allocation_from_folder(folder):
    return json.loads((folder / 'launch.json').read_bytes())['allocation']


def supervise(path, approved_sha, method, shard, rank, *, gpu_authorized=False,
              state_dir='/dockerdata/wm_logs/gpu_hold', local_root=recovery.LOCAL_ROOT):
    require(gpu_authorized is True, 'Explicit --gpu-authorized required in addition to approved sidecar')
    require(Path(local_root).absolute() == recovery.LOCAL_ROOT, 'GPU run requires the fixed approved node-local root')
    execution = verify_execution(path, approved_sha)
    require(method in METHODS and type(shard) is int and 0 <= shard < 16, 'Fixed formal shard required')
    plan, _ = current_local_content(execution, method, rank)
    allocation = execution['allocations'][method][shard]
    require(allocation['node_rank'] == rank, 'Foreign shard dispatch forbidden')
    if shard != 0:
        remaining_admission(execution, method)
    folder = shard_folder(plan, shard)
    require(not folder.exists(), 'One-shot launch exists; no retry or old output reuse')
    boundary = manager_boundary(state_dir, allocation['gpu_index'])
    owner = identity(os.getpid())
    child, session, error, interrupted, timed_out, queries, signals = None, None, None, None, False, [], []
    release_boundary, tracked = None, []
    old = {}
    def interrupt(signum, _frame):
        nonlocal interrupted
        interrupted = signum
    for signum in (signal.SIGTERM, signal.SIGINT):
        old[signum] = signal.signal(signum, interrupt)
    # Durable allocation start precedes holderexec wait, content verification,
    # worker load/compile/hash/save/decode and all cleanup/release queries.
    start, epoch = time.monotonic(), time.time()
    cap = allocation['cap_seconds']
    launch = dict(schema='teacher_generation_launch_v1', execution=record(path, nonempty=True),
        execution_sha256=approved_sha, plan=execution['plans'][method], plan_sha256=execution['plans'][method]['sha256'],
        method=method, shard=shard, allocation=allocation, hostname=os.uname().nodename, owner=owner,
        common_execution_sha256=execution['common_execution_sha256'], score_bundle_sha256=execution['score_bundle']['sha256'],
        start_monotonic=start, start_epoch=epoch, gpu_uuid=boundary['gpu_uuid'], boundary=boundary,
        local_root=str(Path(local_root).absolute()), state_dir=str(Path(state_dir).absolute()))
    success, tree_released, gpu_released = False, False, False
    try:
        with local_transaction(execution, approved_sha, method, rank, local_root) as ledger:
            item = next(a for a in ledger['allocations'] if a['shard'] == shard)
            require(item['status'] == 'reserved', 'Existing allocation owner; never relaunch')
            item.update(status='running', owner=owner, start_monotonic=start, start_epoch=epoch)
        folder.mkdir(mode=0o700)
        dev.exclusive(folder / 'launch.json', launch)
        require(time.monotonic() - start < cap - CLEANUP_SECONDS, 'Prelaunch clock exhausted')
        command = command_for(path, approved_sha, method, shard, rank, state_dir, folder, cap)
        launch['command'] = command
        # launch immutable: command recorded separately before spawning.
        dev.exclusive(folder / 'dispatch.json', dict(command=command, launch=record(folder / 'launch.json', nonempty=True)))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=boundary['gpu_uuid'], WORLD_SIZE='1', LOCAL_WORLD_SIZE='1',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', DIFFSYNTH_SKIP_DOWNLOAD='True', OMP_NUM_THREADS='1',
                   PYTHONPATH=str(Path(plan['root']) / 'DiffSynth-Studio'))
        with (folder / 'worker.log').open('xb') as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            session = identity(child.pid)
            require(session['session'] == child.pid and session['ppid'] == owner['pid'], 'Fresh parent-owned session required')
            dev.exclusive(folder / 'supervised.json', dict(schema='teacher_generation_supervised_v1',
                launch=record(folder / 'launch.json', nonempty=True), dispatch=record(folder / 'dispatch.json', nonempty=True),
                owner=owner, session=session, observed_monotonic=time.monotonic(), observed_epoch=time.time()))
            while child.poll() is None and interrupted is None:
                owned_members(session, tracked)
                if time.monotonic() >= start + cap - CLEANUP_SECONDS:
                    timed_out = True
                    break
                # Ledger damage or another owner must stop this owned tree, not
                # defer detection until final accounting or rewrite bad bytes.
                ledger = ledger_path(execution, method, local_root)
                with recovery.local_locked(ledger, local_root, timeout_seconds=1):
                    current = validate_local(read_bound(record(ledger, nonempty=True)), execution, approved_sha, method, rank)
                    item = next(a for a in current['allocations'] if a['shard'] == shard)
                    require(item['status'] == 'running' and item['owner'] == owner
                            and item['start_monotonic'] == start and item['start_epoch'] == epoch,
                            'Actual running ledger owner/clock changed')
                time.sleep(.05)
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        try:
            if child is not None and session is None:
                if child.poll() is None:
                    try:
                        recovered = identity(child.pid)
                        require(recovered['pid'] == recovered['session'] == child.pid
                                and recovered['ppid'] == owner['pid'] and recovered['start_ticks'] > 0,
                                'Cannot authenticate newly spawned child session')
                    except (OSError, ValueError, KeyError):
                        recovered = None
                    if recovered is not None:
                        signals = cleanup_owned(child, recovered, start + cap - 70, tracked)
                    else:
                        # An unreaped Popen child cannot have its PID reused.
                        # Never synthesize a session or signal an arbitrary group.
                        child.terminate()
                        try:
                            child.wait(timeout=max(.01, min(KILL_SECONDS, start + cap - 70 - time.monotonic())))
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait(timeout=max(.01, start + cap - 70 - time.monotonic()))
                else:
                    child.wait(timeout=max(.01, start + cap - 70 - time.monotonic()))
                error = f'{error}; spawned child reaped but original session/tree release proof remains unknown'
            if child is not None and session is not None:
                signals = cleanup_owned(child, session, start + cap - 70, tracked)
                tree_released = not owned_members(session, tracked)
                while time.monotonic() < start + cap - 10:
                    query = query_release(boundary['gpu_uuid'])
                    queries.append(query)
                    if not query['pids']:
                        if len(queries) >= 2 and not queries[-2]['pids']:
                            release_boundary = manager_boundary(state_dir, allocation['gpu_index'], boundary['gpu_uuid'], require_idle=False)
                            require(release_boundary['manager']['pid'] == boundary['manager']['pid']
                                    and release_boundary['manager']['start_ticks'] == boundary['manager']['start_ticks'],
                                    'No-hold manager identity changed during release')
                            gpu_released = True
                            break
                    time.sleep(.2)
            else:
                # No dispatched session is not a measured generation completion.
                tree_released = child is None
        except BaseException as exc:
            error = f'Cleanup/release failure: {type(exc).__name__}: {exc}'
        end, end_epoch = time.monotonic(), time.time()
        worker = None
        try:
            worker = record(folder / 'worker.json', nonempty=True)
            body = read_bound(worker)
            success = bool(body['success'] is True and child.returncode == 0 and tree_released and gpu_released
                           and interrupted is None and error is None and not timed_out and end - start <= cap)
        except (OSError, ValueError, KeyError, AttributeError):
            pass
        receipt = dict(allocation, schema=RECEIPT_SCHEMA, execution_id=recovery.EXECUTION_ID, method=method,
            execution=record(path, nonempty=True), execution_sha256=approved_sha, plan_sha256=execution['plans'][method]['sha256'],
            hostname=os.uname().nodename, owner_pid=owner['pid'], owner=owner, session=session,
            start=epoch, end=end_epoch, start_monotonic=start, end_monotonic=end, elapsed_seconds=end - start,
            owned_tree_released=tree_released, GPU_released=gpu_released, release_queries=queries,
            release_boundary=release_boundary, cleanup_signals=signals,
            owned_processes=tracked, final_session_members=[] if tree_released else (session_members(session) if session else None),
            final_owned_processes=[] if tree_released else (owned_members(session, tracked) if session else None),
            success=success, exit_code=child.returncode if child else None, timed_out=timed_out, interrupted=interrupted,
            error=error, worker=worker, launch=record(folder / 'launch.json', nonempty=True) if (folder / 'launch.json').exists() else None,
            supervised=record(folder / 'supervised.json', nonempty=True) if (folder / 'supervised.json').exists() else None,
            quality_pass=False, sota_claim=False)
        try:
            if folder.exists():
                dev.exclusive(folder / 'parent.json', receipt)
            with local_transaction(execution, approved_sha, method, rank, local_root) as ledger:
                item = next(a for a in ledger['allocations'] if a['shard'] == shard)
                require(item.get('owner') == owner, 'Never finalize another owner')
                item.update(status='finished' if success else 'failed', receipt=record(folder / 'parent.json', nonempty=True),
                            elapsed_seconds=end - start)
        finally:
            for signum, handler in old.items():
                signal.signal(signum, handler)
    require(success, 'Generation failed/unknown release: quota remains charged, no retry')
    verify_parent_release(receipt, plan)
    return folder / 'parent.json', sha((folder / 'parent.json').read_bytes())


def verify_worker_scope(execution, launch, method, shard, rank, state_dir):
    allocation = execution['allocations'][method][shard]
    require(launch['allocation'] == allocation and allocation['node_rank'] == rank
            and launch['hostname'] == os.uname().nodename and same_process(launch['owner'])
            and os.environ.get('CUDA_VISIBLE_DEVICES') == launch['gpu_uuid']
            and os.environ.get('WORLD_SIZE', '1') == '1' and os.environ.get('LOCAL_WORLD_SIZE', '1') == '1',
            'Worker live parent/single-GPU scope mismatch')
    ancestors, pid = [], os.getppid()
    while pid > 1:
        row = identity(pid)
        ancestors.append(row)
        pid = row['ppid']
    require(any(r['pid'] == launch['owner']['pid'] and r['start_ticks'] == launch['owner']['start_ticks']
                and '--run' in r['argv'] for r in ancestors), 'Live authorized parent ancestry required')
    require(any('gpu_hold.py' in {os.path.basename(a) for a in r['argv']} and '--action' in r['argv']
                and r['argv'][r['argv'].index('--action') + 1] == 'exec'
                and '--gpus' in r['argv'] and r['argv'][r['argv'].index('--gpus') + 1] == str(allocation['gpu_index'])
                for r in ancestors), 'Single physical-GPU gpu_hold exec ancestry required')
    require(any('timeout' in {os.path.basename(a) for a in r['argv']}
                and '--signal=TERM' in r['argv'] and f'--kill-after={KILL_SECONDS}s' in r['argv']
                and f'{allocation["cap_seconds"] - CLEANUP_SECONDS}s' in r['argv'] for r in ancestors), 'Real GNU timeout ancestry required')
    supervised = read_bound(record(shard_folder(read_bound(execution['plans'][method]), shard) / 'supervised.json', nonempty=True))
    require(any(r['pid'] == supervised['session']['pid'] and r['start_ticks'] == supervised['session']['start_ticks']
                for r in ancestors), 'Actual supervised session ancestry required')
    require(launch['local_root'] == str(recovery.LOCAL_ROOT)
            and launch['state_dir'] == str(Path(state_dir).absolute()), 'Worker fixed local ledger/manager boundary drift')
    local_root = Path(launch['local_root'])
    ledger = ledger_path(execution, method, local_root)
    with recovery.local_locked(ledger, local_root):
        state = validate_local(read_bound(record(ledger, nonempty=True)), execution, launch['execution_sha256'], method, rank)
        item = next(a for a in state['allocations'] if a['shard'] == shard)
        require(item['status'] == 'running' and item['owner'] == launch['owner']
                and item['start_monotonic'] == launch['start_monotonic'] and item['start_epoch'] == launch['start_epoch'],
                'Worker requires its actual durable running owner ledger/clock')
    boundary = manager_boundary(state_dir, allocation['gpu_index'], launch['gpu_uuid'], require_idle=False)
    require(boundary['manager']['pid'] == launch['boundary']['manager']['pid']
            and boundary['manager']['start_ticks'] == launch['boundary']['manager']['start_ticks'], 'Manager identity changed')
    owned = {r['pid'] for r in ancestors} | {os.getpid()}
    require(set(boundary['query']['pids']) <= owned, 'External GPU compute appeared; do not stop external owner')
    return ancestors


def execute_worker(path, digest, method, shard, rank, launch_path, state_dir):
    launch = read_bound(record(launch_path, nonempty=True))
    start, cap = launch['start_monotonic'], launch['allocation']['cap_seconds']
    def check():
        require(0 <= time.monotonic() - start < cap - CLEANUP_SECONDS - WORKER_MARGIN_SECONDS
                and abs((time.time() - launch['start_epoch']) - (time.monotonic() - start)) < 5,
                'Worker allocation clock exhausted/drifted')
    check()
    execution = verify_worker_execution(path, digest, launch)
    require(launch['execution_sha256'] == digest and launch['execution'] == record(path, nonempty=True), 'Worker launch binding drift')
    supervised_path = Path(launch_path).parent / 'supervised.json'
    while not supervised_path.exists():
        check()
        require(same_process(launch['owner']), 'Parent vanished before supervised session publication')
        time.sleep(.01)
    ancestors = verify_worker_scope(execution, launch, method, shard, rank, state_dir)
    folder = Path(launch_path).parent
    require(folder == shard_folder(read_bound(execution['plans'][method]), shard), 'Worker fresh output root drift')
    old = {}
    def interrupt(signum, _frame):
        raise RuntimeError(f'Worker interrupted by signal {signum}')
    for signum in (signal.SIGTERM, signal.SIGINT):
        old[signum] = signal.signal(signum, interrupt)
    evidence = dict(schema='teacher_generation_worker_v1', execution_sha256=digest, plan_sha256=launch['plan_sha256'],
        method=method, shard=shard, allocation=launch['allocation'], owner=identity(os.getpid()), ancestors=ancestors,
        launch=record(launch_path, nonempty=True), common_execution_sha256=execution['common_execution_sha256'],
        score_bundle_sha256=execution['score_bundle']['sha256'], success=False, cases=[], error=None,
        start_monotonic=start, start_epoch=launch['start_epoch'], load_seconds=None, quality_pass=False, sota_claim=False)
    try:
        plan, models = current_local_content(execution, method, rank)
        check()
        import torch
        require(torch.cuda.device_count() == 1, 'Exactly one visible CUDA GPU required')
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        import gen_jobs
        require(gen_jobs.NEG == plan['negative_prompt'], 'Imported native NEG drift')
        load_start = time.monotonic()
        pipe = smoke.build_pipe('NONE', models, gen_jobs)
        evidence['load_seconds'] = time.monotonic() - load_start
        check()
        if method == 'base':
            import run_teacher_base as base
            helper = base._generate_native_case
            context = None
        elif method == 'alg':
            from teacher_alg import teacher_alg
            helper, context = dev.generate_case, teacher_alg
        else:
            from teacher_dymos import teacher_dymos
            helper, context = dev.generate_case, teacher_dymos
        jobs = plan['jobs'][shard::16]
        require(len(jobs) == (12 if shard < 12 else 11), 'Formal frozen shard count mismatch')
        for job in jobs:
            check()
            case = plan['cases'][job['index'] % 94]
            input_binding = dict(case_id=case['id'], prompt_sha256=case['prompt_sha256'],
                image=record(case['image'], nonempty=True), gt=record(case['gt'], nonempty=True))
            # Capture the actual helper's per-job hashes, rather than a boolean.
            captured, original = [], smoke.parameter_hashes
            def hashes(*args, **kwargs):
                value = original(*args, **kwargs)
                captured.append(value)
                return value
            smoke.parameter_hashes = hashes
            try:
                if method == 'base':
                    body = helper(pipe, torch, gen_jobs, plan, launch['plan_sha256'], job, check)
                else:
                    body = helper(pipe, torch, gen_jobs, context, plan, launch['plan_sha256'], job, check)
            finally:
                smoke.parameter_hashes = original
            require(len(captured) == 2 and captured[0] == captured[1], 'Actual before/after parameter SHA evidence required')
            case_row = dict(job=job, case_receipt=record(Path(job['output']).with_suffix('.receipt.json'), nonempty=True),
                            video=body['video'], parameters_before=captured[0], parameters_after=captured[1],
                            input_binding=input_binding, stats=body['stats'])
            if method != 'base':
                case_row['claim'] = record(Path(job['output']).with_suffix('.claim.json'), nonempty=True)
                require(read_bound(case_row['claim']) == dict(plan_sha256=launch['plan_sha256'], job=job,
                    pid=os.getpid(), hostname=os.uname().nodename), 'Actual helper claim/owner drift')
            verify_case(case_row, plan, launch['plan_sha256'], decode=False)
            evidence['cases'].append(case_row)
            recovery.durable_write(folder / 'progress.json', evidence)
        check()
        evidence['success'] = True
    except BaseException as exc:
        evidence['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        evidence.update(end_monotonic=time.monotonic(), end_epoch=time.time())
        dev.exclusive(folder / 'worker.json', evidence)
        for signum, handler in old.items():
            signal.signal(signum, handler)
    return evidence


def verify_case(row, plan, plan_sha, decode=True):
    job = row['job']
    require(type(job['index']) is int and 0 <= job['index'] < 188 and plan['jobs'][job['index']] == job, 'Foreign/duplicate case job')
    receipt = read_bound(row['case_receipt'])
    require(receipt['success'] is True and receipt['failure'] is False and receipt['method'] == plan['method']
            and receipt['plan_sha256'] == plan_sha and receipt['job'] == job and receipt['parameters_unchanged'] is True,
            'Case receipt is not frozen successful execution')
    before, after = row['parameters_before'], row['parameters_after']
    import prepare_teacher_comparison as comparison
    comparison.finite_evidence(receipt)
    comparison.validate_parameter_maps(before, after)
    case = plan['cases'][job['index'] % 94]
    expected_input = dict(case_id=case['id'], prompt_sha256=case['prompt_sha256'],
        image=record(case['image'], nonempty=True), gt=record(case['gt'], nonempty=True))
    require(row['input_binding'] == expected_input and sha(case['prompt'].encode()) == case['prompt_sha256']
            and expected_input['image']['sha256'] == case['image_sha256']
            and expected_input['gt']['sha256'] == case['video_sha256'], 'Actual original image/GT/prompt content drift')
    require(recovery.finite(receipt.get('elapsed_seconds')) and receipt['elapsed_seconds'] > 0, 'Invalid measured case elapsed time')
    if plan['method'] == 'base':
        import run_teacher_base as base
        base.validate_native_stats(receipt['stats'])
        require(receipt['parameters_before'] == before and receipt['parameters_after'] == after
                and receipt['method_context_installed'] is False and receipt['checkpoint'] == 'NONE', 'Native helper/context drift')
    else:
        smoke.validate_stats(plan['method'], receipt['stats']['method'], receipt['stats']['observed'])
    require(row['stats'] == receipt['stats'] and row['video'] == receipt['video'], 'Case stats/media evidence drift')
    require(row['video']['file']['path'] == job['output'], 'Foreign media output')
    recovery.check_content(row['video']['file'])
    require(row['video']['decoded_frames'] == 121 and row['video']['width'] == 1280
            and row['video']['height'] == 704 and row['video']['fps'] == 24 and row['video']['full_decode_xerror'] is True,
            'Full exact 121-frame media proof required')
    if decode:
        require(smoke.validate_video(job['output'], plan['runtime']['tools']) == row['video'], 'Actual full media decode/SHA drift')
    return row


def verify_parent_release(receipt, plan):
    require(receipt.get('schema') == RECEIPT_SCHEMA and receipt.get('execution_id') == recovery.EXECUTION_ID
            and receipt.get('method') == plan['method'], 'Not a supervised execution parent receipt')
    execution = verify_execution(receipt['execution']['path'], receipt['execution']['sha256'])
    recovery.check_content(receipt['execution'])
    digest = execution['plans'][plan['method']]['sha256']
    require(receipt['execution_sha256'] == receipt['execution']['sha256'] and receipt['plan_sha256'] == digest
            and read_bound(execution['plans'][plan['method']]) == plan, 'Parent approved common/plan drift')
    require(type(receipt.get('shard')) is int and 0 <= receipt['shard'] < 16, 'Foreign parent shard')
    allocation = execution['allocations'][plan['method']][receipt['shard']]
    require(all(receipt.get(k) == v for k, v in allocation.items()), 'Foreign parent cap/node/GPU scope')
    folder = shard_folder(plan, receipt['shard'])
    require(read_bound(record(folder / 'parent.json', nonempty=True)) == receipt
            and all(receipt[k]['path'] == str(folder / f'{k}.json') for k in ('launch', 'worker', 'supervised')),
            'Parent/launch/worker/supervised immutable canonical location mismatch')
    owner = next(o for o in execution['owners'][plan['method']] if o['node_rank'] == allocation['node_rank'])
    require(receipt['hostname'] == owner['hostname'] and receipt['owner_pid'] == receipt['owner']['pid']
            and receipt['owner']['start_ticks'] > 0 and receipt['owner_pid'] > 1, 'Unattested parent owner')
    launch, supervised, worker = (read_bound(receipt[k]) for k in ('launch', 'supervised', 'worker'))
    require(launch['schema'] == 'teacher_generation_launch_v1' and launch['owner'] == receipt['owner']
            and launch['allocation'] == allocation and launch['plan_sha256'] == digest
            and launch['execution'] == receipt['execution'] and launch['hostname'] == receipt['hostname']
            and launch['common_execution_sha256'] == execution['common_execution_sha256']
            and launch['score_bundle_sha256'] == execution['score_bundle']['sha256'], 'Launch/common/source chain mismatch')
    require(supervised['schema'] == 'teacher_generation_supervised_v1' and supervised['owner'] == receipt['owner']
            and supervised['launch'] == receipt['launch'] and supervised['session'] == receipt['session'], 'Supervised session chain mismatch')
    session = receipt['session']
    require(session['pid'] == session['session'] and session['ppid'] == receipt['owner_pid'] and session['start_ticks'] > 0,
            'Not a fresh owned session')
    dispatch = read_bound(supervised['dispatch'])
    require(dispatch['launch'] == receipt['launch'] and 'gpu_hold.py' in {os.path.basename(a) for a in dispatch['command']}
            and '--action' in dispatch['command'] and 'exec' in dispatch['command']
            and '--signal=TERM' in dispatch['command'] and f'--kill-after={KILL_SECONDS}s' in dispatch['command']
            and f'{allocation["cap_seconds"] - CLEANUP_SECONDS}s' in dispatch['command'], 'Actual dispatch lacks bounded holderexec chain')
    require(worker['schema'] == 'teacher_generation_worker_v1' and worker['launch'] == receipt['launch']
            and worker['allocation'] == allocation and worker['execution_sha256'] == receipt['execution_sha256']
            and worker['plan_sha256'] == digest and worker['method'] == plan['method']
            and worker['shard'] == receipt['shard'] and worker['owner']['session'] == session['session']
            and any(r['pid'] == session['pid'] and r['start_ticks'] == session['start_ticks'] for r in worker['ancestors'])
            and any(r['pid'] == receipt['owner_pid'] and r['start_ticks'] == receipt['owner']['start_ticks'] for r in worker['ancestors']),
            'Actual worker/parent/session ancestry chain mismatch')
    require(all(recovery.finite(receipt.get(k)) for k in ('start', 'end', 'start_monotonic', 'end_monotonic', 'elapsed_seconds'))
            and receipt['start'] > 0 and receipt['elapsed_seconds'] > 0
            and receipt['start'] == launch['start_epoch'] and receipt['start_monotonic'] == launch['start_monotonic']
            and abs(receipt['end_monotonic'] - receipt['start_monotonic'] - receipt['elapsed_seconds']) < .001
            and abs(receipt['end'] - receipt['start'] - receipt['elapsed_seconds']) < 5
            and launch['start_monotonic'] <= supervised['observed_monotonic'] <= worker['end_monotonic'] <= receipt['end_monotonic'],
            'Durable full-lifecycle monotonic/epoch clock proof mismatch')
    require(receipt['owned_tree_released'] is True and receipt['GPU_released'] is True
            and receipt['final_session_members'] == [] and receipt['final_owned_processes'] == [],
            'Missing owned tree/GPU release evidence')
    for row in receipt['owned_processes']:
        require(type(row['pid']) is int and row['pid'] > 1 and row['start_ticks'] > 0
                and row['pid'] not in (receipt['owner_pid'], launch['boundary']['manager']['pid']),
                'Owned cleanup evidence must never include the parent or independent manager')
        if receipt['hostname'] == os.uname().nodename:
            require(not same_process(row), 'Observed owned descendant remains alive')
    for row in worker['cases']:
        if plan['method'] != 'base':
            require(read_bound(row['claim']) == dict(plan_sha256=digest, job=row['job'],
                pid=worker['owner']['pid'], hostname=receipt['hostname']), 'Persistent helper claim/actual worker mismatch')
        else:
            require('claim' not in row, 'Native base cannot import an intervention claim')
    release = receipt['release_boundary']
    require(release['manager']['pid'] == launch['boundary']['manager']['pid']
            and release['manager']['start_ticks'] == launch['boundary']['manager']['start_ticks']
            and release['gpu_uuid'] == launch['gpu_uuid']
            and release['state']['running'] is True and release['state']['monitor_only'] is True
            and release['state']['query_ok'] is True and release['state']['pid'] == release['manager']['pid']
            and 0 <= receipt['end'] - release['state']['updated_at'] < 5
            and release['state']['gpus'][str(allocation['gpu_index'])]['gpu_uuid'] == launch['gpu_uuid']
            and release['state']['gpus'][str(allocation['gpu_index'])]['holder_pid'] is None
            and release['query']['pids'] == [] and release['query']['command'] == QUERY_COMMAND,
            'Final raw no-hold manager/device release identity required')
    require(worker['start_monotonic'] == launch['start_monotonic'] and worker['start_epoch'] == launch['start_epoch']
            and worker['common_execution_sha256'] == execution['common_execution_sha256']
            and worker['score_bundle_sha256'] == execution['score_bundle']['sha256'], 'Worker common/clock drift')
    queries = receipt['release_queries']
    require(len(queries) >= 2, 'Repeated raw actual GPU release queries required')
    for query in queries[-2:]:
        require(query['command'] == QUERY_COMMAND and query['returncode'] == 0 and query['gpu_uuid'] == launch['gpu_uuid']
                and worker['end_monotonic'] <= query['start_monotonic'] <= query['end_monotonic'] <= receipt['end_monotonic']
                and query['pids'] == [] and query['stderr'] == '', 'Raw GPU release query/clock mismatch')
        raw = list(csv.reader(io.StringIO(query['stdout'])))
        require(all(len(r) == 2 and r[0].strip() != launch['gpu_uuid'] for r in raw if r), 'Raw actual GPU still occupied')
    if receipt['hostname'] == os.uname().nodename:
        require(not same_process(session) and not session_members(session), 'Owned process tree still alive')
    return True


def collect_costs(execution, method, receipts):
    plan, digest = read_bound(execution['plans'][method]), execution['plans'][method]['sha256']
    if method != 'base':
        return recovery.collect_costs(plan, digest, receipts)
    unknown, over, actual, charged = [], [], 0., 0.
    require(all(type(k) is int and 0 <= k < 16 for k in receipts), 'Unexpected collector shard')
    for fixed in execution['allocations'][method]:
        receipt, reported, valid = None, None, False
        try:
            receipt = read_bound(receipts[fixed['shard']])
            if receipt.get('plan_sha256') == digest and recovery.finite(receipt.get('elapsed_seconds')) and receipt['elapsed_seconds'] > 0:
                reported = receipt['elapsed_seconds']
            valid = verify_parent_release(receipt, plan) is True
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if valid:
            actual += reported
            charged += reported
        else:
            unknown.append(fixed['shard'])
            charged += max(fixed['cap_seconds'], reported or 0)
        if (reported or 0) > fixed['cap_seconds']:
            over.append(fixed['shard'])
    return dict(new_actual_seconds=actual if not unknown else None, generation_charged_upper_seconds=charged,
                unknown_shards=unknown, over_cap_shards=over, budget_exceeded=bool(over) or charged > 72000,
                allocation_evidence_complete=not unknown and not over)


def complete_generation(path, digest, method, parent_receipts, output):
    execution = verify_execution(path, digest)
    require(set(parent_receipts) == set(range(16)), 'All16 parent-owned receipts required')
    plan = read_bound(execution['plans'][method])
    costs = collect_costs(execution, method, parent_receipts)
    require(costs['allocation_evidence_complete'] is True and costs['budget_exceeded'] is False, 'Budget/release completion blocked')
    cases = []
    for shard in range(16):
        parent = read_bound(parent_receipts[shard])
        verify_parent_release(parent, plan)
        require(parent['success'] is True and parent['exit_code'] == 0 and parent['error'] is None
                and parent['interrupted'] is None and parent['timed_out'] is False, 'Parent release is not generation PASS')
        worker = read_bound(parent['worker'])
        require(worker['success'] is True and [r['job'] for r in worker['cases']] == plan['jobs'][shard::16], 'Fixed shard job set incomplete')
        cases.extend(verify_case(r, plan, execution['plans'][method]['sha256']) for r in worker['cases'])
    cases.sort(key=lambda r: r['job']['index'])
    require(len(cases) == 188 and [r['job'] for r in cases] == plan['jobs'], 'Exactly all94x2seed generation required')
    value = dict(schema='teacher_generation_completion_v1', private=True, method=method,
        execution=record(path, nonempty=True), plan=execution['plans'][method], plan_sha256=execution['plans'][method]['sha256'],
        common_execution_sha256=execution['common_execution_sha256'], score_bundle_sha256=execution['score_bundle']['sha256'],
        parent_receipts=[parent_receipts[s] for s in range(16)], cases=cases, costs=costs,
        generation_complete=True, GPU_used=True, quality_pass=False, sota_claim=False)
    dev.exclusive(output, value)
    return Path(output), sha(Path(output).read_bytes())


def verify_generation_completion(path, approved_sha, *, common_execution_sha256, score_bundle_sha256, method):
    value = read_bound(dict(record(path, nonempty=True), sha256=approved_sha))
    require(value.get('schema') == 'teacher_generation_completion_v1' and value['method'] == method
            and value['generation_complete'] is True and value['quality_pass'] is False and value['sota_claim'] is False
            and value['common_execution_sha256'] == common_execution_sha256
            and value['score_bundle_sha256'] == score_bundle_sha256, 'Approved common scoring completion mismatch')
    execution = verify_execution(value['execution']['path'], value['execution']['sha256'])
    require(value['plan'] == execution['plans'][method] and value['plan_sha256'] == execution['plans'][method]['sha256'], 'Completion fresh plan mismatch')
    plan = read_bound(value['plan'])
    require(len(value['parent_receipts']) == 16 and len(value['cases']) == 188, 'Helper/partial evidence is not all16x188 completion')
    receipts = {}
    cases = []
    for row in value['parent_receipts']:
        parent = read_bound(row)
        require(parent['shard'] not in receipts and parent['success'] is True and parent['exit_code'] == 0, 'Duplicate/failed parent receipt')
        verify_parent_release(parent, plan)
        receipts[parent['shard']] = row
        worker = read_bound(parent['worker'])
        require(worker['success'] is True and [r['job'] for r in worker['cases']] == plan['jobs'][parent['shard']::16], 'Worker fixed coverage drift')
        cases.extend(worker['cases'])
    cases.sort(key=lambda r: r['job']['index'])
    require(value['cases'] == cases and [r['job'] for r in cases] == plan['jobs'], 'Completion all94x2seed coverage drift')
    costs = collect_costs(execution, method, receipts)
    require(value['costs'] == costs and costs['allocation_evidence_complete'] is True and costs['budget_exceeded'] is False, 'Completion lifecycle budget mismatch')
    for row in cases:
        verify_case(row, plan, value['plan_sha256'])
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--prepare', action='store_true')
    actions.add_argument('--prepare-scoring-owner', action='store_true')
    actions.add_argument('--complete', action='store_true')
    actions.add_argument('--verify', action='store_true')
    actions.add_argument('--reserve-local', action='store_true')
    actions.add_argument('--run', action='store_true')
    actions.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--execution', type=Path)
    parser.add_argument('--approved-execution-sha256')
    for name in ('comparison', 'authorization', 'owner-proofs', 'retirement-evidence', 'parent-receipts', 'score-bundle', 'scoring-owner-proof'):
        parser.add_argument('--' + name, type=Path)
        parser.add_argument('--approved-' + name + '-sha256')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--shard', type=int)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--gpu-authorized', action='store_true')
    parser.add_argument('--state-dir', default='/dockerdata/wm_logs/gpu_hold')
    parser.add_argument('--launch', type=Path)
    args = parser.parse_args(argv)
    os.umask(0o077)
    if args.prepare_scoring_owner:
        require(not args.gpu_authorized and args.score_bundle and args.approved_score_bundle_sha256 and args.output,
                'CPU scoring owner preparation needs approved actual bundle and new output')
        path, digest = prepare_scoring_owner_proof(args.score_bundle, args.approved_score_bundle_sha256, args.output)
        print(json.dumps(dict(scoring_owner_proof=str(path), sha256=digest, GPU_used=False)))
        return
    require(args.execution, 'Explicit execution sidecar path required')
    if args.prepare:
        require(not args.gpu_authorized, 'Preparation is CPU-only; GPU CLI flags forbidden')
        required = ('comparison', 'authorization', 'owner_proofs', 'retirement_evidence')
        for name in required:
            require(getattr(args, name) and getattr(args, 'approved_' + name + '_sha256'), 'Explicit approved prepare evidence required')
        owners = read_bound(dict(record(args.owner_proofs, nonempty=True), sha256=args.approved_owner_proofs_sha256))
        retirement = read_bound(dict(record(args.retirement_evidence, nonempty=True), sha256=args.approved_retirement_evidence_sha256))
        path, digest = prepare_execution(args.comparison, args.approved_comparison_sha256, args.authorization,
            args.approved_authorization_sha256, owners, retirement, args.execution,
            scoring_owner_proof=args.scoring_owner_proof, approved_scoring_owner_sha=args.approved_scoring_owner_proof_sha256)
        print(json.dumps(dict(execution=str(path), sha256=digest, GPU_used=False)))
        return
    require(args.approved_execution_sha256, 'Explicit approved execution SHA required')
    if args.complete:
        require(not args.gpu_authorized and args.parent_receipts and args.approved_parent_receipts_sha256
                and args.method and args.output, 'CPU completion requires approved all16 receipts and output')
        rows = read_bound(dict(record(args.parent_receipts, nonempty=True), sha256=args.approved_parent_receipts_sha256))
        receipts = {int(k): v for k, v in rows.items()}
        path, digest = complete_generation(args.execution, args.approved_execution_sha256, args.method, receipts, args.output)
        print(json.dumps(dict(completion=str(path), sha256=digest)))
    elif args.worker:
        execute_worker(args.execution, args.approved_execution_sha256, args.method, args.shard, args.rank, args.launch, args.state_dir)
    elif args.run:
        path, digest = supervise(args.execution, args.approved_execution_sha256, args.method, args.shard, args.rank,
                                 gpu_authorized=args.gpu_authorized, state_dir=args.state_dir)
        print(json.dumps(dict(parent=str(path), sha256=digest)))
    else:
        require(not args.gpu_authorized, 'CPU actions refuse GPU authorization flag')
        execution = verify_execution(args.execution, args.approved_execution_sha256)
        if args.reserve_local:
            path = reserve_local(execution, args.approved_execution_sha256, args.method, args.rank)
            print(json.dumps(dict(ledger=str(path), GPU_used=False)))
        else:
            print(json.dumps(dict(execution_verified=True, GPU_used=False)))


if __name__ == '__main__':
    main()
