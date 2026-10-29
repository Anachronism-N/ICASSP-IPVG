#!/usr/bin/env python3
"""CPU-only node-local recovery protocol; no GPU execution authorization."""
import argparse
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import tempfile
import time

import run_teacher_devset as dev

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = Path('/dockerdata/wm_tmp')
EXECUTION_ID = 'recovery_a1'
SCHEMA = 'teacher_local_recovery_plan_v1'
CAPS = {'alg': 5000, 'dymos': 4800}
RANKS = {'alg': {1: 0, 2: 8}, 'dymos': {0: 0, 3: 8}}
PRIOR = {
    'alg': ('teacher_alg_race_audit_20261026_1408_0f6ab742.json',
            '59657153babe30e7d90198e5e6aa69070a86bcf4e5d9fd2426f34e54f3f648d9', 5926.31142393779),
    'dymos': ('teacher_dymos_race_audit_20261026_1350_8c43f913.json',
              'b764c0a4885fc7c705f6f72ebb0190c23b5dd9a62f79d8de76e63056563840fc', 9559.88809741661),
}
require, encoded, sha, record = dev.require, dev.encoded, dev.sha, dev.record


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def check_content(expected):
    empty_source = expected.get('bytes') == 0 and Path(expected['path']).suffix == '.py'
    actual = record(expected['path'], nonempty=not empty_source)
    require(all(actual[k] == expected[k] for k in ('path', 'bytes', 'sha256')), 'Evidence content drift')
    return actual


def local_path(path, local_root=LOCAL_ROOT):
    path, local_root = Path(path).absolute(), Path(local_root).absolute()
    require(local_root.resolve() == local_root and path.resolve() == path
            and path.is_relative_to(local_root), 'Explicit node-local root required; symlinks forbidden')
    return path


def durable_write(path, value):
    path = Path(path).absolute()
    require(path.resolve() == path, 'Mutable state symlink forbidden')
    payload = encoded(value)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


@contextlib.contextmanager
def local_locked(path, local_root=LOCAL_ROOT, timeout_seconds=30):
    path = local_path(path, local_root)
    require(finite(timeout_seconds) and timeout_seconds >= 0, 'Invalid lock timeout')
    lock = path.with_name(path.name + '.lock')
    fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    acquired = False
    try:
        until = time.monotonic() + timeout_seconds
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if time.monotonic() >= until:
                    raise TimeoutError('Local lock timeout; never remove or steal lock inode')
                time.sleep(0.01)
        identity, current = os.fstat(fd), lock.stat()
        require((identity.st_dev, identity.st_ino) == (current.st_dev, current.st_ino), 'Lock inode changed')
        yield
    finally:
        if acquired:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def plan_folder(root, method):
    return Path(root) / f'eval/teacher_{method}_devset_20261026_{EXECUTION_ID}'


def allocations(method):
    return [dict(shard=offset + gpu, node_rank=rank, gpu_index=gpu, allocated_gpus=1,
                 cap_seconds=CAPS[method])
            for rank, offset in RANKS[method].items() for gpu in range(8)]


def validate_plan(plan):
    method = plan['method']
    require(method in CAPS and plan['schema'] == SCHEMA and plan['execution_id'] == EXECUTION_ID,
            'Wrong fixed recovery method/execution')
    require(plan['GPU_authorized'] is False and plan['GPU_used'] is False
            and plan['status'] == 'BLOCKED' and plan['old_reservations_retired'] is False,
            'CPU plan is not authorization or historical reservation retirement')
    require(plan['nshards'] == 16 and plan['allocations'] == allocations(method), 'Static quota mapping drift')
    prior = plan['prior_generation']
    require(prior['actual_seconds'] is None and prior['upper_seconds'] == PRIOR[method][2]
            and prior['audit']['sha256'] == PRIOR[method][1], 'Unknown prior cannot be zeroed or reduced')
    require(plan['phase_seconds'] == dev.PHASE_SECONDS and plan['total_seconds'] == dev.TOTAL_SECONDS,
            'Independent phase budgets cannot move')
    require(prior['upper_seconds'] + sum(a['cap_seconds'] for a in plan['allocations'])
            <= plan['phase_seconds']['generation'], 'Old cost plus new caps exceeds 24 GPUh')
    require(plan['video_count'] == 188 and plan['seeds'] == [0, 1]
            and plan['cases_file']['sha256'] == dev.CASES_SHA256
            and plan['call'] == dev.SPEC and plan['parameters'] == dev.PARAMS[method], 'Frozen inputs/method drift')
    require(plan['historical_base_reuse_allowed'] is False, 'Historical base reuse remains unproven')
    require(finite(plan['smoke_charged_seconds']) and 0 < plan['smoke_charged_seconds']
            <= plan['phase_seconds']['smoke'], 'Missing independent smoke cost')
    if plan.get('common_contract_frozen') is True:
        root = Path(plan['root'])
        require(plan['checkpoint'] == 'NONE' and plan['active_steps'] == 10 and plan['fps'] == 24
                and plan['method_context_installed'] is True and plan['shard_cap_seconds'] == [CAPS[method]] * 16
                and plan['quality_pass'] is False and plan['sota_claim'] is False, 'Candidate content contract drift')
        require(plan['output_root'] == str(plan_folder(root, method)), 'Candidate output must be the new recovery root')
        require(len(plan['cases']) == 94 and len({c['id'] for c in plan['cases']}) == 94, 'Wrong frozen candidate cases')
        expected_jobs = [dict(j, output=str(plan_folder(root, method) / f"seed{j['seed']}/videos/{method}/{j['case_id']}.mp4"))
                         for j in dev.jobs_for(plan['cases'], root, method)]
        require(plan['jobs'] == expected_jobs, 'Candidate seed/job mapping drift')
        require(len(plan['source_files']) == len(sources(root))
                and {r['path'] for r in plan['source_files']} == {str(p) for p in sources(root)}, 'Candidate source set drift')
    return plan


def sources(root):
    return sorted(set(dev.sources(Path(root))) | {Path(root) / name for name in (
        'scripts/prepare_teacher_recovery.py', 'tests/test_teacher_recovery.py',
        'scripts/run_teacher_generation.py', 'tests/test_teacher_generation.py',
        'scripts/verify_teacher_history.py', 'tests/test_verify_teacher_history.py')})


def freeze_common_contract(value, node_preflight, approved_node_sha, score_bundle, approved_score_sha):
    root, method = Path(value['root']), value['method']
    raw, cases, inputs, manifest = dev.frozen_cases(root)
    pf_record = record(node_preflight, nonempty=True)
    require(pf_record['sha256'] == approved_node_sha, 'Approved node preflight SHA mismatch')
    pf = json.loads(Path(node_preflight).read_bytes())
    models = dev.smoke.models_from_preflight(pf)
    import score_teacher_devset as scoring
    scoring.verify_bundle(score_bundle, approved_score_sha, verify_content=True)
    score_record = record(score_bundle, nonempty=True)
    require(score_record['sha256'] == approved_score_sha, 'Approved score bundle content drift')
    negative = dev.smoke.generator_negative(root)
    folder = plan_folder(root, method)
    jobs = [dict(job, output=str(folder / f"seed{job['seed']}/videos/{method}/{job['case_id']}.mp4"))
            for job in dev.jobs_for(cases, root, method)]
    value.update(common_contract_frozen=True, cases=cases, inputs=inputs, cases_file=manifest,
                 models=models, model_content_sha256=pf['model_content_sha256'],
                 runtime=dev.smoke.runtime_records(), node_preflight=pf_record, source_files=[record(p) for p in sources(root)],
                 negative_prompt=negative, negative_sha256=sha(negative.encode()), score_bundle=score_record,
                 jobs=jobs, output_root=str(folder), checkpoint='NONE', active_steps=10, fps=24,
                 method_context_installed=True, shard_cap_seconds=[CAPS[method]] * 16,
                 quality_pass=False, sota_claim=False)
    return raw


def prepare(method, root=ROOT, *, node_preflight=None, approved_node_sha=None,
            score_bundle=None, approved_score_sha=None):
    require(method in CAPS, 'One fixed recovery method required')
    root = Path(root).resolve()
    audit_path = root / 'eval' / PRIOR[method][0]
    audit_record = record(audit_path, nonempty=True)
    require(audit_record['sha256'] == PRIOR[method][1], 'Approved historical audit SHA mismatch')
    audit = json.loads(audit_path.read_bytes())
    require(audit['schema'] == 'teacher_devset_readonly_race_audit_v1'
            and audit['unknown_shards'] == [] and audit['originals_modified'] is False, 'Unbounded historical audit')
    shards = audit['shards']
    require(len(shards) == 16 and {s['shard'] for s in shards} == set(range(16))
            and all(finite(s['cost_upper_bound_seconds']) and s['cost_upper_bound_seconds'] >= 0 for s in shards)
            and math.isclose(sum(s['cost_upper_bound_seconds'] for s in shards), PRIOR[method][2], abs_tol=1e-8),
            'Account for all sixteen old shards, not just known launches')
    check_content(audit['plan_record'])
    check_content(audit['ledger_record'])
    old = json.loads(Path(audit['plan_record']['path']).read_bytes())
    require(old['method'] == method and old['phase_seconds'] == dev.PHASE_SECONDS
            and old['cases_file']['sha256'] == dev.CASES_SHA256
            and old['seeds'] == [0, 1] and old['video_count'] == 188, 'Wrong historical plan')
    require(record(root / dev.CASE_ROOT / 'cases.jsonl', nonempty=True)['sha256'] == dev.CASES_SHA256,
            'Frozen case bytes drift')
    value = dict(schema=SCHEMA, private=True, method=method, execution_id=EXECUTION_ID,
                 root=str(root), status='BLOCKED', GPU_authorized=False, GPU_used=False,
                 old_reservations_retired=False, historical_base_reuse_allowed=False,
                 prior_generation=dict(actual_seconds=None, upper_seconds=PRIOR[method][2], audit=audit_record,
                                       old_plan=audit['plan_record'], old_ledger=audit['ledger_record']),
                 smoke_charged_seconds=old['admission']['charged_or_reserved_seconds'],
                 phase_seconds=dict(dev.PHASE_SECONDS), total_seconds=dev.TOTAL_SECONDS,
                 nshards=16, video_count=188, seeds=[0, 1], cases_file=old['cases_file'],
                 call=dict(dev.SPEC), parameters=dict(dev.PARAMS[method]), source=record(__file__, nonempty=True),
                 allocations=allocations(method),
                 required_gates=['Old complete termination/log binding and reservation retirement amendment',
                                 'Common generation/scoring bundle and current node content preflights',
                                 'Real two-node CPU evidence and independent tests',
                                 'Native base throughput and bounded scoring integration',
                                 'No-hold GPU manager and full-lifecycle parent supervision',
                                 'Explicit user GPU resume authorization'])
    value['common_contract_frozen'] = False
    raw = None
    if any(x is not None for x in (node_preflight, approved_node_sha, score_bundle, approved_score_sha)):
        require(all(x is not None for x in (node_preflight, approved_node_sha, score_bundle, approved_score_sha)),
                'Full approved generation and scoring inputs required together')
        raw = freeze_common_contract(value, node_preflight, approved_node_sha, score_bundle, approved_score_sha)
    validate_plan(value)
    folder = plan_folder(root, method)
    require(not folder.exists(), 'Recovery execution exists; never overwrite or retry')
    folder.mkdir(mode=0o700)
    if raw is not None:
        for seed in (0, 1):
            seed_root = folder / f'seed{seed}'
            (seed_root / 'videos' / method).mkdir(mode=0o700, parents=True)
            fd = os.open(seed_root / 'cases.jsonl', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'wb') as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
    path = folder / 'plan.blocked.json'
    dev.exclusive(path, value)
    return path, sha(path.read_bytes())


def verify(path, digest):
    path = Path(path).absolute()
    require(path.resolve() == path and sha(path.read_bytes()) == digest, 'Approved recovery SHA mismatch')
    plan = validate_plan(json.loads(path.read_bytes()))
    require(path == plan_folder(plan['root'], plan['method']) / 'plan.blocked.json', 'Wrong recovery location')
    for r in (plan['source'], plan['cases_file'], plan['prior_generation']['audit'],
              plan['prior_generation']['old_plan'], plan['prior_generation']['old_ledger']):
        check_content(r)
    if plan.get('common_contract_frozen') is True:
        for r in plan['source_files'] + plan['models'] + plan['runtime']['files']:
            check_content(r)
        raw, cases, inputs, manifest = dev.frozen_cases(Path(plan['root']))
        require((cases, inputs, manifest) == (plan['cases'], plan['inputs'], plan['cases_file']), 'Current candidate input drift')
        for seed in (0, 1):
            require((path.parent / f'seed{seed}/cases.jsonl').read_bytes() == raw, 'Candidate seed metadata drift')
        require(dev.smoke.generator_negative(Path(plan['root'])) == plan['negative_prompt']
                and sha(plan['negative_prompt'].encode()) == plan['negative_sha256'], 'Candidate negative prompt drift')
        check_content(plan['node_preflight'])
        models = dev.smoke.models_from_preflight(json.loads(Path(plan['node_preflight']['path']).read_bytes()))
        require([{k: r[k] for k in ('asset', 'bytes', 'sha256')} for r in models]
                == [{k: r[k] for k in ('asset', 'bytes', 'sha256')} for r in plan['models']], 'Candidate model tuple drift')
        import score_teacher_devset as scoring
        scoring.verify_bundle(plan['score_bundle']['path'], plan['score_bundle']['sha256'], verify_content=True)
    return plan


def ledger_path(plan, local_root=LOCAL_ROOT):
    return Path(local_root) / f"teacher_{plan['method']}_devset_{EXECUTION_ID}" / 'allocations.json'


def validate_local(value, plan, digest, rank):
    validate_plan(plan)
    require(rank in RANKS[plan['method']] and value['schema'] == 'teacher_local_recovery_ledger_v1'
            and value['execution_id'] == EXECUTION_ID and value['method'] == plan['method']
            and value['plan_sha256'] == digest and value['node_rank'] == rank
            and value['hostname'] == os.uname().nodename, 'Wrong local owner/plan')
    expected = [a for a in plan['allocations'] if a['node_rank'] == rank]
    require(len(value['allocations']) == 8, 'Only own eight shards are mutable')
    for actual, fixed in zip(value['allocations'], expected):
        require(all(actual.get(k) == v for k, v in fixed.items()) and actual['status'] == 'reserved'
                and 'start' not in actual and actual['GPU_used'] is False, 'Cannot change peer quotas or GPU state in CPU protocol')
    require(type(value['cpu_revision']) is int and value['cpu_revision'] >= 0, 'Invalid CPU revision')
    return value


def reserve_local(plan, digest, rank, local_root=LOCAL_ROOT):
    validate_plan(plan)
    require(sha(encoded(plan)) == digest and rank in RANKS[plan['method']], 'Wrong plan digest/node rank')
    path = ledger_path(plan, local_root)
    require(not path.parent.exists(), 'Local execution exists; never reset')
    local_path(path, local_root)
    path.parent.mkdir(mode=0o700)
    value = dict(schema='teacher_local_recovery_ledger_v1', method=plan['method'], execution_id=EXECUTION_ID,
                 plan_sha256=digest, node_rank=rank, hostname=os.uname().nodename, cpu_revision=0,
                 allocations=[dict(a, status='reserved', GPU_used=False)
                              for a in plan['allocations'] if a['node_rank'] == rank])
    validate_local(value, plan, digest, rank)
    with local_locked(path, local_root):
        dev.exclusive(path, value)
    return path


@contextlib.contextmanager
def local_transaction(path, plan, digest, rank, local_root=LOCAL_ROOT):
    require(sha(encoded(plan)) == digest, 'Transaction plan digest drift')
    require(Path(path).absolute() == ledger_path(plan, local_root).absolute(), 'Unapproved ledger path')
    with local_locked(path, local_root):
        value = validate_local(json.loads(Path(path).read_bytes()), plan, digest, rank)
        yield value
        validate_local(value, plan, digest, rank)
        durable_write(path, value)


def verify_parent_release(receipt, plan):
    import run_teacher_generation as execution
    return execution.verify_parent_release(receipt, plan)


def collect_costs(plan, digest, receipts):
    """Absent release evidence retains caps, never implies measured cost zero."""
    validate_plan(plan)
    require(sha(encoded(plan)) == digest, 'Collector plan digest drift')
    require(type(receipts) is dict and all(type(s) is int and 0 <= s < 16 for s in receipts), 'Unexpected receipts')
    actual, charged, unknown, complete, over = 0.0, 0.0, [], [], []
    for fixed in plan['allocations']:
        shard, receipt = fixed['shard'], None
        if shard in receipts:
            try:
                check_content(receipts[shard])
                receipt = json.loads(Path(receipts[shard]['path']).read_bytes())
            except (OSError, ValueError, UnicodeError):
                receipt = None
        bound = bool(isinstance(receipt, dict) and receipt.get('schema') in ('teacher_recovery_parent_receipt_v1', 'teacher_generation_parent_receipt_v1')
                     and receipt.get('execution_id') == EXECUTION_ID and receipt.get('method') == plan['method']
                     and receipt.get('plan_sha256') == digest and all(receipt.get(k) == v for k, v in fixed.items())
                     and type(receipt.get('owner_pid')) is int and receipt['owner_pid'] > 1
                     and isinstance(receipt.get('hostname'), str) and receipt['hostname']
                     and receipt.get('owned_tree_released') is True and receipt.get('GPU_released') is True
                     and all(finite(receipt.get(k)) for k in ('start', 'end', 'elapsed_seconds'))
                     and receipt['elapsed_seconds'] > 0 and receipt['start'] > 0
                     and abs(receipt['end'] - receipt['start'] - receipt['elapsed_seconds']) < 5)
        reported = None
        if (isinstance(receipt, dict) and receipt.get('plan_sha256') == digest
                and receipt.get('execution_id') == EXECUTION_ID and receipt.get('method') == plan['method']
                and all(receipt.get(k) == v for k, v in fixed.items())
                and finite(receipt.get('elapsed_seconds')) and receipt['elapsed_seconds'] > 0):
            reported = receipt['elapsed_seconds']
        if bound:
            try:
                require(verify_parent_release(receipt, plan) is True, 'Parent release evidence remains unverified')
            except (ValueError, OSError, KeyError, TypeError, IndexError):
                bound = False
        if not bound:
            unknown.append(shard)
            charged += max(fixed['cap_seconds'], reported or 0)
            if reported is not None and reported > fixed['cap_seconds']:
                over.append(shard)
        else:
            elapsed = receipt['elapsed_seconds']
            actual += elapsed
            charged += elapsed
            complete.append(shard)
            if elapsed > fixed['cap_seconds']:
                over.append(shard)
    total = plan['prior_generation']['upper_seconds'] + charged
    return dict(schema='teacher_recovery_cost_summary_v1', execution_id=EXECUTION_ID, method=plan['method'],
                plan_sha256=digest, old_actual_seconds=None, old_generation_upper_seconds=plan['prior_generation']['upper_seconds'],
                new_actual_known_seconds=actual, new_actual_seconds=actual if not unknown else None,
                generation_charged_upper_seconds=total, unknown_shards=unknown, over_cap_shards=over,
                budget_exceeded=bool(over) or total > plan['phase_seconds']['generation'],
                allocation_evidence_complete=len(complete) == 16 and not over,
                generation_complete=False, quality_pass=False, sota_claim=False,
                caveat='Caps are reservations, not measured consumption; old upper is charged exactly once')


def cpu_probe(path, local_root, increments=10, worker=0):
    """Exercise real local lock/write primitives, without CUDA or model imports."""
    require(type(increments) is int and 0 < increments <= 1000 and type(worker) is int and worker >= 0, 'Invalid CPU probe')
    path = local_path(path, local_root)
    waits = []
    for index in range(increments):
        start = time.monotonic()
        with local_locked(path, local_root):
            waits.append(time.monotonic() - start)
            value = json.loads(path.read_bytes())
            require(value['GPU_used'] is False and type(value['counter']) is int, 'Wrong isolated CPU probe')
            value['entries'].append([worker, index])
            value['counter'] += 1
            durable_write(path, value)
    return dict(hostname=os.uname().nodename, worker=worker, increments=increments, waits=waits, GPU_used=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=CAPS)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--approved-plan-sha256')
    parser.add_argument('--reserve-local', action='store_true')
    parser.add_argument('--rank', type=int)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--node-preflight', type=Path)
    parser.add_argument('--approved-node-preflight-sha256')
    parser.add_argument('--score-bundle', type=Path)
    parser.add_argument('--approved-score-bundle-sha256')
    parser.add_argument('--execution', type=Path)
    parser.add_argument('--approved-execution-sha256')
    parser.add_argument('--gpu-authorized', action='store_true')
    parser.add_argument('--shard', type=int)
    parser.add_argument('--state-dir', default='/dockerdata/wm_logs/gpu_hold')
    args = parser.parse_args(argv)
    os.umask(0o077)
    if args.run:
        require(args.execution and args.approved_execution_sha256 and args.plan and args.approved_plan_sha256
                and args.rank is not None and args.shard is not None and args.gpu_authorized,
                'GPU recovery blocked: explicit verified execution sidecar/plan/GPU authorization required')
        import run_teacher_generation as generation
        plan_row = record(args.plan, nonempty=True)
        require(plan_row['sha256'] == args.approved_plan_sha256, 'Approved recovery run plan SHA mismatch')
        plan = validate_plan(generation.read_bound(plan_row))
        execution = generation.verify_execution(args.execution, args.approved_execution_sha256)
        require(execution['plans'][plan['method']] == record(args.plan, nonempty=True),
                'Execution does not bind this immutable blocked CPU plan')
        path, digest = generation.supervise(args.execution, args.approved_execution_sha256, plan['method'],
            args.shard, args.rank, gpu_authorized=args.gpu_authorized, state_dir=args.state_dir)
        print(json.dumps(dict(parent=str(path), sha256=digest)))
        return
    require(not args.gpu_authorized and not args.execution and not args.approved_execution_sha256,
            'CPU protocol does not accept execution/GPU flags without --run')
    if args.plan:
        require(args.approved_plan_sha256, 'Approved immutable plan SHA required')
        plan = verify(args.plan, args.approved_plan_sha256)
        if args.reserve_local:
            require(args.rank is not None, 'Explicit local rank required')
            path = reserve_local(plan, args.approved_plan_sha256, args.rank)
            print(json.dumps(dict(ledger=str(path), GPU_used=False, status='BLOCKED')))
        else:
            print(json.dumps(dict(method=plan['method'], status='BLOCKED', GPU_used=False)))
    else:
        require(args.method and not args.reserve_local, 'CPU prepare requires a fixed method')
        path, digest = prepare(args.method, node_preflight=args.node_preflight,
                               approved_node_sha=args.approved_node_preflight_sha256,
                               score_bundle=args.score_bundle, approved_score_sha=args.approved_score_bundle_sha256)
        print(json.dumps(dict(plan=str(path), sha256=digest, status='BLOCKED', GPU_used=False)))


if __name__ == '__main__':
    main()
