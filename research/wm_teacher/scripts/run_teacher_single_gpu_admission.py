#!/usr/bin/env python3
"""One-shot synthetic reference4/prefix2/fresh-resume2; never formal training."""
import argparse
import csv
import io
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import threading

ROOT = Path(__file__).resolve().parents[1]
LOCAL = Path('/dockerdata/wm_tmp')
AUTH = ROOT / 'eval/diag/teacher_training_admission_authorization_20261101.json'
MODELS = Path('/dockerdata/models')
FRAMEWORK = ROOT / 'DiffSynth-Studio'
SCHEMA, PHASE = 'teacher_single_gpu_deterministic_recovery_v2', 'single_gpu_training_resume'
OUTPUT_ROOT = LOCAL / 'teacher_single_gpu_deterministic_recovery_20261102'
PRIOR_ROOT = LOCAL / 'teacher_single_gpu_admission_20261102_run01'
PRIOR_AUDIT = ROOT / 'eval/diag/teacher_single_gpu_actual_failure_audit_20261102.json'
PRIOR_AUDIT_SHA256 = 'f3f2951345f2df2c9cc00cfe941dfc6219564d25b41b6526bfd4f1c7a85e64d0'
DET_AUTH = ROOT / 'eval/diag/teacher_deterministic_recovery_authorization_20261102.json'
CAP, CLEANUP, TAIL, CHILD_CAP = 7200, 120, 120, 2000
GIT_ROOT = Path('/dockerdata/icaspp_ipvg_publication_20261018')
BRANCH = 'wm-teacher-experiments-20261018'
SNAPSHOT_PREFIX = 'research/wm_teacher'
ORIGIN = 'https://github.com/Anachronism-N/ICASSP-IPVG.git'
STATE_DIR = Path('/dockerdata/wm_logs/teacher_single_gpu_admission_20261102')
SEGMENTS = (('reference', 4, 4), ('prefix', 2, 2), ('resume', 2, 4))
SOURCE_NAMES = ('scripts/run_teacher_single_gpu_admission.py', 'tests/test_teacher_single_gpu_admission.py',
 'scripts/train_e3_lora.py', 'tests/test_e3_train.py', 'scripts/prepare_teacher_training_admission.py',
 'scripts/run_teacher_training_smoke.py', 'scripts/run_teacher_mechanism_probe.py', 'scripts/merge_e3_lora.py',
 'scripts/run_teacher_generation.py', 'scripts/prepare_teacher_recovery.py', 'scripts/run_teacher_devset.py',
 'scripts/run_teacher_inference_smoke.py', 'scripts/measure_teacher_throughput.py',
 'scripts/prepare_teacher_comparison.py', 'scripts/run_teacher_base.py', 'scripts/verify_teacher_history.py',
 'scripts/gpu_hold.py')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def deadline(end):
    if end is not None and time.monotonic() >= end:
        raise TimeoutError('Outer work window exhausted; cleanup reserved')


def asset_record(path, end=None, expected=None, allow_empty=False):
    import run_teacher_mechanism_probe as m1
    p = Path(path).absolute()
    require(p.is_file() and p.resolve() == p, 'Actual nonsymlink file required: ' + str(p))
    before = p.stat()
    row = m1.file_record(p, expected=expected, deadline=end, allow_empty=allow_empty)
    after = p.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), 'File changed while hashing')
    return row


def record(path, end=None, allow_empty=False):
    return {k: v for k, v in asset_record(path, end, allow_empty=allow_empty).items() if k in ('path', 'bytes', 'sha256')}


def read(row, end=None):
    _, _, process = helpers()
    require(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'} and type(row['bytes']) is int
            and 0 < row['bytes'] <= 16 * 1024 * 1024, 'Small exact JSON record required')
    deadline(end)
    value = process.read_bound(row)
    deadline(end)
    return value

def write_new(path, value):
    import run_teacher_mechanism_probe as m1
    m1.atomic_json(path, value)
    return record(path)


def helpers():
    import train_e3_lora as tr
    import prepare_teacher_training_admission as admission
    import run_teacher_generation as process
    return tr, admission, process


def deterministic_authorization(end=None):
    tr, _, _ = helpers()
    row = record(DET_AUTH, end)
    require(row == tr.deterministic_authorization(), 'Additional authorization raw SHA/path differs')
    value = read(row, end)
    expected_scope = dict(new_GPUh_cap=2, allocated_cards=1, wall_cap_seconds=CAP, one_attempt=True,
        reference_updates=4, prefix_updates=2, fresh_process_resume_updates=2, physical_optimizer_updates=8,
        selected_attention_backend='same native FA3; explicit deterministic backward',
        backend_capability_probe='First reference update is the actual capability gate, counted within four reference updates; no extra uncharged probe',
        no_silent_backend_fallback=True)
    require(value.get('schema') == 'teacher_deterministic_recovery_authorization_v1'
            and encoded(value.get('assistant_bounded_scope')) == encoded(expected_scope)
            and type(value.get('prior_failed_technical_GPUh_upper')) is int
            and value['prior_failed_technical_GPUh_upper'] == 2
            and value.get('prior_costs_preserved') is True and value.get('new_scope_additional_not_borrowed') is True
            and encoded(value.get('original_training_stage_caps_GPUh')) == encoded(dict(
                label_rebuild_and_cache=24, single_gpu_training_resume=2, multi_node_technical_validation=6))
            and value.get('requires_new_source_runtime_publication_and_exact_resume_gate') is True
            and all(value.get(k) is False for k in ('labels_budget_borrowing_allowed','formal_training_allowed',
                'AR_allowed','student_distillation_allowed','automatic_retry_allowed')),
            'Additional deterministic authorization scope/schema/types differs')
    return row


def prior_failure(end=None):
    audit_row = record(PRIOR_AUDIT, end)
    require(audit_row['sha256'] == PRIOR_AUDIT_SHA256, 'Original failure audit raw SHA differs')
    audit = read(audit_row, end)
    require(audit.get('schema') == 'teacher_single_gpu_actual_failure_audit_v1'
            and audit.get('stage') == PHASE and type(audit.get('charged_upper_GPUh')) is int
            and audit['charged_upper_GPUh'] == 2 and type(audit.get('physical_updates')) is int
            and audit['physical_updates'] == 8 and audit.get('closure_present') is False
            and audit.get('terminal_success') is False and audit.get('release_verified_at_audit') is True,
            'Original failed allocation audit/fee/updates differs')
    for key, name in (('parent', 'parent.json'), ('failure', 'cost.failure.json'), ('provisional_cost', 'cost.json')):
        require(audit[key]['path'] == str(PRIOR_ROOT / name)
                and record(PRIOR_ROOT / name, end) == audit[key], 'Original failure evidence bytes differ')
    parent, failure, provisional = (read(audit[k], end) for k in ('parent', 'failure', 'provisional_cost'))
    require(parent.get('schema') == 'teacher_single_gpu_admission_v1' and parent.get('phase') == PHASE
            and parent.get('execution_success') is False and parent.get('success') is False
            and parent.get('released') is True and parent.get('comparison') is None
            and parent.get('error') == 'ValueError: Exact resumed trajectory differs: adapters'
            and parent.get('plan') == audit['plan'] and failure.get('plan') == audit['plan']
            and failure.get('schema') == 'teacher_single_gpu_cost_failure_v1' and failure.get('phase') == PHASE
            and failure.get('terminal') is True and failure.get('success') is False
            and failure.get('released_observed') is True and failure.get('parent_receipt') == audit['parent']
            and provisional.get('parent_receipt') == audit['parent'] and provisional.get('success') is False,
            'Original failed parent/cost binding differs')
    require(all(type(v.get('charged_upper_seconds')) is int and v['charged_upper_seconds'] == CAP
                and type(v.get('allocated_GPUh')) in (int, float) and v['allocated_GPUh'] == 2
                for v in (failure, provisional))
            and type(failure.get('cap_seconds')) is int and failure['cap_seconds'] == CAP,
            'Original conservative 2 GPUh charge cannot be reset')
    segments = parent.get('segments')
    require(type(segments) is list and len(segments) == 3
            and [(r.get('steps'), r.get('segment_updates')) for r in segments] == [(4,4),(2,2),(4,2)]
            and all(type(r.get(k)) is int for r in segments for k in ('steps','segment_updates'))
            and not (PRIOR_ROOT / 'cost.closure.json').exists(), 'Original eight physical calls/no closure differs')
    return dict(audit=audit_row, parent=audit['parent'], failure=audit['failure'],
                provisional_cost=audit['provisional_cost'], charged_upper_GPUh=2,
                released_observed=True, physical_optimizer_calls=8, closure_present=False)


def runtime(end=None):
    tr, _, _ = helpers()
    import torch
    require(not torch.cuda.is_initialized(), 'Runtime preparation/verifier must remain CPU-only')
    _, _, _, backend = tr.deterministic_fa3_backend(FRAMEWORK)
    require(not torch.cuda.is_initialized(), 'Actual FA3 CPU import must not initialize CUDA')
    require(backend['backend'] == 'flash_attn_3', 'Actual selected FA3 backend required')
    packages, files = {}, {}
    dist = importlib.metadata.distribution('flash_attn_3')
    packages['flash_attn_3'] = dist.version
    fa3_files = [Path(backend['interface']['path']), Path(backend['native']['path']),
                 Path(backend['native']['path']).parent / '_C.py',
                 Path(dist.locate_file('EGG-INFO/PKG-INFO')).resolve()]
    for p in fa3_files:
        files[str(p)] = asset_record(p, end)
    for name, module in (('torch', 'torch'), ('peft', 'peft'), ('numpy', 'numpy'),
                         ('safetensors', 'safetensors'), ('accelerate', 'accelerate'),
                         ('transformers', 'transformers')):
        dist = importlib.metadata.distribution(name)
        packages[name] = dist.version
        entry = Path(dist.locate_file(module + '/__init__.py')).resolve()
        files[str(entry)] = asset_record(entry, end)
        metadata = next((f for f in dist.files or () if str(f).endswith('.dist-info/METADATA')), None)
        require(metadata is not None, 'Package metadata missing: ' + name)
        meta_path = Path(dist.locate_file(metadata)).resolve()
        files[str(meta_path)] = asset_record(meta_path, end)
        if name == 'torch':
            for p in Path(dist.locate_file('torch')).glob('_C*.so'):
                files[str(p.resolve())] = asset_record(p.resolve(), end)
            for lib in ('libtorch_cpu.so', 'libtorch_cuda.so', 'libc10_cuda.so'):
                p = Path(dist.locate_file('torch/lib/' + lib)).resolve()
                files[str(p)] = asset_record(p, end)
    tools = {n: shutil.which(n) for n in ('timeout', 'nvidia-smi', 'git')}
    require(all(tools.values()), 'Runtime tools missing')
    python = str(Path(sys.executable).resolve())
    for p in [python, *tools.values()]:
        files[str(Path(p).resolve())] = asset_record(Path(p).resolve(), end)
    require(not torch.cuda.is_initialized(), 'Runtime byte freeze must remain CPU-only')
    return dict(python=python, packages=packages, tools=tools, files=list(files.values()),
                actual_FA3=backend, required_flags=dict(tr.DET_FA3_FLAGS), required_environment=dict(tr.DET_FA3_ENV),
                exhaustive_attestation=False, caveat='Selected package entrypoints/native torch/FA3/tools only')


def trainer_args(out, updates, seconds, proof=None):
    require(type(updates) is int and updates in (2, 4) and seconds > 420, 'Insufficient child window')
    args = ['--out-dir', str(out), '--models-dir', str(MODELS), '--framework', str(FRAMEWORK),
            '--smoke', '--synthetic-inputs', '--smoke-deterministic-fa3',
            '--smoke-steps', str(updates), '--smoke-offload', 'on',
            '--offload', 'on', '--max-steps', '4', '--warmup', '1', '--seed', '20261102',
            '--lr', '5e-5', '--weight-decay', '0.01', '--clip', '1', '--anchor-lambda', '0.5',
            '--anchor-weighting', 'training_weight', '--log-every', '1', '--component-grad-every', '10',
            '--load-concurrency', '1', '--max-gpu-hours', str((seconds - 60) / 3600),
            '--cleanup-grace-seconds', '60', '--save-reserve-seconds', '240',
            '--first-step-estimate-seconds', '120', '--target-gpu-hours', '2', '--target-world', '1']
    if proof:
        args += ['--smoke-resume-proof', proof['path'], '--approved-smoke-resume-sha256', proof['sha256']]
    return args


def fixed_config():
    tr, _, _ = helpers()
    args = tr.parser().parse_args(trainer_args(Path('/unused'), 4, 1000))
    return tr.make_cfg(args, 4)


def safe_root(out, fresh=True):
    p = Path(out).absolute()
    require(p == OUTPUT_ROOT and p.parent == LOCAL and p.resolve() == p
            and (not fresh or not p.exists()) and not p.is_symlink(),
            'New node-local allocation root only; no retry/reuse')
    require(LOCAL.is_dir(), 'Node-local parent missing')
    require(not fresh or not list(LOCAL.glob(p.name + '.allocation-failure.*.json')),
            'Allocation already attempted; original startup failure cannot be retried')
    return p


def prepare(out, state_dir):
    tr, admission, process = helpers()
    safe_root(out)
    auth = record(AUTH)
    require(admission.validate_authorization(AUTH) == auth, 'Original authorization mismatch')
    topology = tr.validate_topology()
    require(topology['world_size'] == 1 and topology['node_rank'] == 0, 'Rank0 single-GPU only')
    boundary = process.manager_boundary(state_dir, 0)
    require(Path(state_dir).absolute() == STATE_DIR, 'Fixed lease state directory required')
    source = {n: asset_record(ROOT / n) for n in SOURCE_NAMES}
    framework = tr.framework_records(FRAMEWORK)
    models = {n: asset_record(MODELS / n, expected=tr.m1.CACHE_PINS[n]) for n in tr.SHARD_RELS}
    require(len(models) == 3, 'Exactly three streamed base shards required')
    require(all(models[n]['sha256'] == tr.m1.CACHE_PINS[n] for n in models), 'Actual base content pin differs')
    return dict(schema=SCHEMA, phase=PHASE, private=True, execution_allowed=False,
                formal_training_allowed=False, synthetic_inputs_allowed=True, gold_labels=False,
                retry_allowed=False, borrowing_allowed=False, prior_charged_seconds=CAP,
                new_GPUh_cap=2, old_prior_upper_GPUh=2, cumulative_upper_GPUh=4,
                one_attempt=1, reference_updates=4, prefix_updates=2, restore_updates=2,
                deterministic_FA3_required=True, GPU_capability_unverified=True,
                first_reference_update_is_capability_probe=True, extra_GPU_probe_allowed=False,
                prior_failed_allocation=prior_failure(),
                cap_seconds=CAP, cleanup_seconds=CLEANUP, tail_seconds=TAIL, child_cap_seconds=CHILD_CAP,
                allocated_cards=1, rank=0, gpu_index=0, hostname=os.uname().nodename,
                gpu_uuid=boundary['gpu_uuid'], state_dir=str(Path(state_dir).absolute()),
                output_root=str(Path(out).absolute()), authorization=auth,
                additional_deterministic_authorization=deterministic_authorization(), config=fixed_config(),
                synthetic_samples=64, logical_updates=4, physical_optimizer_calls=8,
                inputs=dict(models_dir=str(MODELS), framework=str(FRAMEWORK), cache_index=None, synthetic_inputs=True),
                trainer_source=tr.training_source_records(),
                synthetic_shapes=dict(x0=list(tr.X0_SHAPE), first=list(tr.FIRST_SHAPE), context=list(tr.CTX_SHAPE), dtype='bfloat16'),
                semantic_approval=False, sota_claim=False, AR_allowed=False, multi_node_execution_allowed=False,
                segments=[dict(name=n, updates=u, cumulative_steps=s,
                               argv=trainer_args(Path(out) / n, u, CHILD_CAP)) for n, u, s in SEGMENTS],
                source=source, framework=framework, models=models, runtime=runtime())


def verify_plan(path, digest, end=None, fresh=True):
    tr, admission, process = helpers()
    bound = record(path, end)
    require(bound['sha256'] == digest, 'Approved plan SHA differs')
    p = read(bound, end)
    require(p.get('schema') == SCHEMA and p.get('phase') == PHASE and p.get('private') is True
            and p.get('execution_allowed') is False and p.get('formal_training_allowed') is False
            and p.get('synthetic_inputs_allowed') is True and p.get('gold_labels') is False
            and p.get('retry_allowed') is False and p.get('borrowing_allowed') is False, 'Technical phase drift')
    expected = dict(prior_charged_seconds=CAP, new_GPUh_cap=2, old_prior_upper_GPUh=2, cumulative_upper_GPUh=4,
                    one_attempt=1, reference_updates=4, prefix_updates=2, restore_updates=2,
                    cap_seconds=CAP, cleanup_seconds=CLEANUP, tail_seconds=TAIL,
                    child_cap_seconds=CHILD_CAP, allocated_cards=1, rank=0, gpu_index=0,
                    synthetic_samples=64, logical_updates=4, physical_optimizer_calls=8)
    require(all(type(p.get(k)) is int and p[k] == v for k, v in expected.items()), 'Fixed budget/count/topology drift')
    require(p.get('deterministic_FA3_required') is True and p.get('GPU_capability_unverified') is True
            and p.get('first_reference_update_is_capability_probe') is True
            and p.get('extra_GPU_probe_allowed') is False
            and encoded(p.get('prior_failed_allocation')) == encoded(prior_failure(end)), 'New deterministic scope/prior failure drift')
    topology = tr.validate_topology()
    require(p['hostname'] == os.uname().nodename and topology['world_size'] == 1
            and topology['node_rank'] == topology['rank'] == topology['local_rank'] == 0
            and p['state_dir'] == str(STATE_DIR) and type(p['gpu_uuid']) is str and p['gpu_uuid'].startswith('GPU-'),
            'Foreign physical host/rank')
    safe_root(p['output_root'], fresh=fresh)
    require(encoded(p['config']) == encoded(fixed_config()), 'Fixed candidate config drift')
    require(p['inputs'] == dict(models_dir=str(MODELS), framework=str(FRAMEWORK), cache_index=None, synthetic_inputs=True)
            and p['trainer_source'] == tr.training_source_records()
            and p['synthetic_shapes'] == dict(x0=list(tr.X0_SHAPE), first=list(tr.FIRST_SHAPE), context=list(tr.CTX_SHAPE), dtype='bfloat16')
            and all(p.get(k) is False for k in ('semantic_approval','sota_claim','AR_allowed','multi_node_execution_allowed')),
            'Input/technical-only scope drift')
    segments = [dict(name=n, updates=u, cumulative_steps=s,
                     argv=trainer_args(Path(p['output_root']) / n, u, CHILD_CAP)) for n, u, s in SEGMENTS]
    require(p['segments'] == segments, 'Fixed three-child command drift')
    require(p['authorization'] == record(AUTH, end)
            and encoded(read(p['authorization'], end)) == encoded(admission.AUTH_EXPECTED), 'Original authorization drift')
    admission.validate_authorization(AUTH)
    require(encoded(p.get('additional_deterministic_authorization')) == encoded(deterministic_authorization(end)),
            'Additional deterministic authorization drift')
    for field, root, names in (('source', ROOT, SOURCE_NAMES), ('framework', FRAMEWORK, tr.FRAMEWORK_FILES),
                               ('models', MODELS, tr.SHARD_RELS)):
        require(set(p[field]) == set(names), 'Frozen asset coverage differs')
        for name, row in p[field].items():
            require(row['path'] == str(root / name) and asset_record(row['path'], end) == row, 'Actual own source/model drift')
            if field == 'models':
                require(row['sha256'] == tr.m1.CACHE_PINS[name], 'Base pin drift')
    rt = p['runtime']
    require(rt['python'] == str(Path(sys.executable).resolve()) and rt['exhaustive_attestation'] is False,
            'Actual Python identity differs')
    require(set(rt['tools']) == {'timeout', 'nvidia-smi', 'git'} and
            all(shutil.which(n) == f for n, f in rt['tools'].items()), 'Actual tool path drift')
    require(set(rt['packages']) == {'torch', 'peft', 'numpy', 'safetensors', 'accelerate', 'transformers', 'flash_attn_3'}
            and all(importlib.metadata.version(n) == v for n, v in rt['packages'].items()), 'Actual package version drift')
    require(len(rt['files']) >= 12 and len({r['path'] for r in rt['files']}) == len(rt['files']), 'Runtime byte coverage missing')
    for row in rt['files']:
        require(asset_record(row['path'], end) == row, 'Runtime byte drift')
    require(encoded(rt) == encoded(runtime(end)), 'Selected runtime file coverage/path/content/type drift')
    process.manager_boundary(p['state_dir'], 0, p['gpu_uuid'])
    deadline(end)
    return p, bound


def verify_approval(path, approved_sha, plan_record, plan, end):
    approval_record = record(path, end)
    require(type(approved_sha) is str and re.fullmatch('[0-9a-f]{64}', approved_sha)
            and approval_record['sha256'] == approved_sha, 'Explicit approved execution approval SHA differs')
    approval = read(approval_record, end)
    keys = {'schema', 'private', 'GPU_authorized', 'plan', 'source_git_root', 'source_commit',
            'publication_manifest', 'remote_receipt', 'phase', 'formal_training_allowed', 'branch', 'snapshot_prefix'}
    require(set(approval) == keys and approval['schema'] == 'teacher_single_gpu_execution_approval_v1'
            and approval['private'] is True and approval['GPU_authorized'] is True
            and approval['formal_training_allowed'] is False and approval['phase'] == PHASE
            and approval['plan'] == plan_record, 'Independent private execution approval required')
    commit = approval['source_commit']
    require(type(commit) is str and re.fullmatch('[0-9a-f]{40}', commit), 'Full immutable source commit required')
    require(approval['source_git_root'] == str(GIT_ROOT) and GIT_ROOT.resolve() == GIT_ROOT
            and approval['branch'] == BRANCH and approval['snapshot_prefix'] == SNAPSHOT_PREFIX,
            'Fixed source Git root/branch/snapshot prefix required')
    manifest = read(approval['publication_manifest'], end)
    receipt = read(approval['remote_receipt'], end)
    require(receipt.get('commit', receipt.get('source_commit')) == commit and receipt.get('remote_equal') is True
            and receipt.get('manifest_sha256') == approval['publication_manifest']['sha256'], 'Original new remote receipt mismatch')
    def git(*args):
        deadline(end)
        r = subprocess.run([plan['runtime']['tools']['git'], '-C', str(GIT_ROOT), *args],
                           capture_output=True, timeout=min(30, max(.01, end - time.monotonic())))
        require(r.returncode == 0 and not r.stderr, 'Actual Git verification failed')
        return r.stdout
    require(git('remote', 'get-url', 'origin').decode().strip() == ORIGIN, 'Foreign remote host/origin forbidden')
    require(git('rev-parse', commit + '^{commit}').decode().strip() == commit, 'Actual new source commit missing')
    require(approval['publication_manifest']['path'] == str(GIT_ROOT / SNAPSHOT_PREFIX / 'PUBLICATION_MANIFEST.json')
            and hashlib.sha256(git('show', commit + ':' + SNAPSHOT_PREFIX + '/PUBLICATION_MANIFEST.json')).hexdigest()
                == approval['publication_manifest']['sha256'], 'Actual committed publication manifest differs')
    rows = manifest.get('files')
    require(type(rows) is list and rows and all(type(r) is dict for r in rows), 'Publication manifest files list required')
    published = {r['path']: r for r in rows}
    require(len(published) == len(rows), 'Duplicate publication file')
    covered = dict(plan['source'])
    covered.update({str(Path(row['path']).relative_to(ROOT)): row for row in plan['framework'].values()})
    for name, row in covered.items():
        require(name in published and set(published[name]) == {'path','source_sha256','snapshot_sha256'}
                and published[name]['source_sha256'] == row['sha256'], 'Published current private source coverage mismatch')
        require(asset_record(row['path'], end) == row, 'Current private source changed after plan')
        raw = git('show', commit + ':' + SNAPSHOT_PREFIX + '/' + name)
        require(hashlib.sha256(raw).hexdigest() == published[name]['snapshot_sha256'],
                'Actual sanitized commit snapshot bytes differ')
    require(type(receipt.get('manifest_files_verified')) is int and receipt['manifest_files_verified'] == len(rows),
            'Actual publication manifest verified count differs')
    changed = [os.fsdecode(raw) for raw in git('diff-tree','--root','--no-commit-id','--name-only','-r','-z',
                commit,'--',SNAPSHOT_PREFIX).split(b'\0') if raw]
    require(type(receipt.get('changed_files_remote_bytes_verified')) is int
            and receipt['changed_files_remote_bytes_verified'] == len(changed) > 0,
            'Actual commit changed-file verified count differs')
    require(SNAPSHOT_PREFIX + '/scripts/run_teacher_single_gpu_admission.py' in changed,
            'New deterministic recovery runner must change in the approved source commit')
    require(all(name == SNAPSHOT_PREFIX + '/PUBLICATION_MANIFEST.json'
                or name.startswith(SNAPSHOT_PREFIX + '/') and name[len(SNAPSHOT_PREFIX)+1:] in published
                for name in changed), 'Changed publication file omitted from manifest')
    for row in rows:
        require(set(row) == {'path','source_sha256','snapshot_sha256'}
                and type(row['path']) is str and not Path(row['path']).is_absolute()
                and '..' not in Path(row['path']).parts and re.fullmatch('[0-9a-f]{64}',row['source_sha256'])
                and re.fullmatch('[0-9a-f]{64}',row['snapshot_sha256']), 'Malformed publication manifest row')
        require(hashlib.sha256(git('show', commit + ':' + SNAPSHOT_PREFIX + '/' + row['path'])).hexdigest()
                == row['snapshot_sha256'], 'Actual full publication snapshot differs: ' + row['path'])
    rows = git('ls-remote', 'origin', 'refs/heads/' + BRANCH).decode().splitlines()
    require(rows == [commit + '\trefs/heads/' + BRANCH], 'Actual remote branch differs; receipt alone cannot allow launch')
    return approval_record

def child_window(start):
    seconds = min(CHILD_CAP, int(start + CAP - TAIL - time.monotonic()))
    require(seconds > 420, 'No remaining work window for another child')
    return seconds


def release_queries_valid(queries, uuid, earliest):
    _, _, process = helpers()
    require(len(queries) >= 2, 'Repeated raw release queries required')
    previous_end = None
    for q in queries:
        require(q.get('command') == process.QUERY_COMMAND and q.get('gpu_uuid') == uuid
                and q.get('returncode') == 0 and ('stderr' not in q or q['stderr'] == '')
                and q.get('pids') == [] and type(q.get('stdout')) is str,
                'Raw release command/result evidence required; not a zero self-claim')
        require(all(type(q.get(k)) in (int, float) and math.isfinite(q[k]) for k in
                    ('start_monotonic', 'end_monotonic'))
                and earliest <= q['start_monotonic'] <= q['end_monotonic'] <= time.monotonic(),
                'Release query allocation-clock mismatch')
        require(previous_end is None or previous_end < q['start_monotonic'],
                'Distinct sequential raw queries required; cloned or overlapping query denied')
        previous_end = q['end_monotonic']
        for row in csv.reader(io.StringIO(q['stdout'])):
            if row:
                require(len(row) == 2 and row[0].strip() != uuid, 'Actual GPU still has a compute PID')
    return True


def run_child(plan, name, updates, start, root, interrupted, proof=None):
    require(threading.current_thread() is threading.main_thread(), 'Only main thread may start GPU work')
    _, _, process = helpers()
    boundary = process.manager_boundary(plan['state_dir'], 0, plan['gpu_uuid'])
    seconds = child_window(start)
    command = [plan['runtime']['python'], str(ROOT / 'scripts/gpu_hold.py'), '--action', 'exec',
               '--gpus', '0', '--state-dir', plan['state_dir'], '--', plan['runtime']['tools']['timeout'],
               '--signal=TERM', '--kill-after=20s', str(seconds) + 's', plan['runtime']['python'], '-B',
               str(ROOT / 'scripts/train_e3_lora.py'), *trainer_args(root / name, updates, seconds, proof)]
    launch = dict(name=name, command=command, parent=process.identity(os.getpid()),
                  started_monotonic=time.monotonic(), started_epoch=time.time(), cap_seconds=seconds,
                  gpu_uuid=boundary['gpu_uuid'], boundary=boundary, state_dir=plan['state_dir'], hostname=os.uname().nodename,
                  deterministic_environment=dict(CUBLAS_WORKSPACE_CONFIG=':4096:8'))
    write_new(root / (name + '.launch.json'), launch)
    child, owner, tracked, queries, error = None, None, [], [], None
    released, final, cleanup_signals = False, None, []
    try:
        require(not interrupted, 'Parent interrupted before spawn')
        env = {k: v for k, v in os.environ.items() if k not in
               ('WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'GROUP_RANK', 'NODE_RANK', 'NNODES')}
        env.update(CUDA_VISIBLE_DEVICES=plan['gpu_uuid'], PYTHONDONTWRITEBYTECODE='1',
                   WORLD_SIZE='1', RANK='0', LOCAL_RANK='0', LOCAL_WORLD_SIZE='1',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', DIFFSYNTH_SKIP_DOWNLOAD='True', OMP_NUM_THREADS='1',
                   PYTHONPATH=str(FRAMEWORK), CUBLAS_WORKSPACE_CONFIG=':4096:8')
        with (root / (name + '.log')).open('xb') as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        candidate = process.identity(child.pid)
        require(candidate['session'] == candidate['pid'] == child.pid
                and candidate['ppid'] == launch['parent']['pid'] and candidate['start_ticks'] > 0,
                'Exact fresh child identity required')
        owner = candidate
        write_new(root / (name + '.supervised.json'), dict(launch=record(root / (name + '.launch.json')), child=owner))
        stop = min(start + CAP - TAIL, launch['started_monotonic'] + seconds)
        while child.poll() is None and time.monotonic() < stop and not interrupted:
            process.owned_members(owner, tracked)
            time.sleep(.1)
        require(not interrupted and child.poll() == 0, 'Child failed/timed out/interrupted; no retry')
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    finally:
        interrupted.cleaning = True
        try:
            if child is not None and owner is not None:
                cleanup_signals = process.cleanup_owned(child, owner, min(start + CAP - 60, time.monotonic() + 24), tracked)
                queries = [process.query_release(plan['gpu_uuid']) for _ in range(2)]
                released = not process.owned_members(owner, tracked) and release_queries_valid(queries, plan['gpu_uuid'], launch['started_monotonic'])
                final = process.manager_boundary(plan['state_dir'], 0, plan['gpu_uuid'], require_idle=False)
                require(final['query']['pids'] == [], 'Actual third GPU release query is occupied')
                require(final['manager']['pid'] == boundary['manager']['pid']
                        and final['manager']['start_ticks'] == boundary['manager']['start_ticks'], 'Manager identity drift')
            elif child is not None:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=2)
                error = error or 'Unknown owned tree after child identity failure; reservation retained'
        except BaseException as exc:
            released = False
            error = 'Owned cleanup/release failed: ' + str(exc)
        receipt = dict(launch=launch, child=owner, exit_code=child.returncode if child else None,
                       owned_processes=tracked, cleanup_signals=cleanup_signals, release_queries=queries,
                       release_boundary=final, released=released,
                       error=error, ended_monotonic=time.monotonic(), success=error is None and released,
                       log=record(root / (name + '.log'), allow_empty=True) if (root / (name + '.log')).is_file() else None)
        write_new(root / (name + '.parent.json'), receipt)
        interrupted.cleaning = False
    require(receipt['success'], 'Child failure/unknown release; no retry')
    return receipt


def segment_state(plan, name, updates, cumulative, root, end):
    tr, _, _ = helpers()
    import torch
    import merge_e3_lora as merge
    from safetensors.torch import load_file
    require(not torch.cuda.is_initialized(), 'Parent CPU verifier must not initialize CUDA')
    require(encoded(runtime(end)) == encoded(plan['runtime']), 'Actual runtime full bytes changed across trainer segments')
    out = root / name
    summary_row, manifest_row = record(out / 'summary.json', end), record(out / 'run_manifest.json', end)
    summary, manifest = read(summary_row, end), read(manifest_row, end)
    backend_row = record(out / 'deterministic_backend.json', end)
    require(summary.get('deterministic_backend') == backend_row and manifest.get('deterministic_backend') == backend_row,
            'Original trainer-written deterministic backend record required')
    backend = read(backend_row, end)
    current = tr.deterministic_fa3_runtime(tr.parser().parse_args(trainer_args(out, updates, CHILD_CAP)))
    require(not torch.cuda.is_initialized(), 'Actual backend verifier must remain CPU-only')
    require(encoded(backend) == encoded(current)
            and encoded({k: backend[k] for k in ('backend','interface','native','torch_version')})
                == encoded(plan['runtime']['actual_FA3'])
            and encoded(backend['flags']) == encoded(plan['runtime']['required_flags'])
            and encoded(backend['environment']) == encoded(plan['runtime']['required_environment'])
            and backend['additional_deterministic_authorization'] == plan['additional_deterministic_authorization'],
            'Actual selected deterministic FA3/readback flags/current runtime full SHA differs')
    parent = read(record(root / (name + '.parent.json'), end), end)
    launch = read(record(root / (name + '.launch.json'), end), end)
    require(launch.get('deterministic_environment') == tr.DET_FA3_ENV, 'Pre-CUDA child deterministic environment drift')
    require(parent['launch'] == launch and parent['success'] is True and parent['released'] is True
            and parent['exit_code'] == 0 and parent['error'] is None, 'Original successful owned parent proof required')
    release_queries_valid(parent['release_queries'], plan['gpu_uuid'], launch['started_monotonic'])
    resume_row = record(root / 'resume_proof.json', end) if cumulative != updates else None
    expected_args = trainer_args(out, updates, launch['cap_seconds'], resume_row)
    require(manifest['argv'] == expected_args and manifest['allocation'] == tr.budget(
                (launch['cap_seconds']-60)/3600,1,60,240,120), 'Original exact child arguments/allocation differs')
    require(summary.get('protocol') == tr.PROTOCOL and summary.get('mode') == 'smoke'
            and summary.get('status') == 'completed' and type(summary.get('completed_steps')) is int
            and summary['completed_steps'] == cumulative and set(summary.get('smoke', {})) == {'on'}
            and type(summary['smoke']['on']['completed_steps']) is int
            and summary['smoke']['on']['completed_steps'] == updates and summary['smoke']['on']['stop'] is None,
            'Actual smoke segment incomplete/stopped')
    require(encoded(manifest['config']) == encoded(plan['config'])
            and manifest['source'] == tr.training_source_records() and type(manifest['world_size']) is int and manifest['world_size'] == 1
            and type(manifest['num_samples']) is int and manifest['num_samples'] == 64
            and manifest['torch_version'] == str(torch.__version__) and manifest['data'] == tr.synthetic_data_identity()
            and manifest['base_shards_hashed'] is True
            and manifest['base_shards'] == {r: tr.m1.CACHE_PINS[r] for r in tr.SHARD_RELS}
            and manifest['framework'] == tr.framework_records(FRAMEWORK)
            and manifest['inputs'] == plan['inputs'] and manifest['source'] == plan['trainer_source']
            and manifest['smoke'] == dict(on=True, synthetic_inputs=True, steps_per_mode=updates, offload_modes=['on'],
                python_rng_rule=tr.SMOKE_PYTHON_RNG_RULE, python_seed=tr.derive_seed('smoke_python', plan['config']['seed']))
            and manifest['segment_requested_updates'] == updates and manifest['logical_start_step'] == cumulative-updates
            and manifest['logical_end_step_requested'] == cumulative and manifest['order_plan_steps'] == updates
            and manifest['order_plan_sha256'] == tr.plan_sha256(
                [[tr.DataOrder(plan['config']['seed'],64).index(s,0,1)] for s in range(cumulative-updates,cumulative)])
            and summary.get('segment_executed_updates') == updates and summary.get('logical_completed_steps') == cumulative
            and summary['smoke']['on']['step_offset'] == cumulative-updates
            and summary['smoke']['on']['logical_completed_steps'] == cumulative, 'Actual segment identity mismatch')
    require(all(type(manifest.get(k)) is int for k in ('segment_requested_updates','logical_start_step',
                'logical_end_step_requested','order_plan_steps'))
            and all(type(summary.get(k)) is int for k in ('segment_executed_updates','logical_completed_steps'))
            and all(type(summary['smoke']['on'].get(k)) is int for k in ('step_offset','logical_completed_steps')),
            'Actual segment counts must be strict integers')
    require(summary['trainer_identity'] == manifest['trainer_identity'], 'Summary/manifest actual trainer identity differs')
    require(summary['allocation'] == manifest['allocation'] and manifest['allocation']['world_size'] == 1
            and 0 < manifest['allocation']['max_gpu_hours'] <= 2
            and summary['ledger']['within_budget'] is True and summary['ledger']['cards'] == 1
            and math.isfinite(summary['ledger']['wall_seconds']) and summary['ledger']['wall_seconds'] > 0,
            'Original complete trainer-process fee evidence required')
    if cumulative == updates:
        require(manifest['resume'] is None and not summary.get('resume'), 'Fresh source cannot be chain-resume')
    else:
        proof_row = record(root / 'resume_proof.json', end)
        require(manifest['resume']['proof'] == proof_row and summary['resume'] == manifest['resume']
                and manifest['resume']['source_completed_steps'] == 2 and manifest['resume']['continuation_updates'] == 2,
                'Actual fresh restore must bind original proof')
    progress_row = record(out / 'progress.jsonl', end)
    rows = [json.loads(line) for line in (out / 'progress.jsonl').read_text().splitlines()]
    require(len(rows) == updates and [r['step'] for r in rows] == list(range(cumulative - updates + 1, cumulative + 1))
            and all(r['status'] == 'ok' and all(type(r[k]) in (int, float) and math.isfinite(r[k])
                    for k in ('loss_total_mean', 'loss_fm_mean', 'loss_anchor_mean', 'grad_norm_total_before_clip')) for r in rows)
            and all(r['grad_norm_total_before_clip'] > 0 for r in rows), 'Actual finite gradient/update mismatch')
    ck = summary['checkpoint']
    require(set(ck) == {'path', 'bytes', 'file_sha256', 'state_sha256'}
            and ck['path'] == str(out / 'private' / tr.CKPT_NAME)
            and record(ck['path'], end) == dict(path=ck['path'], bytes=ck['bytes'], sha256=ck['file_sha256']), 'Full checkpoint SHA mismatch')
    deadline(end)
    state = torch.load(ck['path'], map_location='cpu', weights_only=True)
    require(tr.ts.state_hash(state) == ck['state_sha256'] and type(state['steps']) is int and state['steps'] == cumulative
            and encoded(state['config']) == encoded(dict(plan['config'], completed_steps=cumulative)), 'Actual checkpoint config mismatch')
    require(set(state) == {'protocol', 'steps', 'targets', 'rank', 'alpha', 'adapters', 'adapter_names', 'optimizer', 'rng', 'config'}
            and state['protocol'] == tr.PROTOCOL and state['targets'] == tr.TARGETS and state['rank'] == state['alpha'] == 32
            and len(state['adapters']) == 600 and state['adapter_names'] == list(state['adapters']), 'Native checkpoint schema mismatch')
    require(set(state['optimizer']) == {'param_groups', 'state'}, 'Actual optimizer schema mismatch')
    groups, slots = state['optimizer']['param_groups'], state['optimizer']['state']
    ids = [i for g in groups for i in g['params']]
    require(len(groups) == 1 and len(ids) == 600 and len(set(ids)) == 600 and set(slots) == set(ids), 'Actual AdamW coverage mismatch')
    g = groups[0]
    require(g['lr'] == plan['config']['lr'] * tr.lr_factor(cumulative, 1, 4)
            and g['weight_decay'] == .01 and tuple(g['betas']) == (.9, .999) and g['eps'] == 1e-8, 'Actual AdamW hyperparameter mismatch')
    for pid, tensor in zip(ids, state['adapters'].values()):
        slot = slots[pid]
        require(tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all())
                and set(slot) == {'step', 'exp_avg', 'exp_avg_sq'} and float(slot['step']) == cumulative,
                'Actual AdamW step/native tensor mismatch')
        for k in ('exp_avg', 'exp_avg_sq'):
            require(slot[k].dtype == torch.float32 and slot[k].shape == tensor.shape
                    and bool(torch.isfinite(slot[k]).all()), 'Actual AdamW moments invalid')
        require(bool((slot['exp_avg_sq'] >= 0).all()), 'Negative second moment')
    require(sum(t.numel() for t in state['adapters'].values()) == 80609280, 'Native parameter count mismatch')
    adapter_hash = tr.ts.state_hash(state['adapters'])
    require(summary.get('adapter_final_sha256') == adapter_hash and tr.sha256_text(summary.get('adapter_initial_sha256'))
            and summary['adapter_initial_sha256'] != adapter_hash and tr.sha256_text(summary.get('frozen_base_sha256')),
            'No actual adapter change/frozen base evidence')
    export = summary['export']
    require(export['path'] == str(out / 'export' / tr.EXPORT_NAME) and export['tensor_count'] == 600 and export['parameter_count'] == 80609280
            and record(export['path'], end) == dict(path=export['path'], bytes=export['bytes'], sha256=export['file_sha256']), 'Actual export SHA mismatch')
    tensors = load_file(export['path'], device='cpu')
    require(set(tensors) == set(state['adapters']) and all(torch.equal(tensors[k], state['adapters'][k]) for k in tensors),
            'Actual export/checkpoint adapters differ')
    merge.check_lora(tensors, merge.base_header([MODELS / r for r in tr.SHARD_RELS]))
    require(set(state['rng']) == {'python', 'torch_cpu', 'torch_cuda'} and len(state['rng']['torch_cuda']) == 1,
            'Actual one-device RNG required')
    identity = summary['trainer_identity']
    require(type(identity['pid']) is int and identity['pid'] > 0 and type(identity['start_ticks']) is int
            and identity['start_ticks'] > 0, 'Actual trainer PID/start missing')
    require(any(r['pid'] == identity['pid'] and r['start_ticks'] == identity['start_ticks']
                and str(ROOT / 'scripts/train_e3_lora.py') in r['argv']
                for r in parent['owned_processes']), 'Trainer not actually observed in owned child tree')
    deadline(end)
    trajectory = [{k: r[k] for k in ('step','status','loss_total_mean','loss_fm_mean','loss_anchor_mean',
                    'grad_norm_total_before_clip','grad_norm_fm','grad_norm_anchor_weighted','lr','per_rank')} for r in rows]
    result = dict(summary=summary_row, manifest=manifest_row, progress=progress_row, checkpoint=ck,
                  deterministic_backend=backend_row, backend_identity=backend, exact_trajectory=trajectory,
                  trainer_identity=identity, frozen_base_sha256=summary['frozen_base_sha256'],
                  adapter_sha256=adapter_hash, initial_adapter_sha256=summary['adapter_initial_sha256'],
                  optimizer_sha256=tr.ts.state_hash(state['optimizer']),
                  rng_sha256=tr.ts.state_hash(state['rng']), steps=cumulative, segment_updates=updates)
    return result, state


def make_resume_proof(plan, prefix, root):
    tr, _, _ = helpers()
    original = read(prefix['manifest'])
    require(prefix['steps'] == prefix['segment_updates'] == 2, 'Verified successful original two-step prefix required')
    proof = dict(schema='e3-single-gpu-smoke-resume-v1', phase=PHASE, authorization=plan['authorization'],
                 source_run_manifest=prefix['manifest'], source_summary=prefix['summary'], checkpoint=prefix['checkpoint'],
                 source_completed_steps=2, continuation_updates=2, logical_horizon=4, config=original['config'],
                 source=original['source'], synthetic_inputs_allowed=True, formal_training_allowed=False,
                 additional_deterministic_authorization=plan['additional_deterministic_authorization'],
                 deterministic_backend=prefix['deterministic_backend'])
    return write_new(root / 'resume_proof.json', proof)


def compare_states(reference, resumed):
    tr, _, _ = helpers()
    require(reference['steps'] == resumed['steps'] == 4 and reference['config'] == resumed['config'], 'Final logical trajectory differs')
    for key in ('adapters', 'optimizer', 'rng'):
        require(tr.ts.state_hash(reference[key]) == tr.ts.state_hash(resumed[key]), 'Exact resumed trajectory differs: ' + key)
    full = tr.ts.state_hash(reference)
    require(full == tr.ts.state_hash(resumed), 'Exact full final checkpoint state differs')
    return dict(adapter_optimizer_rng_exact=True, full_state_sha256=full, tolerance_relaxed=False,
                adapter_sha256=tr.ts.state_hash(reference['adapters']), optimizer_sha256=tr.ts.state_hash(reference['optimizer']),
                rng_sha256=tr.ts.state_hash(reference['rng']), logical_updates=4, physical_optimizer_calls=8)


def cost(start, released, publication_allowance=0):
    elapsed = time.monotonic() - start
    require(math.isfinite(elapsed) and elapsed >= 0 and publication_allowance >= 0, 'Invalid local allocation clock')
    charged = elapsed + publication_allowance if released else max(CAP, elapsed + publication_allowance)
    return dict(elapsed_seconds=elapsed, actual_elapsed_seconds=elapsed, charged_upper_seconds=charged,
                allocated_GPUh=charged / 3600, actual_allocated_GPUh=elapsed/3600 if released else None,
                released=released, budget_exceeded=charged > CAP,
                scope='pre_spawn_through_final_release_and_parent_save', clock_host=os.uname().nodename,
                cap_seconds=CAP, prior_charged_seconds=CAP, old_prior_upper_GPUh=2, new_GPUh_cap=2,
                cumulative_charged_upper_seconds=CAP+charged, cumulative_upper_GPUh=(CAP+charged)/3600,
                budgets_independent=True, borrowing_allowed=False, one_attempt=1)


class Interruptions(list):
    cleaning = False
    timed_out = False


def verify_cost_closure(path, digest):
    """Consumer entrypoint; neither parent nor provisional cost is completion evidence."""
    row = record(path)
    require(row['sha256'] == digest and Path(path).name == 'cost.closure.json', 'Explicit closure SHA/path required')
    root = Path(row['path']).parent
    require(not (root / 'cost.failure.json').exists(), 'Terminal publication failed; closure cannot authorize completion')
    value = read(row)
    require(value.get('schema') == 'teacher_single_gpu_cost_closure_v1' and value.get('phase') == PHASE
            and value.get('success') is True and value.get('released') is True
            and value.get('terminal') is True and value.get('cap_seconds') == CAP
            and type(value['cap_seconds']) is int and value.get('clock_host') == os.uname().nodename,
            'Successful same-host terminal cost closure required')
    for key in ('started_monotonic', 'observed_monotonic', 'actual_elapsed_seconds',
                'charged_upper_seconds', 'publication_exit_allowance_seconds'):
        require(type(value.get(key)) in (int, float) and math.isfinite(value[key]), 'Finite strict closure clock required')
    require(value['actual_elapsed_seconds'] == value['observed_monotonic'] - value['started_monotonic']
            and 0 <= value['actual_elapsed_seconds'] + value['publication_exit_allowance_seconds'] <= CAP
            and value['publication_exit_allowance_seconds'] == 1
            and value['charged_upper_seconds'] == CAP and value['budget_exceeded'] is False,
            'Terminal cost/reserved tail/cap differs')
    require(value['parent']['path'] == str(root / 'parent.json')
            and value['cost']['path'] == str(root / 'cost.json'), 'Own terminal evidence paths required')
    plan = read(value['plan'])
    require(plan.get('schema') == SCHEMA and plan.get('phase') == PHASE
            and plan.get('output_root') == str(root), 'Original approved allocation plan binding differs')
    parent, provisional = read(value['parent']), read(value['cost'])
    require(parent['plan'] == value['plan'] and provisional['plan'] == value['plan']
            and parent['started_monotonic'] == value['started_monotonic']
            and parent['execution_success'] is True and parent['success'] is False
            and parent['released'] is True and parent['error'] is None
            and provisional['provisional'] is True and provisional['success'] is False
            and provisional['parent_receipt'] == value['parent']
            and provisional['charged_upper_seconds'] >= CAP
            and type(provisional.get('prior_charged_seconds')) is int and provisional['prior_charged_seconds'] == CAP
            and provisional.get('budgets_independent') is True and provisional.get('borrowing_allowed') is False
            and type(value.get('prior_charged_seconds')) is int and value['prior_charged_seconds'] == CAP
            and type(value.get('old_prior_upper_GPUh')) is int and value['old_prior_upper_GPUh'] == 2
            and type(value.get('new_GPUh_cap')) is int and value['new_GPUh_cap'] == 2
            and type(value.get('cumulative_upper_GPUh')) is int and value['cumulative_upper_GPUh'] == 4
            and value.get('budgets_independent') is True and value.get('borrowing_allowed') is False,
            'Original parent/provisional cost/independent prior budget binding differs')
    require(not (root / 'cost.failure.json').exists(), 'Terminal publication failed during closure verification')
    return value


def run(path, digest, approval_path, approved_approval_sha, gpu_authorized=False):
    require(gpu_authorized is True, 'Explicit --gpu-authorized required')
    require(threading.current_thread() is threading.main_thread(), 'Only main thread may start GPU work')
    # The first hash, helper imports and terminal publication share one clock.
    start, epoch = time.monotonic(), time.time()
    root, bound, plan, process, owner, created = None, None, None, None, None, False
    interrupted, old, children, results = Interruptions(), {}, [], []
    released, error, comparison, approval, initial, final, queries = False, None, None, None, None, None, []
    final_result, terminal_success = None, False
    def stop(signum, _):
        if not interrupted:
            interrupted.append(signum)
    def alarm(signum, frame):
        interrupted.timed_out = True
        stop(signum, frame)
        raise TimeoutError('One allocation cap reached; incomplete publication cannot pass')
    try:
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            old[s] = signal.signal(s, alarm if s == signal.SIGALRM else stop)
        signal.setitimer(signal.ITIMER_REAL, max(.001, start + CAP - TAIL - time.monotonic()))
        bound = record(path, start + CAP - TAIL)
        require(bound['sha256'] == digest, 'Approved plan SHA differs')
        plan = read(bound, start + CAP - TAIL)
        require(plan.get('schema') == SCHEMA and plan.get('phase') == PHASE
                and plan.get('hostname') == os.uname().nodename, 'Foreign schema/phase/host')
        root = safe_root(plan['output_root'])
        root.mkdir(mode=0o700)
        created = True
        _, _, process = helpers()
        owner = process.identity(os.getpid())
        write_new(root / 'launch.json', dict(plan=bound, approval_path=str(approval_path), approved_approval_sha256=approved_approval_sha,
                   parent=owner, hostname=os.uname().nodename, start_monotonic=start, start_epoch=epoch,
                   gpu_uuid=plan['gpu_uuid'], cap_seconds=CAP))
        plan, bound = verify_plan(path, digest, start + CAP - TAIL, fresh=False)
        approval = verify_approval(approval_path, approved_approval_sha, bound, plan, start + CAP - TAIL)
        initial = process.manager_boundary(plan['state_dir'], 0, plan['gpu_uuid'])
        timeout_version = subprocess.run([plan['runtime']['tools']['timeout'], '--version'], capture_output=True,
                                         check=True, text=True, timeout=10)
        require('GNU coreutils' in timeout_version.stdout, 'Authenticated GNU timeout required')
        reference, prefix, proof = None, None, None
        for name, updates, cumulative in SEGMENTS:
            require(not interrupted, 'Parent interrupted; no next child')
            require(record(path, start + CAP - TAIL) == bound
                    and record(plan['authorization']['path'], start + CAP - TAIL) == plan['authorization'],
                    'Approved plan/authorization changed before child spawn')
            tr, admission, _ = helpers()
            admission.validate_authorization(plan['authorization']['path'])
            require(plan['additional_deterministic_authorization'] == deterministic_authorization(start + CAP - TAIL)
                    and plan['prior_failed_allocation'] == prior_failure(start + CAP - TAIL),
                    'Original failure/new independent authorization changed before spawn')
            for row in plan['source'].values():
                require(asset_record(row['path'], start + CAP - TAIL) == row, 'Current source changed before spawn')
            for row in [*plan['framework'].values(), *plan['models'].values(), *plan['runtime']['files']]:
                tr.m1.validate_record(row, deadline=start + CAP - TAIL, hash_content=False)
            if name == 'resume':
                proof = make_resume_proof(plan, results[-1], root)
            child = run_child(plan, name, updates, start, root, interrupted, proof)
            children.append(child)
            result, state = segment_state(plan, name, updates, cumulative, root, start + CAP - TAIL)
            results.append(result)
            if name == 'reference':
                reference = state
            elif name == 'prefix':
                prefix = result
                del state
            else:
                comparison = compare_states(reference, state)
                del state, reference
        require(len({(r['trainer_identity']['pid'], r['trainer_identity']['start_ticks']) for r in results}) == 3,
                'Three actual distinct CUDA trainer processes required')
        require(len({r['frozen_base_sha256'] for r in results}) == 1, 'Common base changed')
        require(all(encoded(r['backend_identity']) == encoded(results[0]['backend_identity']) for r in results),
                'Three actual child deterministic backends must be exactly identical')
        require(encoded(results[0]['exact_trajectory'][:2]) == encoded(results[1]['exact_trajectory'])
                and encoded(results[0]['exact_trajectory'][2:]) == encoded(results[2]['exact_trajectory']),
                'Exact reference/prefix/resume gradients/loss/sample trajectory differs')
        require(results[-1]['steps'] == 4 and prefix['steps'] == 2
                and results[0]['initial_adapter_sha256'] == results[1]['initial_adapter_sha256']
                and prefix['adapter_sha256'] == results[-1]['initial_adapter_sha256'], 'Exact initial/prefix-to-restored identity differs')
        require(not interrupted, 'Parent interrupted after children')
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    finally:
        interrupted.cleaning = True
        signal.setitimer(signal.ITIMER_REAL, max(.001, start + CAP - time.monotonic()))
        try:
            if created:
                try:
                    parents = [read(record(p)) for p in root.glob('*.parent.json')]
                    launches = list(root.glob('*.launch.json'))
                    if not launches:
                        released = True  # No owned GPU child was dispatched, even if parent identity failed.
                    else:
                        require(process is not None, 'Unknown owned dispatch helper; reservation retained')
                        queries = [process.query_release(plan['gpu_uuid']) for _ in range(2)]
                        final = process.manager_boundary(plan['state_dir'], 0, plan['gpu_uuid'], require_idle=False)
                        require(final['query']['pids'] == [], 'Actual final manager device query is occupied')
                        released = all(p['released'] is True for p in parents) and release_queries_valid(queries, plan['gpu_uuid'], start)
                    require(len(parents) == len(launches), 'Unknown dispatch receipt coverage')
                    if initial is not None and final is not None:
                        require(final['manager']['pid'] == initial['manager']['pid']
                                and final['manager']['start_ticks'] == initial['manager']['start_ticks'], 'Final manager identity drift')
                    if error is None:
                        require(len(parents) == len(children) == 3, 'Parent receipt coverage differs')
                except BaseException as exc:
                    released = False
                    error = (error + '; ' if error else '') + 'Final release failed: ' + str(exc)
                receipt = dict(schema=SCHEMA, phase=PHASE, plan=bound, approval=approval, owner=owner,
                               hostname=os.uname().nodename, gpu_uuid=plan['gpu_uuid'],
                               started_monotonic=start, started_epoch=epoch, ended_monotonic=time.monotonic(),
                               interrupted_signals=list(interrupted), timed_out=interrupted.timed_out, children=children, segments=results,
                               initial_boundary=initial, release_boundary=final, release_queries=queries,
                               released=released, comparison=comparison, error=error,
                               no_child_spawned=not list(root.glob('*.launch.json')),
                               execution_success=error is None and released and len(results) == 3 and not interrupted,
                               success=False, provisional=True, completion_requires='cost.closure.json',
                               budget_exceeded=time.monotonic()-start > CAP,
                               formal_training_allowed=False, quality_pass=False, synthetic_only=True,
                               AR_allowed=False, semantic_approval=False, multi_node_execution_allowed=False)
                parent_record, save_error = None, None
                try:
                    parent_record = write_new(root / 'parent.json', receipt)
                except BaseException as exc:
                    save_error = type(exc).__name__ + ': ' + str(exc)
                # Reserve the whole cap until all terminal publication is actually checked.
                provisional = cost(start, False)
                provisional.update(publication_exit_allowance_seconds=1, parent_receipt=parent_record,
                                   parent_save_error=save_error, plan=bound, provisional=True, success=False)
                try:
                    cost_record = write_new(root / 'cost.json', provisional)
                    deadline(start + CAP)
                    require(parent_record is not None and save_error is None, 'Parent receipt save failed: ' + str(save_error))
                    require(receipt['execution_success'] and not interrupted, 'Admission execution failed; no retry')
                    observed = time.monotonic()
                    require(observed - start + 1 <= CAP, 'No bounded terminal publication allowance remains')
                    closure = dict(schema='teacher_single_gpu_cost_closure_v1', phase=PHASE,
                                   plan=bound, parent=parent_record, cost=cost_record, terminal=True,
                                   success=True, released=True, clock_host=os.uname().nodename,
                                   started_monotonic=start, observed_monotonic=observed,
                                   actual_elapsed_seconds=observed-start, charged_upper_seconds=CAP,
                                   allocated_GPUh=CAP/3600, publication_exit_allowance_seconds=1,
                                   cap_seconds=CAP, budget_exceeded=False, prior_charged_seconds=CAP,
                                   old_prior_upper_GPUh=2, new_GPUh_cap=2, cumulative_upper_GPUh=4,
                                   budgets_independent=True, borrowing_allowed=False, one_attempt=1)
                    closure_record = write_new(root / 'cost.closure.json', closure)
                    verify_cost_closure(closure_record['path'], closure_record['sha256'])
                    final_result = record(root / 'cost.closure.json')
                    finished = time.monotonic()
                    require(finished - observed <= 1 and finished - start <= CAP and not interrupted,
                            'Terminal save/return-record exceeded measured tail allowance or allocation cap')
                    terminal_success = True
                except BaseException as exc:
                    error = (error + '; ' if error else '') + 'Terminal closure failed: ' + type(exc).__name__ + ': ' + str(exc)
                    failure = cost(start, False)
                    failure.update(schema='teacher_single_gpu_cost_failure_v1', phase=PHASE, plan=bound,
                                   terminal=True, success=False, error=error, released_observed=released,
                                   parent_receipt=parent_record, parent_save_error=save_error)
                    write_new(root / 'cost.failure.json', failure)
            else:
                failure = cost(start, False)
                failure.update(schema='teacher_single_gpu_cost_failure_v1', phase=PHASE, plan=bound,
                               terminal=True, success=False, no_child_spawned=True, error=error)
                print(json.dumps(failure, sort_keys=True), file=sys.stderr)
                if root is not None:
                    import uuid
                    write_new(root.parent / (root.name + '.allocation-failure.' + uuid.uuid4().hex + '.json'), failure)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for s, handler in old.items():
                signal.signal(s, handler)
    require(terminal_success, 'Admission failed; costs preserved; terminal closure required; no retry')
    return final_result

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'verify', 'run', 'verify-closure'))
    p.add_argument('--out-dir', type=Path)
    p.add_argument('--state-dir', type=Path, default=STATE_DIR)
    p.add_argument('--plan', type=Path)
    p.add_argument('--approved-plan-sha256')
    p.add_argument('--execution-approval', type=Path)
    p.add_argument('--approved-execution-approval-sha256')
    p.add_argument('--gpu-authorized', action='store_true')
    p.add_argument('--closure', type=Path)
    p.add_argument('--approved-closure-sha256')
    args = p.parse_args(argv)
    try:
        require((args.execution_approval is None) == (args.approved_execution_approval_sha256 is None),
                'Execution approval path and raw approved SHA flags must be paired')
        if args.action == 'verify-closure':
            require(args.closure is not None and args.approved_closure_sha256 is not None and not args.gpu_authorized,
                    'CPU closure verify requires explicit raw closure SHA, not GPU authorization')
            print(json.dumps(verify_cost_closure(args.closure, args.approved_closure_sha256)))
        elif args.action == 'prepare':
            require(args.out_dir is not None and args.plan is not None and not args.gpu_authorized,
                    'CPU prepare requires new out-dir/plan, not GPU authorization')
            print(json.dumps(write_new(args.plan, prepare(args.out_dir, args.state_dir))))
        elif args.action == 'verify':
            require(args.plan is not None and args.approved_plan_sha256 is not None and not args.gpu_authorized,
                    'CPU verify requires explicitly approved plan')
            plan, row = verify_plan(args.plan, args.approved_plan_sha256)
            if args.execution_approval is not None:
                verify_approval(args.execution_approval, args.approved_execution_approval_sha256, row, plan, time.monotonic()+600)
            print(json.dumps(dict(plan=row, verified=True, GPU_used=False, execution_allowed=False)))
        else:
            require(args.plan is not None and args.approved_plan_sha256 is not None
                    and args.execution_approval is not None and args.approved_execution_approval_sha256 is not None,
                    'GPU run needs plan and independent raw-byte-approved execution approval')
            print(json.dumps(run(args.plan, args.approved_plan_sha256, args.execution_approval,
                                 args.approved_execution_approval_sha256, args.gpu_authorized)))
        return 0
    except Exception as exc:
        print(type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
