#!/usr/bin/env python3
"""Offline, bounded scoring of the frozen internal-dev 24-video CFG probe.

This is not a public benchmark or a Teacher gate. CPU DreamSim runs first;
independent GPU workers use the original VBench DINO and MUSIQ algorithms.
No resume: run-dir/scoring must not exist. Failures retain logs and resources.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
VBENCH = Path('/dockerdata/wm_eval/VBench')
PYTHON = '/dockerdata/venv_vbench/bin/python'
DINO_REPO = Path('/dockerdata/vbench_cache/dino_model/facebookresearch_dino_main')
DINO_WEIGHT = Path('/dockerdata/vbench_cache/dino_model/dino_vitbase16_pretrain.pth')
MUSIQ_WEIGHT = Path('/dockerdata/vbench_cache/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth')
DREAM_CACHE = VBENCH / 'models'
MODELS = ('base', 'm_full1_a07')
ARMS = {(model, cfg) for model in MODELS for cfg in (3, 5)}
METRICS = ('subject_consistency', 'imaging_quality', 'dreamsim', 'dreamsim_last', 'motion_ratio')
DIRECTIONS = dict(zip(METRICS, ('higher', 'higher', 'lower', 'lower', 'lower')))
BOOTSTRAP_SEED, BOOTSTRAP_REPLICATES = 20261019, 5000
MERGE_SHA256 = '057d9a1e682e2f237594f5ef99f994f07699188c97b4196016fd371ab40402a6'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, label, minimum=None, maximum=None):
    require(type(value) in (int, float) and math.isfinite(value), label + ' must be numeric and finite (not bool)')
    require(minimum is None or value >= minimum, label + ' below range')
    require(maximum is None or value <= maximum, label + ' above range')
    return float(value)


def sha256(path, deadline=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            require(deadline is None or time.perf_counter() < deadline, 'Hash validation reached scoring hard timeout')
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    def invalid_constant(value):
        raise ValueError('Nonfinite JSON number: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid_constant)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def checked_hash(path, expected, label, deadline=None):
    require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected), 'Missing SHA256: ' + label)
    require(Path(path).is_file() and sha256(path, deadline) == expected, 'SHA256 changed: ' + label)


def decode_video(path):
    # PyAV decodes every frame; container metadata alone never proves completion.
    import av
    count = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 0)
        require(math.isclose(fps, 24, abs_tol=0.01), 'Decoded FPS must be 24')
        for frame in container.decode(video=0):
            require((frame.width, frame.height) == (1280, 704), 'Decoded size must be 1280x704')
            count += 1
    require(count == 121, 'Complete decode must contain 121 frames')
    return {'frames': count, 'width': 1280, 'height': 704, 'fps': fps}


def validate_charge(resources, actual, reserved, effective, label):
    actual = number(actual, label + ' actual', 0)
    reserved = number(reserved, label + ' reservation', 0)
    effective = number(effective, label + ' cap', 0)
    require(isinstance(resources, list) and 1 <= len(resources) <= 4, label + ' must have 1..4 workers')
    seen = set()
    charges = []
    for row in resources:
        require(isinstance(row['gpu'], str) and row['gpu'] not in seen, label + ' duplicate GPU')
        seen.add(row['gpu'])
        seconds = number(row['seconds'], label + ' worker seconds', 0)
        charge = number(row['gpu_hours'], label + ' worker GPU hours', 0)
        require(math.isclose(charge, seconds / 3600, abs_tol=1e-10), label + ' worker charge mismatch')
        require(type(row['returncode']) is int and row['returncode'] == 0, label + ' worker exit failed')
        charges.append(charge)
    require(math.isclose(actual, math.fsum(charges), abs_tol=1e-10), label + ' total charge mismatch')
    require(actual <= reserved and actual <= effective, label + ' actual exceeds reservation/cap')
    return actual


def budget(gpus, timeout, maximum, generation_actual):
    require(isinstance(gpus, str) and re.fullmatch(r'[0-9]+(?:,[0-9]+)*', gpus), 'GPU IDs must be comma-separated integers')
    cards = gpus.split(',')
    require(len(cards) == len(set(cards)) and 1 <= len(cards) <= 4 and all(str(int(g)) == g for g in cards),
            'Require 1..4 unique canonical GPU IDs')
    timeout = number(timeout, 'timeout', 0, 7200)
    maximum = number(maximum, 'maximum GPU hours', 0)
    generation_actual = number(generation_actual, 'generation GPU hours', 0, 4)
    require(timeout > 0 and maximum > 0, 'Budgets must be positive')
    reserve = len(cards) * timeout / 3600
    effective = min(maximum, 8.0, 12.0 - generation_actual)
    require(reserve <= effective and generation_actual + reserve <= 12, 'Scoring reservation exceeds 8/12 GPU-hour budget')
    return cards, reserve, effective


def validate_run(run_dir, decode=True):
    run_dir = Path(run_dir).resolve()
    require(run_dir.is_dir(), '--run-dir must exist')
    summary = read_json(run_dir / 'probe_summary.json')
    require(summary['status'] == 'completed' and type(summary['expected_jobs']) is int and summary['expected_jobs'] == 24
            and type(summary['completed_outputs']) is int and summary['completed_outputs'] == 24 and not summary['failures'],
            'Generation must be completed with exactly 24 verified outputs')
    actual = validate_charge(summary['workers'], summary['gpu_hours'], summary['reserved_generation_gpu_hours'],
                             summary['effective_generation_cap_gpu_hours'], 'generation')
    require(actual <= 4 and number(summary['reserved_generation_gpu_hours'], 'generation reserved', 0, 4) <= 4
            and number(summary['effective_generation_cap_gpu_hours'], 'generation effective', 0, 4) <= 4,
            'Generation exceeds hard 4 GPU-hour budget')
    frozen = run_dir / 'jobs.frozen.jsonl'
    checked_hash(frozen, summary['manifest_sha256'], 'frozen jobs')
    manifest = read_json(run_dir / 'probe_manifest.json')
    require(manifest['protocol'] == 'internal-dev-teacher-cfg-v1' and manifest['jobs_frozen_sha256'] == summary['manifest_sha256'],
            'Wrong or changed frozen protocol')
    require(manifest['label_protocol'] == 'agent_authored_conservative_dense_internal_dev_v1'
            and manifest['use_scope'] == 'training_visible_internal_diagnostic_only'
            and manifest['teacher_human_review'] == 'pending', 'Corrected internal-dev label scope/review disclosure missing')
    frozen_sources = manifest['source_sha256']
    require(isinstance(frozen_sources, dict) and {'run_teacher_probe.py', 'gen_jobs.py'} <= frozen_sources.keys(),
            'Generation source code freeze missing')
    generation_hashes = {}
    for name, digest in frozen_sources.items():
        require(Path(name).name == name, 'Generation source snapshot name escapes code directory')
        source = run_dir / 'code' / name
        checked_hash(source, digest, 'frozen generation source')
        generation_hashes[str(source)] = digest
    framework_root = run_dir / 'framework'
    require(isinstance(manifest['framework_root'], str) and Path(manifest['framework_root']).is_absolute()
            and Path(manifest['framework_root']).resolve() == framework_root
            and framework_root.is_dir() and not framework_root.is_symlink(), 'Frozen framework root must be run-dir/framework')
    framework_sources = manifest['framework_source_sha256']
    require(isinstance(framework_sources, dict) and 'diffsynth/__init__.py' in framework_sources, 'Frozen framework mapping missing/incomplete')
    for relative, digest in framework_sources.items():
        require(isinstance(relative, str) and relative and '://' not in relative and '\\' not in relative, 'Invalid framework snapshot path')
        parts = relative.split('/')
        require(parts[0] == 'diffsynth' and all(part not in ('', '.', '..', '__pycache__', '.git') for part in parts)
                and not relative.endswith('.pyc'), 'Framework snapshot path escapes or names excluded runtime cache')
        source = framework_root / relative
        require(source.resolve().is_relative_to(framework_root) and not any(parent.is_symlink() for parent in
                (source, *source.parents) if parent.is_relative_to(framework_root)), 'Framework snapshot symlink/path escapes freeze')
        checked_hash(source, digest, 'frozen generation framework')
        generation_hashes[str(source)] = digest
    checked_hash(run_dir / 'jobs.input.jsonl', manifest['jobs_input_sha256'], 'original job manifest')
    generation_hashes[str(run_dir / 'jobs.input.jsonl')] = manifest['jobs_input_sha256']
    for key in ('reserved_generation_gpu_hours', 'effective_generation_cap_gpu_hours'):
        require(number(manifest[key], key) == number(summary[key], key), 'Generation summary/manifest budget differs')
    cards, reserved_generation = manifest['gpus'], number(manifest['timeout_seconds'], 'generation timeout', 0, 3600)
    require(isinstance(cards, list) and 1 <= len(cards) <= 4 and len(cards) == len(set(cards))
            and set(cards) == {row['gpu'] for row in summary['workers']}, 'Generation GPU assignment differs')
    require(math.isclose(len(cards) * reserved_generation / 3600, summary['reserved_generation_gpu_hours'], abs_tol=1e-10),
            'Generation reservation does not match card count times timeout')
    require(manifest['zero_disabled'] is True and manifest['checkpoint_sha256'] == {'m_full1_a07': MERGE_SHA256},
            'Frozen sampling/checkpoint protocol differs')
    jobs = [json.loads(line) for line in frozen.read_text().splitlines() if line.strip()]
    require(len(jobs) == 24 and jobs == manifest['jobs'], 'Frozen manifest must bind all 24 jobs')
    records = {}
    for record in summary['outputs']:
        require(record['id'] not in records, 'Duplicate output records')
        records[record['id']] = record
    require(len(records) == 24, 'Missing generation output records')
    cases, ids, outputs, hash_cache = defaultdict(list), set(), set(), {}
    for job in jobs:
        require(job['id'] not in ids and job['id'] in records, 'Duplicate/missing job ID')
        ids.add(job['id'])
        require(type(job['cfg']) in (int, float) and (job['model'], job['cfg']) in ARMS, 'Wrong model/CFG cell')
        require(type(job['seed']) is int and job['seed'] == 0, 'P1 fixed seed must be 0')
        require(type(job['shift']) in (int, float) and job['shift'] == 5 and type(job['steps']) is int and job['steps'] == 50,
                'P1 fixed shift/steps differ')
        require(job.get('cfg_zero_star') is False and type(job.get('cfg_zero_steps')) is int and job['cfg_zero_steps'] == 0,
                'P1 Zero must be off')
        for key, fixed in (('width', 1280), ('height', 704), ('num_frames', 121), ('fps', 24), ('quality', 5)):
            require(type(job.get(key)) is int and job[key] == fixed, 'Fixed protocol differs: ' + key)
        require(job['model'] != 'base' or job['ckpt'] == 'NONE', 'Base checkpoint differs')
        require(job['model'] != 'm_full1_a07' or job['ckpt_sha256'] == MERGE_SHA256, 'Fixed A07 checkpoint differs')
        require(isinstance(job['prompt'], str) and len(job['prompt'].split()) >= 80, 'Complete corrected dense prompt required')
        review = job['prompt_review']
        require(review['prompt_sha256'] == hashlib.sha256(job['prompt'].encode()).hexdigest(), 'Corrected prompt hash differs')
        require(review.get('reviewed_by') and review.get('reviewed_at'), 'External prompt review missing')
        for path, digest, label in ((job['image'], job['image_sha256'], 'conditioning source'),
                                    (review['csv_path'], review['csv_sha256'], 'original CSV'),
                                    (review['preserved_csv'], review['csv_sha256'], 'preserved original CSV')):
            require('://' not in str(path), 'Inputs must be local')
            if str(path) not in hash_cache:
                hash_cache[str(path)] = sha256(path)
            require(hash_cache[str(path)] == digest, 'SHA256 changed: ' + label)
        gt = job.get('gt', job['image'] if Path(job['image']).suffix.lower() == '.mp4' else None)
        require(isinstance(gt, str) and Path(gt).suffix.lower() == '.mp4' and '://' not in gt, 'Original GT video required, not an input still image')
        gt_hash = job.get('gt_sha256', job['image_sha256'] if gt == job['image'] else None)
        if gt not in hash_cache:
            hash_cache[gt] = sha256(gt)
        require(hash_cache[gt] == gt_hash, 'Original GT video hash differs')
        target = Path(job['out']).resolve()
        require(target.is_relative_to(run_dir / 'videos') and target not in outputs, 'Duplicate/escaping output path')
        outputs.add(target)
        record = records[job['id']]
        require(record['out'] == str(target), 'Output path differs from frozen assignment')
        checked_hash(target, record['sha256'], 'generated video ' + job['id'])
        decoded = record['decode']
        require(all(type(decoded[key]) is int and decoded[key] == fixed for key, fixed in
                    (('frames', 121), ('width', 1280), ('height', 704)))
                and math.isclose(number(decoded['fps'], 'recorded FPS'), 24, abs_tol=0.01), 'Generation decode record differs')
        if decode:
            decode_video(target)
        job = dict(job, gt=gt, gt_sha256=gt_hash, output_sha256=record['sha256'])
        cases[job['case_id']].append(job)
    require(len(cases) == 6 and ids == set(records), 'Need six complete cases and 24 exact outputs')
    sources, images = set(), set()
    for rows in cases.values():
        require(len(rows) == 4 and {(r['model'], r['cfg']) for r in rows} == ARMS, 'Case missing/duplicate four cells')
        first = rows[0]
        require(isinstance(first['canonical_source'], str) and first['canonical_source'].strip(), 'Missing canonical parent source')
        require(first['canonical_source'] not in sources and first['image_sha256'] not in images, 'Six independent canonical parents required')
        sources.add(first['canonical_source'])
        images.add(first['image_sha256'])
        for key in ('canonical_source', 'image', 'image_sha256', 'gt', 'gt_sha256', 'prompt', 'prompt_review', 'seed'):
            require(all(row[key] == first[key] for row in rows), 'Four-cell pairing changed: ' + key)
    # Snapshot all local protocol sources and original data; finalization checks again.
    hashes = dict(hash_cache, **generation_hashes)
    for name in ('probe_summary.json', 'probe_manifest.json', 'jobs.frozen.jsonl'):
        hashes[str(run_dir / name)] = sha256(run_dir / name)
    for job in (row for rows in cases.values() for row in rows):
        hashes[job['out']] = job['output_sha256']
    return [row for rows in cases.values() for row in rows], actual, hashes


def assert_idle(gpus):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                            check=True, capture_output=True, text=True)
    memory = {parts[0].strip(): int(parts[1].strip()) for line in result.stdout.splitlines()
              if len(parts := line.split(',')) == 2}
    require(all(g in memory and memory[g] < 100 for g in gpus), 'Selected GPUs must each have <100 MiB used')
    # Refuse generation competition even when a pipeline has not allocated memory yet.
    for pid in os.listdir('/proc'):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            cmd = (Path('/proc') / pid / 'cmdline').read_bytes().split(b'\0')
            active = {'gen_jobs.py', 'run_teacher_probe.py', 'eval_val_loss.py', 'score_teacher_probe.py', 'score_pool.py'}
            if not any(os.path.basename(arg.decode(errors='replace')) in active for arg in cmd):
                continue
            env = (Path('/proc') / pid / 'environ').read_bytes().split(b'\0')
            visible = next((e.split(b'=', 1)[1].decode().split(',') for e in env if e.startswith(b'CUDA_VISIBLE_DEVICES=')), None)
            require(visible is not None and not set(gpus).intersection(visible), 'Competing selected/unspecified GPU process PID ' + pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as error:
            raise ValueError('Cannot verify competing process PID ' + pid) from error


def metric_assets():
    files = [DINO_WEIGHT, MUSIQ_WEIGHT, ROOT / 'scripts/gt_ref_metrics.py', Path(__file__).resolve(),
             VBENCH / 'vbench/subject_consistency.py', VBENCH / 'vbench/imaging_quality.py', VBENCH / 'vbench/utils.py']
    files += sorted(DINO_REPO.rglob('*.py'))
    files += [DREAM_CACHE / name for name in ('dino_vitb16_pretrain.pth', 'clip_vitb16_pretrain.pth.tar',
              'open_clip_vitb16_pretrain.pth.tar', 'ensemble_lora/adapter_config.json', 'ensemble_lora/adapter_model.safetensors',
              'checkpoints/dino_vitbase16_pretrain.pth')]
    site = Path(PYTHON).parents[1] / 'lib/python3.12/site-packages'
    files += sorted((site / 'dreamsim').rglob('*.py'))
    files += [site / 'pyiqa/archs/musiq_arch.py']
    require(DINO_REPO.is_dir() and (site / 'dreamsim/model.py').is_file(), 'Local metric source repositories required')
    hashes = {str(path): sha256(path) for path in files}
    require(hashes[str(DINO_WEIGHT)] == hashes[str(DREAM_CACHE / 'checkpoints/dino_vitbase16_pretrain.pth')],
            'Offline DINO backbone must be byte-identical to original DreamSim cached checkpoint')
    for name in ('hubconf.py', 'vision_transformer.py'):
        original = DREAM_CACHE / 'facebookresearch_dino_main' / name
        hashes[str(original)] = sha256(original)
        require(hashes[str(original)] == hashes[str(DINO_REPO / name)], 'Offline DINO architecture differs from original cached source')
    return hashes


def verify_hashes(hashes, deadline=None):
    for path, digest in hashes.items():
        checked_hash(path, digest, path, deadline)


def normalized_imaging(aggregate, raw):
    aggregate = number(aggregate, 'MUSIQ aggregate', 0, 1)
    raw = number(raw, 'MUSIQ per-video raw', 0, 100)
    normalized = raw / 100.0
    require(math.isclose(aggregate, normalized, abs_tol=1e-6), 'Original MUSIQ per-video units must be raw / 100')
    return normalized


def validate_rows(jobs, rows):
    require(len(rows) == 24, 'Need exactly 24 scored rows')
    expected = {j['id']: j for j in jobs}
    require(len(expected) == 24, 'Duplicate job IDs')
    seen, cells, sources = set(), defaultdict(set), {}
    for row in rows:
        require(row['id'] in expected and row['id'] not in seen, 'Duplicate/unexpected scored ID')
        seen.add(row['id'])
        job = expected[row['id']]
        for key in ('case_id', 'canonical_source', 'model', 'cfg'):
            require(row[key] == job[key], 'Scored row binding mismatch: ' + key)
        require(type(row['cfg']) in (int, float), 'CFG cannot be bool')
        for key in METRICS:
            number(row[key], key, 0, 1 if key in ('subject_consistency', 'imaging_quality') else None)
        for key in ('gen_motion', 'gt_motion', 'motion_amount_ratio', 'dur'):
            number(row[key], key, 0)
        require(row['gen_motion'] > 0 and row['gt_motion'] > 0 and row['dur'] > 0, 'Positive motion/duration required')
        require(math.isclose(row['motion_amount_ratio'], row['gen_motion'] / row['gt_motion'], abs_tol=1e-8)
                and math.isclose(row['motion_ratio'], abs(math.log(row['motion_amount_ratio'])), abs_tol=1e-8), 'Original motion formula differs')
        cells[row['case_id']].add((row['model'], row['cfg']))
        sources[row['case_id']] = row['canonical_source']
    require(seen == set(expected) and len(cells) == 6 and all(cell == ARMS for cell in cells.values())
            and len(set(sources.values())) == 6, 'Need six independent sources with four complete scored cells')


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low = int(index)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low)


def aggregate(jobs, rows):
    validate_rows(jobs, rows)
    cases = sorted({row['case_id'] for row in rows})
    lookup = {(r['case_id'], r['model'], r['cfg']): r for r in rows}
    rng = random.Random(BOOTSTRAP_SEED)
    samples = [[rng.randrange(6) for _ in cases] for _ in range(BOOTSTRAP_REPLICATES)]
    paired, model_results = {}, {}
    for model in MODELS:
        deltas = {key: [lookup[(case, model, 3)][key] - lookup[(case, model, 5)][key] for case in cases] for key in METRICS}
        paired[model] = deltas
        metrics = {}
        for key, values in deltas.items():
            draws = [math.fsum(values[i] for i in indices) / 6 for indices in samples]
            metrics[key] = {'cfg3_minus_cfg5': math.fsum(values) / 6, 'per_source_delta': values,
                            'direction': DIRECTIONS[key], 'positive_sources': sum(x > 0 for x in values),
                            'source_bootstrap_ci95': [percentile(draws, .025), percentile(draws, .975)]}
        primary = ((metrics['subject_consistency']['cfg3_minus_cfg5'] >= .003 and metrics['subject_consistency']['positive_sources'] >= 4)
                   or (metrics['imaging_quality']['cfg3_minus_cfg5'] >= .005 and metrics['imaging_quality']['positive_sources'] >= 4))
        guards = {key: metrics[key]['cfg3_minus_cfg5'] <= limit for key, limit in
                  (('dreamsim', .005), ('dreamsim_last', .010), ('motion_ratio', .05))}
        motion_cases = []
        for case in cases:
            candidate, baseline = lookup[(case, model, 3)], lookup[(case, model, 5)]
            meaningful = baseline['gt_motion'] - 1e-3 >= 1.0
            # All four outputs have the same timing, hence the same sampled GT motion.
            require(math.isclose(candidate['gt_motion'], baseline['gt_motion'], abs_tol=1e-8), 'Paired GT motion differs')
            relative = candidate['gen_motion'] / baseline['gen_motion']
            motion_cases.append({'case_id': case, 'gt_motion': baseline['gt_motion'], 'gt_motion_meaningful': meaningful,
                                 'cfg3_over_cfg5_gen_motion': relative, 'passed': not meaningful or relative >= .7})
        guards['anti_static_motion'] = all(row['passed'] for row in motion_cases)
        passed = primary and all(guards.values())
        model_results[model] = {'metrics': metrics, 'primary_passed': primary, 'guards': guards,
                               'motion_cases': motion_cases, 'machine_metrics': 'PASS' if passed else 'FAIL',
                               'decision': 'ready_for_review' if passed else 'reject_or_tradeoff',
                               'human_semantic_action_review': 'pending', 'promotion_allowed': False, 'teacher_passed': False}
    did = {}
    for key in METRICS:
        values = [a - b for a, b in zip(paired[MODELS[1]][key], paired[MODELS[0]][key])]
        draws = [math.fsum(values[i] for i in indices) / 6 for indices in samples]
        did[key] = {'a07_delta_minus_base_delta': math.fsum(values) / 6, 'per_source_delta': values,
                    'source_bootstrap_ci95': [percentile(draws, .025), percentile(draws, .975)], 'causal_claim': False}
    return {'models': model_results, 'difference_in_differences': did, 'case_order': cases,
            'bootstrap': {'unit': 'independent canonical parent source', 'sources': 6, 'replicates': BOOTSTRAP_REPLICATES,
                          'rng_seed': BOOTSTRAP_SEED, 'across_seed_uncertainty': False, 'human_score_variance_included': False},
            'teacher_gate': False, 'promotion_allowed': False, 'scope': 'training-visible internal-dev diagnostic; not public benchmark selection'}


def offline_loaders():
    import socket
    import torch
    from unittest.mock import patch
    original = torch.hub.load
    def local_dino(repo, model, *args, **kwargs):
        require(repo == 'facebookresearch/dino:main' and model == 'dino_vitb16', 'Unapproved implicit TorchHub model')
        net = original(str(DINO_REPO), 'dino_vitb16', source='local', pretrained=False)
        net.load_state_dict(torch.load(str(DINO_WEIGHT), map_location='cpu', weights_only=True), strict=True)
        return net
    def blocked(*args, **kwargs):
        raise RuntimeError('Network access is forbidden for teacher probe scoring')
    from contextlib import ExitStack
    stack = ExitStack()
    stack.enter_context(patch.object(torch.hub, 'load', local_dino))
    stack.enter_context(patch.object(torch.hub, 'download_url_to_file', blocked))
    stack.enter_context(patch.object(socket.socket, 'connect', blocked))
    stack.enter_context(patch.object(socket.socket, 'connect_ex', blocked))
    return stack


def worker(scoring, kind, index, assignment_sha):
    require(os.environ.get('TEACHER_SCORE_PARENT_PID') == str(os.getppid()), 'Internal worker requires its live parent')
    path = scoring / (kind + '_assignment_' + str(index) + '.json')
    checked_hash(path, assignment_sha, 'worker assignment')
    assignment = read_json(path)
    require(assignment['gpu'] == os.environ.get('CUDA_VISIBLE_DEVICES'), 'Worker GPU assignment changed')
    report = {'status': 'running', 'kind': kind, 'gpu': assignment['gpu'], 'pid': os.getpid(), 'rows': []}
    started = time.perf_counter()
    report_path = scoring / (kind + '_worker_' + str(index) + '.json')
    try:
        import torch
        torch.set_num_threads(8)
        sys.path.insert(0, str(VBENCH))
        verify_hashes(assignment['asset_sha256'])
        if kind == 'cpu':
            from dreamsim import dreamsim
            import gt_ref_metrics as gt_ref
            torch.set_num_threads(8)  # gt_ref import has its original thread default.
            with offline_loaders():
                model, preprocess = dreamsim(pretrained=True, device='cpu', cache_dir=str(DREAM_CACHE))
            model.eval()
            gt_cache = {}
        else:
            from vbench.subject_consistency import subject_consistency
            from vbench.imaging_quality import technical_quality
            from pyiqa.archs.musiq_arch import MUSIQ
            with offline_loaders():
                dino = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16').eval().to('cuda')
                musiq = MUSIQ(pretrained_model_path=str(MUSIQ_WEIGHT)).eval().to('cuda')
        for job in assignment['jobs']:
            t0 = time.perf_counter()
            checked_hash(job['out'], job['output_sha256'], 'worker output')
            row = {key: job[key] for key in ('id', 'case_id', 'canonical_source', 'model', 'cfg')}
            if kind == 'cpu':
                checked_hash(job['gt'], job['gt_sha256'], 'original GT video')
                if job['case_id'] not in gt_cache:
                    gt_cache[job['case_id']] = gt_ref.read(job['gt'])
                gf, gfps = gt_cache[job['case_id']]
                vf, vfps = gt_ref.read(job['out'])
                require(len(gf) > 1 and gfps > 0 and len(vf) == 121 and math.isclose(vfps, 24, abs_tol=.01), 'Invalid GT/generated video timing')
                dur = min(len(gf) / gfps, len(vf) / vfps) - 1e-3
                a = [frame.resize((256, 144)) for frame in gt_ref.sample(gf, gfps, dur)]
                b = [frame.resize((256, 144)) for frame in gt_ref.sample(vf, vfps, dur)]
                require(len(a) == len(b) == 12 and dur > 0, 'Original shared-duration K12 sampling failed')
                with torch.no_grad():
                    distances = model(torch.cat([preprocess(x) for x in a]), torch.cat([preprocess(x) for x in b])).flatten().tolist()
                require(len(distances) == 12 and all(type(x) in (int, float) and math.isfinite(x) for x in distances), 'Invalid DreamSim distances')
                gm, vm = gt_ref.motion(a), gt_ref.motion(b)
                row.update(dreamsim=math.fsum(distances) / 12, dreamsim_last=distances[-1], gen_motion=vm, gt_motion=gm,
                           motion_amount_ratio=vm / gm, motion_ratio=abs(math.log(vm / gm)), dur=dur)
            else:
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad(), offline_loaders():
                    subject, subject_rows = subject_consistency(dino, [job['out']], 'cuda', False)
                    imaging, imaging_rows = technical_quality(musiq, [job['out']], 'cuda')
                require(len(subject_rows) == len(imaging_rows) == 1 and subject_rows[0]['video_path'] == imaging_rows[0]['video_path'] == job['out'],
                        'Original VBench per-video binding differs')
                subject = number(subject, 'DINO aggregate', 0, 1)
                require(math.isclose(subject, number(subject_rows[0]['video_results'], 'DINO per-video', 0, 1), abs_tol=1e-6), 'DINO result mismatch')
                row.update(subject_consistency=subject, imaging_quality=normalized_imaging(imaging, imaging_rows[0]['video_results']),
                           imaging_quality_raw=imaging_rows[0]['video_results'])
                torch.cuda.synchronize()
                row.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
            row['seconds'] = time.perf_counter() - t0
            report['rows'].append(row)
            atomic_json(report_path, report)
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        traceback.print_exc()
    finally:
        report['seconds'] = time.perf_counter() - started
        atomic_json(report_path, report)
    return 0 if report['status'] == 'completed' else 1


def stop_workers(workers, grace):
    stopping = []
    for info in workers:
        # Signal our group even if its leader exited: decoder children can remain.
        stopping.append(info)
        try:
            os.killpg(info['process'].pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    until = time.perf_counter() + grace
    for info in workers:
        try:
            info['process'].wait(timeout=max(0, until - time.perf_counter()))
        except subprocess.TimeoutExpired:
            pass
    for info in stopping:
        # A reaped leader can still have its own live decoder descendants.
        try:
            os.killpg(info['process'].pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for info in workers:
        info['process'].wait()
        info.setdefault('ended', time.perf_counter())


def launch_workers(scoring, kind, assignments, deadline):
    workers, failure = [], None
    grace = min(2.0, max(0, deadline - time.perf_counter()) / 10)
    try:
        for index, assignment in enumerate(assignments):
            require(time.perf_counter() < deadline - grace - .1, 'Scoring hard timeout reached before launch')
            path = scoring / (kind + '_assignment_' + str(index) + '.json')
            atomic_json(path, assignment)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=assignment['gpu'], TEACHER_SCORE_PARENT_PID=str(os.getpid()),
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', THREADS='8', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
            env['PYTHONPATH'] = str(ROOT / 'scripts') + os.pathsep + str(VBENCH) + os.pathsep + env.get('PYTHONPATH', '')
            log_path = scoring / (kind + '_worker_' + str(index) + '.log')
            log = log_path.open('x')
            started = time.perf_counter()
            try:
                proc = subprocess.Popen([PYTHON, str(Path(__file__).resolve()), '--run-dir', str(scoring.parent), '--worker-kind', kind,
                                         '--worker-index', str(index), '--assignment-sha', sha256(path)],
                                        cwd=str(VBENCH), env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
            except BaseException:
                log.close()
                raise
            workers.append({'index': index, 'gpu': assignment['gpu'], 'process': proc, 'started': started, 'log': log, 'log_path': str(log_path)})
            require(all(info['process'].poll() in (None, 0) for info in workers), 'Worker failed during startup')
        while True:
            now = time.perf_counter()
            running = False
            for info in workers:
                rc = info['process'].poll()
                if rc is not None:
                    info.setdefault('ended', now)
                    require(rc == 0, 'Scoring worker exited nonzero; inspect ' + info['log_path'])
                else:
                    running = True
            if not running:
                break
            require(now < deadline - grace - .1, 'Scoring reached hard timeout (shutdown grace is inside budget)')
            time.sleep(.02)
    except BaseException as error:
        failure = type(error).__name__ + ': ' + str(error)
    finally:
        stop_workers(workers, grace)
        for info in workers:
            info['log'].close()
    resources = [{'index': info['index'], 'gpu': info['gpu'], 'pid': info['process'].pid, 'returncode': info['process'].returncode,
                  'seconds': info['ended'] - info['started'], 'gpu_hours': (info['ended'] - info['started']) / 3600 if kind == 'gpu' else 0,
                  'log': info['log_path']} for info in workers]
    return resources, failure


def load_report_rows(scoring, kind, resources, jobs):
    require(resources and all(r['returncode'] == 0 for r in resources), 'Missing/nonzero scoring workers')
    rows = []
    for resource in resources:
        report = read_json(scoring / (kind + '_worker_' + str(resource['index']) + '.json'))
        require(report['status'] == 'completed' and report['kind'] == kind and report['gpu'] == resource['gpu'], 'Scoring worker report incomplete')
        rows.extend(report['rows'])
    require(len(rows) == 24 and len({r['id'] for r in rows}) == 24 and {r['id'] for r in rows} == {j['id'] for j in jobs},
            'Missing/duplicate scoring worker outputs')
    return {row['id']: row for row in rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--timeout-seconds', type=float, default=7200)
    parser.add_argument('--max-gpu-hours', type=float, default=8)
    parser.add_argument('--worker-kind', choices=('cpu', 'gpu'), help=argparse.SUPPRESS)
    parser.add_argument('--worker-index', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--assignment-sha', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    scoring = run_dir / 'scoring'
    if args.worker_kind is not None:
        require(type(args.worker_index) is int and args.worker_index >= 0 and args.assignment_sha, 'Invalid internal worker arguments')
        return worker(scoring, args.worker_kind, args.worker_index, args.assignment_sha)
    require(not scoring.exists() and not scoring.is_symlink(), 'scoring directory already exists; no resume or overwrite')
    # Check invalid budgets before file hashing or touching a GPU.
    budget(args.gpus, args.timeout_seconds, args.max_gpu_hours, 0)
    jobs, generation_actual, input_hashes = validate_run(run_dir)
    cards, reserved, effective = budget(args.gpus, args.timeout_seconds, args.max_gpu_hours, generation_actual)
    assets = metric_assets()
    assert_idle(cards)
    scoring.mkdir(exist_ok=False)
    def terminate_parent(signum, frame):
        raise KeyboardInterrupt('Scoring parent received signal ' + str(signum))
    previous_term = signal.signal(signal.SIGTERM, terminate_parent)
    started = time.perf_counter()
    deadline = started + args.timeout_seconds
    result = {'status': 'failed', 'expected_jobs': 24, 'completed_rows': 0, 'generation_gpu_hours': generation_actual,
              'reserved_scoring_gpu_hours': reserved, 'effective_scoring_cap_gpu_hours': effective, 'experiment_cap_gpu_hours': 12,
              'metric_asset_sha256': assets, 'frozen_input_sha256': input_hashes, 'cpu_workers': [], 'gpu_workers': [],
              'actual_scoring_gpu_hours': 0, 'training_performed': False, 'public_benchmark_selection': False,
              'teacher_passed': False, 'promotion_allowed': False, 'human_semantic_action_review': 'pending', 'failures': []}
    try:
        atomic_json(scoring / 'score_manifest.json', dict(result, jobs=jobs, gpus=cards, timeout_seconds=args.timeout_seconds,
                    algorithm={'subject': 'original subject_consistency(model,[video],cuda,False)', 'imaging': 'original technical_quality default longer; per_video /100',
                               'gt': 'original read/sample/motion; K12 shared duration, skip t0; resize256x144; DreamSim CPU preprocess'}))
        cpu_resources, error = launch_workers(scoring, 'cpu', [{'gpu': '', 'jobs': jobs, 'asset_sha256': assets}], deadline)
        result['cpu_workers'] = cpu_resources
        require(not error, error or 'CPU scoring failed')
        cpu_rows = load_report_rows(scoring, 'cpu', cpu_resources, jobs)
        assert_idle(cards)
        assignments = [{'gpu': gpu, 'jobs': jobs[index::len(cards)], 'asset_sha256': assets} for index, gpu in enumerate(cards)]
        gpu_resources, error = launch_workers(scoring, 'gpu', assignments, deadline)
        result['gpu_workers'] = gpu_resources
        result['actual_scoring_gpu_hours'] = math.fsum(row['gpu_hours'] for row in gpu_resources)
        require(not error, error or 'GPU scoring failed')
        validate_charge(gpu_resources, result['actual_scoring_gpu_hours'], reserved, effective, 'GPU process scoring')
        require(time.perf_counter() <= deadline, 'Scoring wall-clock budget exceeded')
        gpu_rows = load_report_rows(scoring, 'gpu', gpu_resources, jobs)
        rows = []
        for job in jobs:
            cpu, gpu = cpu_rows[job['id']], gpu_rows[job['id']]
            for key in ('id', 'case_id', 'canonical_source', 'model', 'cfg'):
                require(cpu[key] == gpu[key], 'CPU/GPU row mismatch')
            rows.append(dict(cpu, **{k: v for k, v in gpu.items() if k not in cpu}, cpu_seconds=cpu['seconds'], gpu_seconds=gpu['seconds']))
        diagnostics = aggregate(jobs, rows)
        verify_hashes(input_hashes, deadline)
        verify_hashes(assets, deadline)
        atomic_json(scoring / 'score_rows.json', rows)
        atomic_json(scoring / 'paired_diagnostics.json', diagnostics)
        result.update(status='completed', completed_rows=24, diagnostics=diagnostics,
                      score_rows_sha256=sha256(scoring / 'score_rows.json'))
    except BaseException as error:
        result['failures'].append(type(error).__name__ + ': ' + str(error))
        traceback.print_exc()
    finally:
        wall = time.perf_counter() - started
        # Reserve every selected card throughout CPU setup, scoring, and verification.
        # This conservative allocated ledger also charges failed attempts in full.
        result['wall_seconds'] = wall
        result['gpu_process_gpu_hours'] = math.fsum(row['gpu_hours'] for row in result['gpu_workers'])
        result['allocated_gpu_workers'] = [{'gpu': gpu, 'seconds': wall, 'gpu_hours': wall / 3600,
                                           'returncode': 0 if result['status'] == 'completed' else 1} for gpu in cards]
        result['actual_scoring_gpu_hours'] = len(cards) * wall / 3600
        result['actual_experiment_gpu_hours'] = generation_actual + result['actual_scoring_gpu_hours']
        result['allocation_includes_cpu_setup_and_final_verification'] = True
        if wall > args.timeout_seconds or result['actual_scoring_gpu_hours'] > reserved or result['actual_scoring_gpu_hours'] > effective \
                or result['actual_experiment_gpu_hours'] > 12:
            result['status'] = 'failed'
            result['failures'].append('Actual allocated GPU charge / wall-clock exceeds frozen budget')
        atomic_json(scoring / 'score_summary.json', result)
        signal.signal(signal.SIGTERM, previous_term)
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print('Scoring refused: ' + str(error), file=sys.stderr)
        raise SystemExit(1)
