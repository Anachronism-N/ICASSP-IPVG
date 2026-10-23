"""Versioned same-cohort Stand-In attention experiment; no automatic winner or action GT."""
import argparse
from collections import Counter
import copy
import hashlib
import math
from pathlib import Path
import time

from ipvg.common import fingerprint, load_json, sha256_file, write_json
from ipvg.data import read_manifest
from ipvg.experiments import _source_hash, run_process
from analyze_native_baselines import generation_evidence, identity_coverage_summary
from evaluation_panel_report import (CORE_METRICS, finite_values, paired_comparisons,
                                     report_records, scorer_signature)
from prepare_native_action_benchmark import (ACTION_MANIFEST_SHA256, BASELINE_HISTORY_RUN_ID,
                                             PARTITION_GATES, PROMPTS, partition_rows)

VERSION = 'standin_partition_20261023_r001'
METRICS = ('cur', 'gme', 'quality', 'extended')
PHASES = ('pilot_portrait', 'pilot_fullbody', 'main')
GROUPS = ('baseline', 'partition')
ROOT = Path('/dockerdata/ipvg')


def scorer_argv(metric, shard, host, work_dir, repo_root, eval_root):
    if metric not in METRICS or shard not in (0, 1) or host not in ('launcher', 'worker'):
        raise ValueError('A frozen metric, shard and host are required')
    folder = work_dir / 'evaluation' / f'{metric}_shard{shard}_r001'
    common = ['--manifest', str(work_dir / f'quality_shard{shard}.json'), '--output', str(folder / 'scores.json')]
    python = str(eval_root / 'venv/bin/python')
    if metric == 'gme':
        return [python, str(repo_root / 'scripts/gme_text_eval.py'), *common,
                '--model', str(eval_root / 'gme'), '--model-lock', str(eval_root / 'gme.lock.json')]
    if metric == 'cur':
        return [python, str(repo_root / 'scripts/curricularface_eval.py'), *common,
                '--upstream', str(eval_root / 'ConsisID_source'),
                '--weights', str(eval_root / 'cur/face_encoder/glint360k_curricular_face_r101_backbone.bin'),
                '--face-model-dir', str(ROOT / 'models/identity'), '--model-lock', str(eval_root / 'cur.lock.json')]
    vbench = '/dockerdata/wm_eval/VBench' if host == 'launcher' else str(eval_root / 'VBench_clean')
    cache = '/dockerdata/vbench_cache'
    if metric == 'extended':
        return ['/dockerdata/venv_vbench/bin/python', str(repo_root / 'scripts/vbench_extended_eval.py'),
                *common, '--vbench', vbench, '--cache', cache, '--model-lock', str(eval_root / 'vbench.lock.json')]
    return ['/dockerdata/venv_vbench/bin/python', str(repo_root / 'scripts/mm2026_quality_eval.py'),
            *common, '--vbench', vbench, '--cache', cache, '--amt', cache + '/amt_model/amt-s.pth',
            '--musiq', cache + '/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth']


def frozen_ledger(work_dir):
    ledger = load_json(work_dir / 'ledger.json')
    expected = {(c, p) for c in ('native', 'partition') for p in PHASES}
    if (ledger.get('benchmark_version') != VERSION or len(ledger['runs']) != 6
            or {(e.get('condition'), e['phase']) for e in ledger['runs']} != expected
            or ledger.get('baseline_history_run_id') != BASELINE_HISTORY_RUN_ID
            or ledger.get('gates') != PARTITION_GATES or ledger.get('retry_count') != 0
            or not ledger.get('pilot_before_main_required')
            or ledger.get('generation_reserved_gpu_hours') != 5
            or ledger.get('evaluation_reserved_gpu_hours') != 2
            or ledger.get('single_engineering_repair_allowance_gpu_hours') != .5
            or ledger.get('new_hard_cap_gpu_hours') != 8
            or ledger.get('original_overall_cap_gpu_hours') != 51.5 or ledger.get('prior_gpu_hours_upper') != 5.31):
        raise ValueError('Partition campaign/threshold/budget definition changed')
    for e in ledger['runs']:
        main = e['phase'] == 'main'
        if (e['arm'] != 'standin' or e['count'] != (16 if main else 1)
                or e['reserved_gpu_hours'] != (2 if main else .25)
                or e['run_id'] != f"standin_partition_{e['condition']}_{e['phase']}_20261023_r001"
                or sha256_file(work_dir / 'configs' / (e['run_id'] + '.json')) != e['config_sha256']):
            raise ValueError('Frozen generation entry differs')
    rows, pilots = partition_rows(work_dir / 'manifest.jsonl')
    if (sha256_file(work_dir / 'manifest.jsonl') != ledger['eval_manifest_sha256']
            or ledger['original_manifest_sha256'] != ACTION_MANIFEST_SHA256
            or sha256_file(ledger['original_manifest']) != ACTION_MANIFEST_SHA256
            or ledger['pilot_sample_ids'] != {k: r['sample_id'] for k, r in pilots.items()}):
        raise ValueError('Frozen original cohort or deterministic pilots changed')
    return ledger, rows, pilots


def structure_evidence(profile, condition):
    """Validate owner-wrapper observations, never infer evidence from the chosen config."""
    mode = 'native' if condition == 'native' else 'partition_balanced'
    if profile.get('attention_mode') != mode:
        raise ValueError('Observed attention mode differs')
    for sample in profile['samples']:
        evidence = sample['attention_structure']
        expected = {'mode': mode, 'block_count': 40, 'joint_video_reads': 1600,
                    'ip_self_reads': 40, 'plain_reads': 0,
                    'kernel_calls': 1640 if condition == 'native' else 3240,
                    'partition_reads': 0 if condition == 'native' else 1600}
        if any(evidence.get(k) != v for k, v in expected.items()):
            raise ValueError('Attention structure counts differ')
        blocks = evidence['blocks']
        if len(blocks) != 40 or {b['block_index'] for b in blocks} != set(range(40)):
            raise ValueError('All forty attention owners must be observed')
        for block in blocks:
            if (block['joint_video_reads'] != 40 or block['ip_self_reads'] != 1 or block['plain_reads'] != 0
                    or block['kernel_calls'] != (41 if condition == 'native' else 81)
                    or block['partition_reads'] != (0 if condition == 'native' else 40)):
                raise ValueError('Per-block attention reads differ')


def verify_run_evidence(folder, entry, ledger, runs_dir, work_dir=None):
    """Verify one completed pilot/main; no other new generation runs need to exist."""
    folder = Path(folder); runs_dir = Path(runs_dir)
    work_dir = Path(work_dir) if work_dir is not None else Path(ledger['eval_manifest']).parent
    checked_ledger, rows, pilots = frozen_ledger(work_dir)
    if ledger != checked_ledger or entry not in ledger['runs'] or folder.resolve() != (runs_dir / entry['run_id']).resolve():
        raise ValueError('Verify only the exact frozen execution entry')
    historical_folder = runs_dir / BASELINE_HISTORY_RUN_ID
    for name, digest in ledger['baseline_history_file_sha256'].items():
        if sha256_file(historical_folder / name) != digest:
            raise ValueError('Historical native evidence changed')
    history, template, history_profile, _ = generation_evidence(historical_folder, ledger['baseline_history_entry'])
    expected_hashes = ledger['baseline_history_output_sha256']
    if (history['output_sha256'] != expected_hashes
            or set(expected_hashes) != {r['sample_id'] + '.mp4' for r in rows}
            or sha256_file(Path(ledger['prior_work_dir']) / 'ledger.json') != ledger['prior_ledger_sha256']):
        raise ValueError('Historical native cohort/output ledger changed')
    result, config, profile, cost = generation_evidence(folder, entry)
    if result.get('manifest_sha256') != entry['manifest_sha256']:
        raise ValueError('Execution manifest hash differs from ledger')
    if config != load_json(work_dir / 'configs' / (entry['run_id'] + '.json')):
        raise ValueError('Execution configuration differs from frozen configuration')
    expected_config = copy.deepcopy(template)
    expected_config.update(run_id=config['run_id'], manifest=config['manifest'],
                           resources=config['resources'], hypothesis=config['hypothesis'])
    expected_config['parameters']['attention_mode'] = 'native' if entry['condition'] == 'native' else 'partition_balanced'
    expected_config['parameters']['expected_attention_backend'] = 'FA2'
    if config != expected_config:
        raise ValueError('Only attention mode/backend lock and experiment bookkeeping may change')
    wanted_resources = {'gpus': 1, 'timeout_s': 7200 if entry['phase'] == 'main' else 900,
                        'max_gpu_hours': entry['reserved_gpu_hours']}
    if config['resources'] != wanted_resources or result['resources'] != wanted_resources:
        raise ValueError('Execution resource controls differ')
    wanted_rows = rows if entry['phase'] == 'main' else [pilots[entry['phase']]]
    if read_manifest(folder / 'manifest.jsonl') != wanted_rows:
        raise ValueError('Generation prompt/reference/actor/action inputs changed')
    controls = ('model_lock_sha256', 'standin_checkpoint_sha256', 'matched_ip_tensors',
                'code_revision', 'native_vram_management', 'attention_backend', 'preprocessor')
    if profile.get('attention_backend') != 'FA2' or any(profile.get(k) != history_profile.get(k) for k in controls):
        raise ValueError('Native model/preprocessing/FA2 backend controls changed')
    structure_evidence(profile, entry['condition'])
    observations = {s['sample_id']: s for s in profile['samples']}
    for row in wanted_rows:
        sample = observations[row['sample_id']]
        expected_seed = (42 + int(hashlib.sha256(row['sample_id'].encode()).hexdigest()[:8], 16)) % 2**31
        historical_sample = next(s for s in history_profile['samples'] if s['sample_id'] == row['sample_id'])
        if (sample['seed'] != expected_seed or sample['generated_frames'] != 49
                or sample['reference_pixels_sha256'] != historical_sample['reference_pixels_sha256']):
            raise ValueError('Original seed/reference preprocessing differs')
    if set(result['output_sha256']) != {r['sample_id'] + '.mp4' for r in wanted_rows}:
        raise ValueError('Generation output hashes incomplete')
    if entry['condition'] == 'native':
        if any(result['output_sha256'][r['sample_id'] + '.mp4'] != expected_hashes[r['sample_id'] + '.mp4'] for r in wanted_rows):
            raise ValueError('Native control media must exactly match historical bytes')
        if entry['phase'] != 'main' and result['output_sha256'][wanted_rows[0]['sample_id'] + '.mp4'] != ledger['pilot_historical_output_sha256'][entry['phase']]:
            raise ValueError('Native pilot historical match differs')
    return result, config, profile, cost


def campaign_evidence(work_dir, runs_dir):
    ledger, rows, _ = frozen_ledger(work_dir)
    all_runs = {}; summaries = []; hours = 0
    for entry in ledger['runs']:
        folder = runs_dir / entry['run_id']
        result, config, profile, cost = verify_run_evidence(folder, entry, ledger, runs_dir, work_dir)
        hours += cost
        all_runs[(entry['condition'], entry['phase'])] = (entry, result, profile)
        summaries.append({'run_id': entry['run_id'], 'condition': entry['condition'], 'phase': entry['phase'],
                          'status': result['status'], 'code_commit': result['code_commit'],
                          'source_sha256': result['source_sha256'], 'config_sha256': entry['config_sha256'],
                          'manifest_sha256': entry['manifest_sha256'], 'gpu_hours': cost,
                          'model_revisions': result.get('model_revisions', {}),
                          'model_lock_sha256': profile['model_lock_sha256']})
    pilot_finish = max(r[1]['finished_at'] for key, r in all_runs.items() if key[1] != 'main')
    if any(r[1]['started_at'] < pilot_finish for key, r in all_runs.items() if key[1] == 'main'):
        raise ValueError('Both arms must complete both pilots before any main starts')
    if hours > ledger['generation_reserved_gpu_hours']:
        raise ValueError('Generation budget exceeded')
    return ledger, rows, all_runs, summaries, hours


def freeze(work_dir, runs_dir, repo_root, eval_root):
    destinations = [work_dir / name for name in ('quality_frozen.json', 'quality_shard0.json', 'quality_shard1.json', 'evaluation_frozen.json')]
    if any(p.exists() for p in destinations):
        raise FileExistsError('Every freeze output must be fresh')
    ledger, rows, runs, _, _ = campaign_evidence(work_dir, runs_dir)
    records = []
    for condition, group in (('native', 'baseline'), ('partition', 'partition')):
        entry, result, _ = runs[(condition, 'main')]
        for row in rows:
            path = runs_dir / entry['run_id'] / 'outputs' / (row['sample_id'] + '.mp4')
            records.append({'sample_id': group + '__' + row['sample_id'], 'base_sample_id': row['sample_id'],
                            'group': group, 'identity_id': row['identity_id'], 'split': 'dev', 'prompt': row['prompt'],
                            'prompt_sha256': hashlib.sha256(row['prompt'].encode()).hexdigest(),
                            'reference_image': row['reference_images'][0], 'reference_sha256': row['reference_sha256'],
                            'video_path': str(path), 'video_sha256': result['output_sha256'][path.name],
                            'action_category': row['action_category'], 'framing': row['framing']})
    shards = []
    # Divide within each action, rather than trusting arbitrary manifest interleaving.
    for shard in (0, 1):
        base_ids = {r['sample_id'] for action in PROMPTS
                    for r in [r for r in rows if r['action_category'] == action][shard::2]}
        subset = [r for r in records if r['base_sample_id'] in base_ids]
        if len(subset) != 16 or Counter((r['group'], r['action_category']) for r in subset) != Counter({(g, a): 2 for g in GROUPS for a in PROMPTS}):
            raise ValueError('Each shard must contain sixteen arm/action-balanced inputs')
        shards.append({'samples': subset})
    signatures = {}; evaluator_hashes = {}
    for metric in METRICS:
        label = 'vbench' if metric == 'extended' else metric
        for shard in (0, 1):
            folder = Path(ledger['prior_work_dir']) / 'evaluation' / f'{metric}_shard{shard}_r001'
            result = load_json(folder / 'result.json'); report = load_json(folder / 'scores.json')
            if result['status'] != 'completed' or sha256_file(folder / 'scores.json') != result['score_sha256']:
                raise ValueError('Historical scoring lock incomplete')
            signature = scorer_signature(report, label)
            if label in signatures and signatures[label] != signature:
                raise ValueError('Historical scorer signatures differ across shards')
            signatures[label] = signature
            cmd = scorer_argv(metric, shard, 'launcher', work_dir, repo_root, eval_root)
            evaluator_hashes[metric] = sha256_file(cmd[1])
            if evaluator_hashes[metric] != result['evaluator_sha256']:
                raise ValueError('Native scoring implementation changed')
    metadata = {'benchmark_version': VERSION, 'ledger_sha256': sha256_file(work_dir / 'ledger.json'),
                'controller_sha256': sha256_file(Path(__file__)), 'signatures': signatures,
                'scorer_signatures_sha256': fingerprint(signatures), 'evaluator_sha256': evaluator_hashes,
                'generation_source_sha256': sorted({e['source_sha256'] for e in ledger['runs']}),
                'commands': {f'{metric}_{shard}_{host}': scorer_argv(metric, shard, host, work_dir, repo_root, eval_root)
                             for metric in METRICS for shard in (0, 1) for host in ('launcher', 'worker')}}
    for path, shard in zip(destinations[1:3], shards):
        write_json(path, shard)
    write_json(destinations[0], {'samples': records})
    metadata['panel_sha256'] = sha256_file(destinations[0])
    metadata['shard_sha256'] = [sha256_file(p) for p in destinations[1:3]]
    write_json(destinations[3], metadata)
    return metadata


def score(metric, shard, host, work_dir, repo_root, eval_root):
    metadata = load_json(work_dir / 'evaluation_frozen.json')
    cmd = scorer_argv(metric, shard, host, work_dir, repo_root, eval_root)
    manifest = work_dir / f'quality_shard{shard}.json'
    if (sha256_file(manifest) != metadata['shard_sha256'][shard]
            or len(load_json(manifest)['samples']) != 16
            or sha256_file(cmd[1]) != metadata['evaluator_sha256'][metric]
            or sha256_file(Path(__file__)) != metadata['controller_sha256']
            or cmd != metadata['commands'][f'{metric}_{shard}_{host}']
            or sha256_file(work_dir / 'ledger.json') != metadata['ledger_sha256']):
        raise ValueError('Frozen scorer/controller/shard changed')
    folder = work_dir / 'evaluation' / f'{metric}_shard{shard}_r001'
    if folder.exists():
        raise FileExistsError('No scoring retries or overwrites')
    folder.mkdir(parents=True)
    result = {'status': 'running', 'run_id': folder.name, 'gpus': 1, 'timeout_s': 900,
              'reserved_gpu_hours': .25, 'manifest_sha256': sha256_file(manifest),
              'evaluator_sha256': sha256_file(cmd[1]), 'controller_sha256': sha256_file(Path(__file__)),
              'source_sha256': _source_hash(repo_root), 'host': host, 'command': cmd}
    write_json(folder / 'result.json', result)
    start = time.monotonic()
    try:
        run_process(cmd, cwd=repo_root, timeout_s=900, stdout_path=folder / 'stdout.log', stderr_path=folder / 'stderr.log')
        report = load_json(folder / 'scores.json')
        if len(report['records']) != 16 or report['manifest_sha256'] != result['manifest_sha256']:
            raise ValueError('Incomplete sixteen-score frozen shard')
        label = 'vbench' if metric == 'extended' else metric
        if scorer_signature(report, label) != metadata['signatures'][label]:
            raise ValueError('Original scoring weights/protocol/precision changed')
        result.update(status='completed', score_sha256=sha256_file(folder / 'scores.json'))
    except BaseException as error:
        result.update(status='failed', error_class=type(error).__name__)
        raise
    finally:
        result['backend_elapsed_seconds'] = time.monotonic() - start
        result['allocated_gpu_hours'] = result['backend_elapsed_seconds'] / 3600
        if result['allocated_gpu_hours'] > .25:
            result.update(status='failed', error_class='BudgetExceeded')
        write_json(folder / 'result.json', result)
    if result['status'] != 'completed':
        raise ValueError('Scoring budget exceeded')
    return result


def gate_results(comparison, coverage, groups):
    stats = comparison['comparisons']['main_same_input']['groups']['partition']['core_metrics']
    checks = {}
    for metric in ('gme_aesr', 'cur_score', 'arc_score', 'imaging_quality'):
        delta = stats[metric]['paired_video_mean_delta']
        checks[metric] = (delta is not None and delta >= PARTITION_GATES[metric + '_min_delta']
                          and stats[metric]['paired_videos'] > 0
                          and stats[metric]['candidate']['coverage'] >= stats[metric]['baseline']['coverage'])
    checks['missing_identity'] = coverage['partition']['all_missing_identity_videos'] <= PARTITION_GATES['max_missing_identity_videos']
    checks['face_frame_coverage'] = coverage['partition']['mean_face_frame_coverage'] >= coverage['baseline']['mean_face_frame_coverage']
    for metric in ('cur_scoring_coverage', 'arc_scoring_coverage'):
        left = groups['baseline']['scores'][metric]['mean']; right = groups['partition']['scores'][metric]['mean']
        checks[metric] = left is not None and right is not None and right >= left
    passed = all(checks.values())
    return {'thresholds': PARTITION_GATES, 'checks': checks, 'passed': passed,
            'decision': 'candidate_for_further_verification' if passed else 'not_promoted',
            'automatic_winner_selected': None}


def audit(work_dir, runs_dir, repo_root, output):
    if output.exists():
        raise FileExistsError('Fresh public audit output required')
    if not output.parent.is_dir():
        raise FileNotFoundError('Audit parent must exist')
    ledger = load_json(work_dir / 'ledger.json')
    jobs = []; total = 0; failures = []
    # Account every existing failed job before rejecting incomplete scientific evidence.
    for entry in ledger['runs']:
        path = runs_dir / entry['run_id'] / 'result.json'
        if not path.exists():
            failures.append('missing_generation'); continue
        result = load_json(path)
        h = result['backend_elapsed_seconds'] * result['resources']['gpus'] / 3600
        if not math.isfinite(h) or h < 0 or not math.isclose(h, result['estimated_backend_gpu_hours'], abs_tol=1e-10):
            raise ValueError('Generation cost inconsistent')
        total += h
        jobs.append({'run_id': entry['run_id'], 'status': result['status'], 'gpu_hours': h})
        if result['status'] != 'completed' or h > entry['reserved_gpu_hours']:
            failures.append('generation_failed_or_over_budget')
    for metric in METRICS:
        for shard in (0, 1):
            path = work_dir / 'evaluation' / f'{metric}_shard{shard}_r001' / 'result.json'
            if not path.exists():
                failures.append('missing_scoring'); continue
            result = load_json(path); h = result['backend_elapsed_seconds'] / 3600
            if (not math.isfinite(h) or h < 0 or result['gpus'] != 1
                    or not math.isclose(h, result['allocated_gpu_hours'], abs_tol=1e-10)):
                raise ValueError('Scoring cost inconsistent')
            total += h
            jobs.append({'run_id': result['run_id'], 'status': result['status'], 'gpu_hours': h})
            if result['status'] != 'completed' or h > .25:
                failures.append('scoring_failed_or_over_budget')
    if total > 8:
        failures.append('hard_budget_exceeded')
    if failures:
        report = {'status': 'failed_partition_benchmark', 'decision': 'not_promoted', 'failure_codes': sorted(set(failures)),
                  'jobs': jobs, 'actual_new_gpu_hours': total, 'new_hard_cap_gpu_hours': 8, 'retry_count': 0,
                  'official_final_score': None, 'action_naturalness_verified': False, 'automatic_winner_selected': None}
        write_json(output, report)
        return report
    ledger, rows, _, generation, _ = campaign_evidence(work_dir, runs_dir)
    metadata = load_json(work_dir / 'evaluation_frozen.json')
    if (sha256_file(work_dir / 'ledger.json') != metadata['ledger_sha256']
            or sha256_file(work_dir / 'quality_frozen.json') != metadata['panel_sha256']):
        raise ValueError('Frozen evaluation ledger or panel changed')
    panel = load_json(work_dir / 'quality_frozen.json')['samples']; ids = {r['sample_id']: r for r in panel}
    if len(panel) != 32 or len(ids) != 32 or Counter((r['group'], r['action_category']) for r in panel) != Counter({(g, a): 4 for g in GROUPS for a in PROMPTS}):
        raise ValueError('Expected thirty-two paired balanced frozen videos')
    scores = {sid: {} for sid in ids}; identities = []; signatures = {}; scoring = []
    for row in panel:
        original = next(r for r in rows if r['sample_id'] == row['base_sample_id'])
        if (row['identity_id'] != original['identity_id'] or row['prompt'] != original['prompt']
                or row['reference_image'] != original['reference_images'][0]
                or row['reference_sha256'] != sha256_file(row['reference_image'])
                or row['prompt_sha256'] != hashlib.sha256(row['prompt'].encode()).hexdigest()
                or row['video_sha256'] != sha256_file(row['video_path'])):
            raise ValueError('Frozen original prompt/reference/media changed')
        condition = 'native' if row['group'] == 'baseline' else 'partition'
        main = next(e for e in ledger['runs'] if e['condition'] == condition and e['phase'] == 'main')
        expected_path = runs_dir / main['run_id'] / 'outputs' / (row['base_sample_id'] + '.mp4')
        if Path(row['video_path']).resolve() != expected_path.resolve():
            raise ValueError('Panel did not use the frozen execution outputs')
    for metric in METRICS:
        seen = set(); label = 'vbench' if metric == 'extended' else metric
        for shard in (0, 1):
            manifest = work_dir / f'quality_shard{shard}.json'
            shard_rows = load_json(manifest)['samples']; shard_ids = {r['sample_id'] for r in shard_rows}
            if (sha256_file(manifest) != metadata['shard_sha256'][shard] or len(shard_rows) != 16
                    or any(r != ids.get(r['sample_id']) for r in shard_rows)
                    or Counter((r['group'], r['action_category']) for r in shard_rows) != Counter({(g, a): 2 for g in GROUPS for a in PROMPTS})):
                raise ValueError('Frozen sixteen-video arm/action-balanced shard differs')
            folder = work_dir / 'evaluation' / f'{metric}_shard{shard}_r001'
            result = load_json(folder / 'result.json'); report = load_json(folder / 'scores.json')
            if (result['run_id'] != folder.name or result['reserved_gpu_hours'] != .25 or result['timeout_s'] != 900
                    or sha256_file(folder / 'scores.json') != result['score_sha256']
                    or result['manifest_sha256'] != sha256_file(manifest)
                    or report['manifest_sha256'] != result['manifest_sha256']
                    or result['evaluator_sha256'] != metadata['evaluator_sha256'][metric]
                    or result['controller_sha256'] != metadata['controller_sha256'] or not result.get('source_sha256')
                    or result['command'] != metadata['commands'][f"{metric}_{shard}_{result['host']}"]
                    or sha256_file(result['command'][1]) != result['evaluator_sha256']):
                raise ValueError('Scoring execution/source/hash controls changed')
            signature = scorer_signature(report, label)
            if signature != metadata['signatures'][label]:
                raise ValueError('Original evaluator weights/protocol/precision changed')
            signatures[label] = signature
            records = report_records(report)
            if len(records) != 16 or {r['sample_id'] for r, _ in records} != shard_ids:
                raise ValueError('Raw scores must cover all sixteen shard inputs')
            for row, values in records:
                sid = row['sample_id']; frozen = ids[sid]
                if sid in seen or row['group'] != frozen['group'] or row['video_sha256'] != frozen['video_sha256']:
                    raise ValueError('Scorer duplicate/group/media mismatch')
                if metric == 'cur' and (row['reference_sha256'] != frozen['reference_sha256'] or row['identity_id'] != frozen['identity_id']):
                    raise ValueError('Identity scoring anchor/actor changed')
                if metric in ('gme', 'extended') and row['prompt_sha256'] != frozen['prompt_sha256']:
                    raise ValueError('Scoring original prompt changed')
                for value in values.values():
                    finite_values([value])
                if metric != 'cur' and any(value is None for value in values.values()):
                    raise ValueError('Non-identity automatic metrics must be fully observed')
                if metric == 'cur':
                    for name in ('cur', 'arc'):
                        if not any(v is not None for v in row[name + '_frame_cosines']) and row[name + '_score'] != 0:
                            raise ValueError('Original author all-missing identity video must retain zero')
                    identities.append(row)
                scores[sid].update(values); seen.add(sid)
            scoring.append({k: result[k] for k in ('run_id', 'status', 'manifest_sha256', 'score_sha256',
                                                   'evaluator_sha256', 'controller_sha256', 'source_sha256', 'allocated_gpu_hours')})
        if seen != set(ids):
            raise ValueError('Each metric must cover all thirty-two raw inputs')
    required = set(CORE_METRICS) | {'face_detection_coverage', 'cur_scoring_coverage', 'arc_scoring_coverage',
                                  'dynamic_degree', 'aesthetic_quality', 'subject_consistency', 'background_consistency', 'overall_consistency'}
    if any(not required.issubset(s) for s in scores.values()):
        raise ValueError('Complete core and auxiliary score columns required')
    from ipvg.evaluate import probe_video
    comparison = paired_comparisons(panel, scores, {sid: probe_video(r['video_path']) for sid, r in ids.items()}, signatures)
    coverage = identity_coverage_summary(identities)
    groups = {}; actions = {}
    for action in (None, *PROMPTS):
        target = groups if action is None else actions.setdefault(action, {})
        for group in GROUPS:
            subset = [scores[sid] for sid, row in ids.items() if row['group'] == group and (action is None or row['action_category'] == action)]
            target[group] = {'videos': len(subset), 'scores': {k: dict(finite_values([s[k] for s in subset]),
                             coverage=sum(s[k] is not None for s in subset) / len(subset)) for k in sorted(required)}}
            identity_subset = [r for r in identities if r['group'] == group and (action is None or ids[r['sample_id']]['action_category'] == action)]
            target[group]['identity_coverage_diagnostics'] = identity_coverage_summary(identity_subset)[group]
    gates = gate_results(comparison, coverage, groups)
    report = {'status': 'completed_partition_benchmark', 'decision': gates['decision'], 'gates': gates,
              'comparison': comparison, 'groups': groups, 'action_stratified_groups': actions,
              'identity_coverage_diagnostics': coverage, 'generation_runs': generation, 'scoring_runs': scoring,
              'analysis_script_sha256': sha256_file(Path(__file__)), 'ledger_sha256': metadata['ledger_sha256'],
              'actual_new_gpu_hours': total, 'new_hard_cap_gpu_hours': 8,
              'original_overall_cap_gpu_hours': 51.5, 'prior_gpu_hours_upper': 5.31,
              'generation_reserved_gpu_hours': 5, 'evaluation_reserved_gpu_hours': 2,
              'single_engineering_repair_allowance_gpu_hours': .5, 'main_comparison_videos': 32,
              'new_generated_videos': 36, 'retry_count': 0, 'training_performed': False,
              'combined_score': None, 'official_final_score': None, 'action_naturalness_verified': False,
              'automatic_winner_selected': None,
              'limitations': ['Same eight development actors/four reference-only action prompts/one seed; no action ground truth.',
                              'Passing frozen gates permits further verification, not promotion or an automatic winner.',
                              'Identity all-missing author-style zeros retained with explicit coverage/missing denominators.']}
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'score', 'audit'))
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--runs-dir', type=Path, default=ROOT / 'runs')
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--eval-root', type=Path, default=ROOT / 'eval_complete_20261021')
    parser.add_argument('--metric', choices=METRICS)
    parser.add_argument('--shard', type=int, choices=(0, 1))
    parser.add_argument('--host', choices=('launcher', 'worker'), default='launcher')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze(args.work_dir, args.runs_dir, args.repo_root, args.eval_root)
    elif args.mode == 'score':
        if args.metric is None or args.shard is None:
            parser.error('score requires --metric and --shard')
        score(args.metric, args.shard, args.host, args.work_dir, args.repo_root, args.eval_root)
    else:
        if args.output is None:
            parser.error('audit requires --output')
        audit(args.work_dir, args.runs_dir, args.repo_root, args.output)


if __name__ == '__main__':
    main()
