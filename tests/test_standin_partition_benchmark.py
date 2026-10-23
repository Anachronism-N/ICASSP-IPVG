import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from ipvg.common import load_json, sha256_file, write_json
from ipvg.data import read_manifest, write_manifest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import prepare_native_action_benchmark as prepare
import standin_partition_benchmark as benchmark


def structure(condition):
    mode = 'native' if condition == 'native' else 'partition_balanced'
    return {'mode': mode, 'block_count': 40, 'joint_video_reads': 1600,
            'ip_self_reads': 40, 'plain_reads': 0, 'kernel_calls': 1640 if condition == 'native' else 3240,
            'partition_reads': 0 if condition == 'native' else 1600,
            'blocks': [{'block_index': i, 'joint_video_reads': 40, 'ip_self_reads': 1, 'plain_reads': 0,
                        'kernel_calls': 41 if condition == 'native' else 81,
                        'partition_reads': 0 if condition == 'native' else 40} for i in range(40)]}


def report_for(metric, rows, manifest):
    protocol = {'version': 'original_' + metric, 'precision': 'float32', 'detector_provider': 'CPU'}
    report = {'protocol': protocol, 'manifest_sha256': sha256_file(manifest), 'records': []}
    if metric == 'cur':
        report.update(model_source={'files': {'weight': 'curhash'}, 'source_files': {'source': 'sourcehash'}}, device='cuda')
    elif metric == 'gme':
        report.update(model_file_sha256={'weight': 'gmehash'}, execution={'device': 'cuda', 'dtype': 'float16',
                      'batch_size': 1, 'attention_implementation': 'original'})
    else:
        report.update(weight_hashes={'weight': metric + 'hash'}, actual_parameter_dtypes={'dtype': 'original'})
    for row in rows:
        candidate = row['group'] == 'partition'
        record = {k: row[k] for k in ('sample_id', 'group', 'video_sha256')}
        if metric == 'cur':
            value = .82 if candidate else .8
            record.update(identity_id=row['identity_id'], reference_sha256=row['reference_sha256'],
                          cur_score=value, arc_score=value, cur_frame_cosines=[value] * 16,
                          arc_frame_cosines=[value] * 16, cur_scoring_coverage=1, arc_scoring_coverage=1,
                          face_detection_coverage=1)
        elif metric == 'gme':
            record.update(prompt_sha256=row['prompt_sha256'], scores={'gme_aesr': .55 if candidate else .5})
        elif metric == 'quality':
            record['scores'] = {'motion_smoothness': .9, 'imaging_quality': .8}
        else:
            record.update(prompt_sha256=row['prompt_sha256'], scores={'dynamic_degree': .5, 'aesthetic_quality': .8,
                          'subject_consistency': .9, 'background_consistency': .9, 'overall_consistency': .7})
        report['records'].append(record)
    return report


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'; (repo / 'scripts').mkdir(parents=True); (repo / 'src').mkdir()
    for name in ('gme_text_eval.py', 'curricularface_eval.py', 'mm2026_quality_eval.py', 'vbench_extended_eval.py'):
        (repo / 'scripts' / name).write_text('original scorer')
    runs = tmp_path / 'runs'; runs.mkdir()
    prior = tmp_path / 'prior'; prior.mkdir()
    reference = tmp_path / 'reference.png'; reference.write_bytes(b'original ref')
    rows = []
    for i in range(8):
        actions = [('portrait_turn', 'halfbody_raise_hand')[i % 2], ('fullbody_walk', 'fullbody_sit')[(i // 2) % 2]]
        for action in actions:
            rows.append({'sample_id': f'actor{i}_{action}', 'identity_id': f'actor{i}', 'source_video_id': f'source{i}',
                         'reference_images': [str(reference)], 'reference_sha256': sha256_file(reference),
                         'prompt': prepare.PROMPTS[action], 'track': 1, 'split': 'dev', 'timeline': [],
                         'action_category': action, 'framing': 'fullbody' if action.startswith('fullbody') else 'portrait'})
    manifest = prior / 'manifest.jsonl'; write_manifest(manifest, rows)
    monkeypatch.setattr(prepare, 'ACTION_MANIFEST_SHA256', sha256_file(manifest))
    monkeypatch.setattr(benchmark, 'ACTION_MANIFEST_SHA256', sha256_file(manifest))
    folder = runs / prepare.BASELINE_HISTORY_RUN_ID; (folder / 'outputs').mkdir(parents=True)
    config = {'schema_version': 1, 'task': 'inference', 'purpose': 'development', 'split': 'dev',
              'runs_dir': str(runs), 'backend': {'kind': 'command', 'command': ['python', 'standin.py', '--request', '{request}']},
              'model_revisions': {'standin': 'pinned'}, 'parameters': {'width': 480, 'height': 352, 'num_frames': 49,
              'fps': 16, 'num_inference_steps': 20, 'cfg_scale': 5, 'seed': 42},
              'run_id': prepare.BASELINE_HISTORY_RUN_ID, 'manifest': str(manifest), 'hypothesis': 'original',
              'resources': {'gpus': 1, 'timeout_s': 7200, 'max_gpu_hours': 2}}
    write_json(folder / 'config.json', config); (folder / 'manifest.jsonl').write_bytes(manifest.read_bytes())
    profile = {'model_lock_sha256': 'modellock', 'standin_checkpoint_sha256': '52f47630b4fefbdac841d3fd1ce146cca2a6a94ec0a6ab9407d16f2f635177e9',
               'matched_ip_tensors': 240, 'code_revision': 'original', 'native_vram_management': True,
               'attention_backend': 'FA2', 'preprocessor': 'original', 'samples': []}
    for row in rows:
        sid = row['sample_id']; (folder / 'outputs' / (sid + '.mp4')).write_bytes(('original_' + sid).encode())
        profile['samples'].append({'sample_id': sid, 'seed': (42 + int(benchmark.hashlib.sha256(sid.encode()).hexdigest()[:8], 16)) % 2**31,
                                  'generated_frames': 49, 'reference_original_sha256': row['reference_sha256'],
                                  'reference_pixels_sha256': 'pixels', 'native_denoiser_calls': 40,
                                  'ip_latent_calls': [{'ip_latent_shape': [1, 16, 1, 64, 64]}],
                                  'ip_q_lora_calls': [[1, 1024, 5120]], 'generation_seconds': 1,
                                  'peak_allocated_vram_gb': 1, 'peak_reserved_vram_gb': 1})
    write_json(folder / 'backend_metrics.json', profile)
    result = {'status': 'completed', 'run_id': config['run_id'], 'source_sha256': 'oldsource',
              'resources': config['resources'], 'backend_elapsed_seconds': 1, 'estimated_backend_gpu_hours': 1 / 3600,
              'manifest_sha256': sha256_file(manifest), 'code_commit': 'git', 'model_revisions': config['model_revisions'],
              'output_sha256': {r['sample_id'] + '.mp4': sha256_file(folder / 'outputs' / (r['sample_id'] + '.mp4')) for r in rows},
              'started_at': '2026-10-23T00:00:00', 'finished_at': '2026-10-23T00:00:01'}
    write_json(folder / 'result.json', result)
    entry = {'arm': 'standin', 'phase': 'main', 'run_id': config['run_id'], 'config_sha256': sha256_file(folder / 'config.json'),
             'manifest_sha256': sha256_file(manifest), 'source_sha256': 'oldsource', 'count': 16, 'reserved_gpu_hours': 2}
    write_json(prior / 'ledger.json', {'runs': [entry]})
    for metric in benchmark.METRICS:
        for shard in (0, 1):
            job = prior / 'evaluation' / f'{metric}_shard{shard}_r001'; job.mkdir(parents=True)
            write_json(job / 'scores.json', report_for(metric, [], manifest))
            evaluator = benchmark.scorer_argv(metric, shard, 'launcher', tmp_path / 'work', repo, tmp_path / 'eval')[1]
            write_json(job / 'result.json', {'status': 'completed', 'score_sha256': sha256_file(job / 'scores.json'),
                                           'evaluator_sha256': sha256_file(evaluator)})
    args = SimpleNamespace(manifest=manifest, prior_work_dir=prior, output_dir=tmp_path / 'work',
                           runs_dir=runs, repo_root=repo)
    ledger = prepare.prepare_standin_partition(args)
    for e in ledger['runs']:
        target = runs / e['run_id']; (target / 'outputs').mkdir(parents=True)
        c = load_json(args.output_dir / 'configs' / (e['run_id'] + '.json'))
        write_json(target / 'config.json', c); (target / 'manifest.jsonl').write_bytes(Path(c['manifest']).read_bytes())
        selected = read_manifest(c['manifest']); selected_ids = {r['sample_id'] for r in selected}
        p = copy.deepcopy(profile); p['attention_mode'] = c['parameters']['attention_mode']
        p['samples'] = [s for s in p['samples'] if s['sample_id'] in selected_ids]
        for s in p['samples']:
            s['attention_structure'] = structure(e['condition'])
            data = (folder / 'outputs' / (s['sample_id'] + '.mp4')).read_bytes()
            (target / 'outputs' / (s['sample_id'] + '.mp4')).write_bytes(data if e['condition'] == 'native' else data + b'partition')
        write_json(target / 'backend_metrics.json', p)
        r = copy.deepcopy(result); r.update(run_id=e['run_id'], source_sha256=e['source_sha256'], resources=c['resources'],
                  manifest_sha256=e['manifest_sha256'], started_at='2026-10-23T00:01:00' if e['phase'] == 'main' else '2026-10-23T00:00:00',
                  finished_at='2026-10-23T00:02:00' if e['phase'] == 'main' else '2026-10-23T00:00:01',
                  output_sha256={sid + '.mp4': sha256_file(target / 'outputs' / (sid + '.mp4')) for sid in selected_ids})
        write_json(target / 'result.json', r)
    def media_check(manifest, outputs, **kwargs):
        return {'ok': True, 'samples': [{'sample_id': r['sample_id'], 'media': {'decoded_frames': 49}} for r in read_manifest(manifest)]}
    monkeypatch.setattr('ipvg.evaluate.evaluate_outputs', media_check)
    monkeypatch.setattr('ipvg.evaluate.probe_video', lambda p: {'ok': True, 'width': 480, 'height': 352,
                        'decoded_frames': 49, 'fps': 16, 'duration_s': 49 / 16})
    return SimpleNamespace(work=args.output_dir, runs=runs, repo=repo, eval=tmp_path / 'eval', ledger=ledger, args=args)


def mock_scores(campaign, monkeypatch):
    def run(command, **kwargs):
        manifest = Path(command[command.index('--manifest') + 1]); output = Path(command[command.index('--output') + 1])
        metric = output.parent.name.split('_shard')[0]
        write_json(output, report_for(metric, load_json(manifest)['samples'], manifest))
    monkeypatch.setattr(benchmark, 'run_process', run)
    for metric in benchmark.METRICS:
        for shard in (0, 1):
            benchmark.score(metric, shard, 'worker', campaign.work, campaign.repo, campaign.eval)


def test_prepare_freeze_score_audit_cpu(campaign, monkeypatch):
    c = campaign
    assert (c.work / 'manifest.jsonl').read_bytes() == c.args.manifest.read_bytes()
    assert len(c.ledger['runs']) == 6 and sum(e['reserved_gpu_hours'] for e in c.ledger['runs']) == 5
    assert [e['phase'] for e in c.ledger['runs']] == ['pilot_portrait'] * 2 + ['pilot_fullbody'] * 2 + ['main'] * 2
    benchmark.freeze(c.work, c.runs, c.repo, c.eval)
    for shard in (0, 1):
        records = load_json(c.work / f'quality_shard{shard}.json')['samples']
        assert len(records) == 16
        assert benchmark.Counter((r['group'], r['action_category']) for r in records) == benchmark.Counter({(g, a): 2 for g in benchmark.GROUPS for a in prepare.PROMPTS})
    mock_scores(c, monkeypatch)
    result = benchmark.audit(c.work, c.runs, c.repo, c.work / 'audit.json')
    assert result['status'] == 'completed_partition_benchmark' and result['gates']['passed']
    assert result['decision'] == 'candidate_for_further_verification'
    assert result['official_final_score'] is None and result['automatic_winner_selected'] is None
    assert len(result['generation_runs']) == 6 and len(result['scoring_runs']) == 8
    assert result['comparison']['comparisons']['main_same_input']['groups']['partition']['core_metrics']['cur_score']['paired_videos'] == 16
    public = (c.work / 'audit.json').read_text()
    assert all(word not in public for word in ('reference_image', 'identity_id', 'frame_cosines', 'video_sha256', str(c.work), 'actor0'))
    with pytest.raises(FileExistsError): benchmark.freeze(c.work, c.runs, c.repo, c.eval)
    with pytest.raises(FileExistsError): benchmark.audit(c.work, c.runs, c.repo, c.work / 'audit.json')
    with pytest.raises(FileExistsError): benchmark.score('cur', 0, 'worker', c.work, c.repo, c.eval)
    with pytest.raises(FileExistsError): prepare.prepare_standin_partition(c.args)


@pytest.mark.parametrize('change', ['native_hash', 'source', 'seed', 'calls', 'structure', 'owner', 'parameter', 'timing'])
def test_campaign_rejects_execution_and_structure_drift(campaign, change):
    c = campaign; entry = next(e for e in c.ledger['runs'] if e['condition'] == 'native' and e['phase'] == 'main')
    folder = c.runs / entry['run_id']
    if change == 'native_hash':
        r = load_json(folder / 'result.json'); name = next(iter(r['output_sha256']))
        (folder / 'outputs' / name).write_bytes(b'changed'); r['output_sha256'][name] = sha256_file(folder / 'outputs' / name)
        write_json(folder / 'result.json', r)
    elif change in ('source', 'timing'):
        r = load_json(folder / 'result.json'); r['source_sha256' if change == 'source' else 'started_at'] = 'drift' if change == 'source' else '2026-10-22T00:00:00'
        write_json(folder / 'result.json', r)
    elif change == 'parameter':
        r = load_json(folder / 'config.json'); r['parameters']['cfg_scale'] = True; write_json(folder / 'config.json', r)
    else:
        r = load_json(folder / 'backend_metrics.json'); sample = r['samples'][0]
        if change == 'seed': sample['seed'] += 1
        if change == 'calls': sample['native_denoiser_calls'] = 39
        if change == 'structure': sample['attention_structure']['blocks'][0]['kernel_calls'] = 40
        if change == 'owner': sample['attention_structure']['blocks'][0]['block_index'] = 1
        write_json(folder / 'backend_metrics.json', r)
    with pytest.raises(ValueError): benchmark.freeze(c.work, c.runs, c.repo, c.eval)
    assert not (c.work / 'quality_frozen.json').exists()


@pytest.mark.parametrize('bad', ['timeout', 'short', 'precision', 'interrupt', 'budget'])
def test_score_failure_always_accounts_real_time(campaign, monkeypatch, bad):
    c = campaign; benchmark.freeze(c.work, c.runs, c.repo, c.eval)
    def run(command, **kwargs):
        if bad == 'timeout': raise TimeoutError('bounded')
        if bad == 'interrupt': raise KeyboardInterrupt()
        manifest = c.work / 'quality_shard0.json'
        report = report_for('gme', load_json(manifest)['samples'], manifest)
        if bad == 'short': report['records'].pop()
        if bad == 'precision': report['execution']['dtype'] = 'float32'
        write_json(c.work / 'evaluation/gme_shard0_r001/scores.json', report)
    monkeypatch.setattr(benchmark, 'run_process', run)
    clock = iter([0, 901 if bad == 'budget' else 9])
    monkeypatch.setattr(benchmark.time, 'monotonic', lambda: next(clock))
    with pytest.raises((TimeoutError, ValueError, KeyboardInterrupt)):
        benchmark.score('gme', 0, 'launcher', c.work, c.repo, c.eval)
    result = load_json(c.work / 'evaluation/gme_shard0_r001/result.json')
    assert result['status'] == 'failed' and result['allocated_gpu_hours'] == (901 if bad == 'budget' else 9) / 3600
    assert result['source_sha256'] and result['evaluator_sha256'] and result['controller_sha256']
    failure = benchmark.audit(c.work, c.runs, c.repo, c.work / 'failed.json')
    assert failure['status'] == 'failed_partition_benchmark' and failure['decision'] == 'not_promoted'
    assert failure['actual_new_gpu_hours'] >= result['allocated_gpu_hours']


@pytest.mark.parametrize('metric', benchmark.METRICS)
def test_original_driver_argv_and_worker_vbench(metric, campaign, monkeypatch):
    c = campaign
    spec = importlib.util.spec_from_file_location('original_driver', '/dockerdata/ipvg/logs/native_baselines_eval_20261022.py')
    driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
    driver.WORK = c.work; driver.REPO = c.repo; driver.EVAL = c.eval
    write_json(c.work / 'quality_shard0.json', {'samples': []})
    captured = []
    def original_run(command, **kwargs):
        captured.append(command)
        write_json(Path(command[command.index('--output') + 1]), {'records': [{}] * 24})
    monkeypatch.setattr(driver, 'run_process', original_run)
    driver.run(metric, 0, 'worker')
    assert captured == [benchmark.scorer_argv(metric, 0, 'worker', c.work, c.repo, c.eval)]
    assert '--dtype' not in benchmark.scorer_argv(metric, 0, 'launcher', c.work, c.repo, c.eval)
    cmd = benchmark.scorer_argv(metric, 0, 'worker', c.work, c.repo, c.eval)
    assert cmd[cmd.index('--manifest') + 1] == str(c.work / 'quality_shard0.json')
    if metric in ('quality', 'extended'): assert cmd[cmd.index('--vbench') + 1] == str(c.eval / 'VBench_clean')


@pytest.mark.parametrize('drift', ['score_hash', 'shard_hash', 'cohort', 'reference', 'prompt', 'movie', 'coverage', 'author_zero', 'weights', 'controller', 'source'])
def test_audit_rejects_frozen_scoring_and_identity_drift(campaign, monkeypatch, drift):
    c = campaign; benchmark.freeze(c.work, c.runs, c.repo, c.eval); mock_scores(c, monkeypatch)
    score_path = c.work / 'evaluation/cur_shard0_r001/scores.json'
    report = load_json(score_path); row = report['records'][0]
    if drift == 'shard_hash':
        path = c.work / 'quality_shard0.json'; changed = load_json(path); changed['samples'].pop(); write_json(path, changed)
    elif drift == 'cohort':
        path = c.work / 'quality_frozen.json'; changed = load_json(path); changed['samples'][0]['group'] = 'wrong'; write_json(path, changed)
    elif drift in ('controller', 'source'):
        path = score_path.parent / 'result.json'; changed = load_json(path)
        changed['controller_sha256' if drift == 'controller' else 'source_sha256'] = 'drift' if drift == 'controller' else ''
        write_json(path, changed)
    else:
        if drift == 'reference': row['reference_sha256'] = 'changed'
        if drift == 'prompt':
            score_path = c.work / 'evaluation/gme_shard0_r001/scores.json'; report = load_json(score_path)
            report['records'][0]['prompt_sha256'] = 'changed'
        if drift == 'movie': row['video_sha256'] = 'changed'
        if drift == 'coverage': row['cur_scoring_coverage'] = .5
        if drift == 'author_zero':
            row.update(cur_frame_cosines=[None] * 16, arc_frame_cosines=[None] * 16,
                       cur_scoring_coverage=0, arc_scoring_coverage=0, cur_score=42, arc_score=42)
        if drift == 'weights': report['model_source']['files']['weight'] = 'different'
        if drift == 'score_hash': row['cur_score'] += .1
        write_json(score_path, report)
        if drift != 'score_hash':
            path = score_path.parent / 'result.json'; changed = load_json(path)
            changed['score_sha256'] = sha256_file(score_path); write_json(path, changed)
    with pytest.raises(ValueError): benchmark.audit(c.work, c.runs, c.repo, c.work / 'rejected.json')
    assert not (c.work / 'rejected.json').exists()


def test_audit_failed_generation_and_hard_budget_accounted(campaign):
    c = campaign
    entry = c.ledger['runs'][0]; path = c.runs / entry['run_id'] / 'result.json'
    result = load_json(path); result.update(status='failed', backend_elapsed_seconds=9 * 3600, estimated_backend_gpu_hours=9)
    write_json(path, result)
    report = benchmark.audit(c.work, c.runs, c.repo, c.work / 'failed.json')
    assert report['status'] == 'failed_partition_benchmark' and report['actual_new_gpu_hours'] >= 9
    assert 'hard_budget_exceeded' in report['failure_codes']
    assert 'generation_failed_or_over_budget' in report['failure_codes']


def test_verify_one_pilot_does_not_require_other_new_runs(campaign, monkeypatch):
    c = campaign
    entry = c.ledger['runs'][0]
    allowed = (c.runs / entry['run_id']).resolve()
    original = benchmark.generation_evidence
    def verify(folder, e):
        if e['run_id'] != prepare.BASELINE_HISTORY_RUN_ID and Path(folder).resolve() != allowed:
            raise AssertionError('Single pilot verification accessed another new run')
        return original(folder, e)
    monkeypatch.setattr(benchmark, 'generation_evidence', verify)
    result, config, profile, hours = benchmark.verify_run_evidence(allowed, entry, c.ledger, c.runs)
    assert result['status'] == 'completed' and hours < .25
    assert config['parameters']['expected_attention_backend'] == 'FA2'
    assert profile['samples'][0]['attention_structure']['mode'] == 'native'


def test_gate_missing_author_zero_and_coverage_not_promoted():
    metrics = {m: {'paired_video_mean_delta': .1, 'paired_videos': 16,
                  'candidate': {'coverage': 1}, 'baseline': {'coverage': 1}} for m in benchmark.CORE_METRICS}
    comparison = {'comparisons': {'main_same_input': {'groups': {'partition': {'core_metrics': metrics}}}}}
    coverage = {g: {'all_missing_identity_videos': 0, 'mean_face_frame_coverage': 1} for g in benchmark.GROUPS}
    groups = {g: {'scores': {m: {'mean': 1} for m in ('cur_scoring_coverage', 'arc_scoring_coverage')}} for g in benchmark.GROUPS}
    assert benchmark.gate_results(comparison, coverage, groups)['passed']
    coverage['partition']['all_missing_identity_videos'] = 3
    assert not benchmark.gate_results(comparison, coverage, groups)['passed']
    coverage['partition']['all_missing_identity_videos'] = 0
    metrics['cur_score']['paired_video_mean_delta'] = None
    assert not benchmark.gate_results(comparison, coverage, groups)['passed']
    records = [{'group': 'partition', 'cur_frame_cosines': [None] * 16, 'arc_frame_cosines': [None] * 16,
                'cur_scoring_coverage': 0, 'arc_scoring_coverage': 0, 'cur_score': 0, 'arc_score': 0,
                'face_detection_coverage': 0}]
    summary = benchmark.identity_coverage_summary(records)['partition']
    assert summary['all_missing_identity_videos'] == 1 and summary['valid_video_cur_mean_diagnostic_only'] is None
