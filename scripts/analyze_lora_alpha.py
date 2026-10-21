"""Fixed-checkpoint fusion-strength audit; development gates, not official scores."""
import argparse
from pathlib import Path
import numpy as np
from ipvg.common import load_json, sha256_file, write_json
from ipvg.data import read_manifest


def repeat_checks(pair, base_quality, fresh_quality, exact_count, total, thresholds):
    a, b = pair['narrow'], pair['wide']
    checks = {'coverage_not_down': b['identity_coverage'] >= a['identity_coverage'],
              'seven_frame_videos_not_down': b['videos_at_least_seven_frames'] >= a['videos_at_least_seven_frames'],
              'identity_mean_stable': a['valid_video_identity_mean'] is not None and b['valid_video_identity_mean'] is not None and abs(b['valid_video_identity_mean'] - a['valid_video_identity_mean']) <= thresholds['repeat_identity_max'],
              'imaging_stable': abs(fresh_quality['imaging_quality'] - base_quality['imaging_quality']) <= thresholds['repeat_imaging_max'],
              'clip_stable': abs(fresh_quality['clip_frame_text_cosine_mean'] - base_quality['clip_frame_text_cosine_mean']) <= thresholds['repeat_clip_max'],
              'gray_motion_stable': thresholds['repeat_motion_min'] <= pair['motion_ratio'] <= thresholds['repeat_motion_max']}
    return {'passed': exact_count == total or all(checks.values()), 'exact_video_hash_matches': exact_count,
            'total_videos': total, 'checks': checks, 'identity_comparison': pair}


def candidate_checks(identity, mouth, quality_deltas, full_gain, thresholds):
    low = identity['identity_actor_bootstrap_ci95'][0]
    gain = identity['identity_actor_mean_delta']
    p10 = identity['identity_p10_actor_delta']
    a, b = identity['narrow'], identity['wide']
    m0, m1 = mouth['single'], mouth['dual']
    return {'identity_coverage_complete': b['identity_coverage'] == 1,
            'seven_frame_videos_not_down': b['videos_at_least_seven_frames'] >= a['videos_at_least_seven_frames'],
            'eight_paired_actors': identity['paired_actors'] == 8,
            'sixteen_paired_videos': identity['paired_videos'] == 16,
            'retain_60percent_full_identity_gain': full_gain is not None and full_gain > 0 and gain is not None and gain >= thresholds['gain_retention'] * full_gain,
            'identity_ci_lower_positive': low is not None and low > 0,
            'identity_p10_not_down': p10 is not None and p10 >= 0,
            'gray_motion_retained': identity['motion_ratio'] >= thresholds['gray_motion_min'],
            'mouth_activity_retained': m1['activity_video_rate_all'] >= m0['activity_video_rate_all'] - thresholds['mouth_rate_drop_max'],
            'mouth_paircoverage_retained': m1['mean_pair_coverage'] >= thresholds['mouth_paircoverage_min'],
            'mouth_flow_retained': mouth['paired_flow_ratio'] is not None and mouth['paired_flow_ratio'] >= thresholds['mouth_flow_min'],
            'imaging_not_degraded': quality_deltas['imaging_quality'] >= -thresholds['imaging_drop_max'],
            'clip_not_degraded': quality_deltas['clip_frame_text_cosine_mean'] >= -thresholds['clip_drop_max']}


def arm_data(arm, ledger, runs, scores, quality):
    from ipvg.evaluate import evaluate_outputs
    folder = runs / ledger['run_names'][arm]
    run, config, profile = [load_json(folder / f'{name}.json') for name in ('result', 'config', 'backend_metrics')]
    manifest = Path(ledger['eval_manifest'])
    if (run['status'] != 'completed' or sha256_file(manifest) != ledger['eval_manifest_sha256']
            or sha256_file(folder / 'manifest.jsonl') != ledger['eval_manifest_sha256']):
        raise ValueError('Incomplete run or input drift')
    rows = read_manifest(manifest)
    ids = {r['sample_id'] for r in rows}
    if len(ids) != 16 or len({r['identity_id'] for r in rows}) != 8 or any(r['split'] != 'dev' for r in rows):
        raise ValueError('Unexpected development cohort')
    media = evaluate_outputs(manifest, folder / 'outputs', expected_width=480, expected_height=352,
                             expected_fps=16, expected_duration=49 / 16, fail_on_extra=True)
    if not media['ok'] or any(r['media']['decoded_frames'] != 49 for r in media['samples']):
        raise ValueError('Invalid complete-generation media')
    historical = {'baseline': 'baseline', 'historical1': 'lr5'}
    prefix = 'real_vace_' + historical[arm] if arm in historical else 'lora_' + arm
    date = '20261020' if arm in historical else '20261021'
    reports = {k: load_json(scores / f'{prefix}_{k}_{date}.json') for k in ('identity', 'clip', 'mouth')}
    indexed = {}
    expected_actors = {r['sample_id']: r['identity_id'] for r in rows}
    versions = {'identity': 'arcface_cremad_v1', 'clip': 'clip_b32_frame_text_v1', 'mouth': 'local_mouth_motion_v1'}
    for kind, report in reports.items():
        if report['protocol']['version'] != versions[kind]:
            raise ValueError('Wrong frozen scorer protocol')
        if kind in ('identity', 'mouth') and not report.get('weight_hashes'):
            raise ValueError('Missing scorer weights')
        if kind in ('clip', 'mouth') and report['manifest_sha256'] != ledger['eval_manifest_sha256']:
            raise ValueError('Scoring input manifest differs')
        if kind == 'mouth' and report['calibration_status'] != 'passed_local_activity_controls':
            raise ValueError('Mouth calibration not passed')
        index = {r['sample_id']: r for r in report['records']}
        if len(index) != len(report['records']) or set(index) != ids:
            raise ValueError('Scoring IDs incomplete or duplicate')
        if any(r['identity_id'] != expected_actors[s] for s, r in index.items()):
            raise ValueError('Score sample actor differs from manifest')
        if kind == 'clip' and any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)
                                 for r in index.values() for v in r['scores'].values()):
            raise ValueError('Nonfinite or malformed CLIP scores')
        indexed[kind] = index
    observations = {r['sample_id']: r for r in profile['samples']}
    if set(observations) != ids or len(profile['samples']) != 16:
        raise ValueError('Generation observation coverage differs')
    for sample in ids:
        digest = sha256_file(folder / 'outputs' / (sample + '.mp4'))
        record = quality[arm + '__' + sample]
        if record['group'] != arm:
            raise ValueError('Quality group differs')
        if any(h != digest for h in (run['output_sha256'][sample + '.mp4'], indexed['identity'][sample]['output_sha256'],
                                     indexed['clip'][sample]['video_sha256'], indexed['mouth'][sample]['video_sha256'], record['video_sha256'])):
            raise ValueError('Scoring or video hash drift')
    alpha = {'baseline': None, 'historical1': 1, 'alpha1': 1, 'alpha05': .5, 'alpha025': .25}[arm]
    if alpha is not None:
        reload = profile['lora_reload']
        if (reload['checkpoint_sha256'] != ledger['checkpoint_sha256'] or reload['matched_pairs'] != 150
                or reload['changed_base_tensors'] <= 0 or reload.get('alpha', 1) != alpha
                or config['parameters'].get('lora_alpha', 1) != alpha):
            raise ValueError('Wrong checkpoint, fusion strength or real reload')
    measured_hours = run['resources']['gpus'] * run['backend_elapsed_seconds'] / 3600
    if (not np.isfinite(measured_hours) or measured_hours < 0
            or not np.isclose(measured_hours, run['estimated_backend_gpu_hours'])):
        raise ValueError('Inference cost inconsistent')
    if arm not in historical:
        entry = next(r for r in ledger['entries'] if r['name'] == arm)
        if (sha256_file(folder / 'config.json') != entry['config_sha256'] or run['source_sha256'] != entry['source_sha256']
                or run['estimated_backend_gpu_hours'] > entry['reserved_gpu_hours']):
            raise ValueError('Frozen execution source/config or budget drift')
    panel = {k: float(np.mean([quality[arm + '__' + s]['scores'][k] for s in sorted(ids)])) for k in ('imaging_quality', 'motion_smoothness')}
    panel['clip_frame_text_cosine_mean'] = float(np.mean([indexed['clip'][s]['scores']['clip_frame_text_cosine_mean'] for s in sorted(ids)]))
    return {'identity': [dict(r, seed_offset=42) for r in indexed['identity'].values()], 'mouth': indexed['mouth'],
            'panel': panel, 'reports': reports, 'config': config, 'observations': observations,
            'hashes': run['output_sha256'], 'hours': run['estimated_backend_gpu_hours'],
            'reload': profile.get('lora_reload'), 'result': run}


def main():
    from compare_anyid_context import summarize_pairs
    from compare_anyid_multiscale import motion_summary
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('ledger', 'runs-dir', 'scores-dir', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--stage', choices=['repeat', 'final'], required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Fresh audit path required')
    ledger = load_json(args.ledger)
    expected_thresholds = {'gain_retention': .6, 'gray_motion_min': .8, 'mouth_rate_drop_max': .05,
                           'mouth_paircoverage_min': .95, 'mouth_flow_min': .8, 'imaging_drop_max': .01, 'clip_drop_max': .005,
                           'repeat_identity_max': .02, 'repeat_imaging_max': .01, 'repeat_clip_max': .005,
                           'repeat_motion_min': .9, 'repeat_motion_max': 1.1}
    if ledger['thresholds'] != expected_thresholds:
        raise ValueError('Predeclared thresholds changed')
    quality, protocols, hours = {}, [], 0
    for stage in range(1, 2 if args.stage == 'repeat' else 3):
        report = load_json(args.scores_dir / f'lora_quality_stage{stage}_20261021.json')
        wall = load_json(args.scores_dir / f'lora_quality_stage{stage}_20261021_wall.json')
        if report['status'] != 'completed' or wall['returncode'] != 0 or wall['wall_seconds'] > 1830:
            raise ValueError('Quality not complete or exceeded budget')
        if (wall['gpus'] != 1 or not np.isfinite(wall['wall_seconds']) or wall['wall_seconds'] < 0
                or not np.isclose(wall['allocated_gpu_hours'], wall['wall_seconds'] / 3600)):
            raise ValueError('Quality allocation cost inconsistent')
        protocols.append((report['protocol'], report['weight_hashes']))
        hours += wall['allocated_gpu_hours']
        for row in report['records']:
            if row['sample_id'] in quality or not np.isfinite(list(row['scores'].values())).all():
                raise ValueError('Duplicate or nonfinite quality score')
            quality[row['sample_id']] = row
    if any(p != protocols[0] for p in protocols): raise ValueError('Quality protocol changed')
    names = ['baseline', 'historical1', 'alpha1'] + ([] if args.stage == 'repeat' else ['alpha05', 'alpha025'])
    arms = {a: arm_data(a, ledger, args.runs_dir, args.scores_dir, quality) for a in names}
    sample_ids = {r['sample_id'] for r in arms['baseline']['identity']}
    if set(quality) != {a + '__' + s for a in names for s in sample_ids}: raise ValueError('Extra or missing quality inputs')
    baseline = arms['baseline']
    for arm, data in arms.items():
        left, right = dict(baseline['config']['parameters']), dict(data['config']['parameters'])
        for p in (left, right): p.pop('lora_path', None); p.pop('lora_alpha', None)
        if left != right or data['config']['model_revisions'] != baseline['config']['model_revisions']:
            raise ValueError('Non-alpha generation settings differ')
        for kind in ('identity', 'clip', 'mouth'):
            a, b = baseline['reports'][kind], data['reports'][kind]
            if any(a.get(k) != b.get(k) for k in ('protocol', 'weight_hashes', 'threshold')):
                raise ValueError('Evaluator weights, threshold or protocol differs')
        for s in sample_ids:
            if any(data['observations'][s][k] != baseline['observations'][s][k] for k in ('seed', 'reference_pixels_sha256', 'generated_frames')):
                raise ValueError('Actual sample seed or reference pixels differ')
    actors = sorted({r['identity_id'] for r in baseline['identity']})
    pair = lambda a, b: summarize_pairs(arms[a]['identity'], arms[b]['identity'], actors, seed=42)
    repeat = repeat_checks(pair('historical1', 'alpha1'), arms['historical1']['panel'], arms['alpha1']['panel'],
                           sum(v == arms['historical1']['hashes'][s] for s, v in arms['alpha1']['hashes'].items()), 16, ledger['thresholds'])
    full = pair('baseline', 'alpha1')
    repeat['positive_full_identity_gain'] = (full['paired_videos'] == 16 and full['paired_actors'] == 8
                                               and full['identity_actor_mean_delta'] is not None and full['identity_actor_mean_delta'] > 0)
    repeat['passed'] = repeat['passed'] and repeat['positive_full_identity_gain']
    comparisons = {}
    for arm in names[2:]:
        identity = pair('baseline', arm)
        mouth = motion_summary(baseline['mouth'], arms[arm]['mouth'], baseline['reports']['mouth']['threshold'])
        deltas = {k: arms[arm]['panel'][k] - baseline['panel'][k] for k in baseline['panel']}
        checks = candidate_checks(identity, mouth, deltas, full['identity_actor_mean_delta'], ledger['thresholds'])
        comparisons[arm] = {'identity': identity, 'mouth': mouth, 'quality_deltas': deltas, 'checks': checks,
                            'passes_local_candidate_checks': repeat['passed'] and all(checks.values())}
    pilots = []
    for entry in ledger['entries']:
        if not entry['name'].startswith('pilot'): continue
        folder = args.runs_dir / entry['run_id']
        result, profile = load_json(folder / 'result.json'), load_json(folder / 'backend_metrics.json')
        reload = profile['lora_reload']
        if (result['status'] != 'completed' or result['source_sha256'] != entry['source_sha256']
                or sha256_file(folder / 'config.json') != entry['config_sha256']
                or reload['alpha'] != entry['alpha'] or reload['checkpoint_sha256'] != ledger['checkpoint_sha256']
                or reload['matched_pairs'] != 150 or reload['changed_base_tensors'] <= 0):
            raise ValueError('Pilot evidence differs')
        from ipvg.evaluate import evaluate_outputs
        pilot_manifest = folder / 'manifest.jsonl'
        if sha256_file(pilot_manifest) != entry['selected_manifest_sha256']:
            raise ValueError('Pilot manifest drift')
        media = evaluate_outputs(pilot_manifest, folder / 'outputs', expected_width=480, expected_height=352,
                                 expected_fps=16, expected_duration=49 / 16, fail_on_extra=True)
        if not media['ok'] or len(media['samples']) != 1 or media['samples'][0]['media']['decoded_frames'] != 49:
            raise ValueError('Pilot output incomplete')
        observations = profile['samples']
        if len(observations) != 1 or len(result['output_sha256']) != 1:
            raise ValueError('Pilot observations incomplete')
        observation = observations[0]
        sample = observation['sample_id']
        for k in ('seed', 'reference_pixels_sha256', 'generated_frames'):
            if observation[k] != arms['alpha1']['observations'][sample][k]:
                raise ValueError('Pilot seed/reference differs')
        file = folder / 'outputs' / (sample + '.mp4')
        if sha256_file(file) != result['output_sha256'][file.name]:
            raise ValueError('Pilot video hash drift')
        measured = result['resources']['gpus'] * result['backend_elapsed_seconds'] / 3600
        if not np.isfinite(measured) or measured < 0 or not np.isclose(measured, result['estimated_backend_gpu_hours']) or measured > entry['reserved_gpu_hours']:
            raise ValueError('Pilot budget inconsistent')
        pilots.append({'run_id': entry['run_id'], 'reload': reload, 'gpu_hours': measured})
        hours += measured
    hours += sum(arms[a]['hours'] for a in names if a not in ('baseline', 'historical1'))
    if hours > ledger['new_hard_cap_gpu_hours']: raise ValueError('New campaign budget exceeded')
    passing = [a for a in ('alpha05', 'alpha025') if a in comparisons and comparisons[a]['passes_local_candidate_checks']]
    decision = 'repeat_passed' if args.stage == 'repeat' and repeat['passed'] else 'repeat_failed_stop'
    chosen = None
    if args.stage == 'final' and repeat['passed']:
        chosen = max(passing, key=lambda a: comparisons[a]['identity']['identity_actor_mean_delta']) if passing else None
        decision = 'local_fusion_candidate_only' if chosen else 'fusion_strength_not_promoted'
    output = {'status': 'completed', 'stage': args.stage, 'decision': decision, 'selected_candidate': chosen,
              'repeat': repeat, 'quality_panel': {a: arms[a]['panel'] for a in names}, 'comparisons': comparisons,
              'pilots': pilots, 'thresholds': ledger['thresholds'], 'checkpoint_sha256': ledger['checkpoint_sha256'],
              'eval_manifest_sha256': ledger['eval_manifest_sha256'], 'actual_new_gpu_hours': hours,
              'new_hard_cap_gpu_hours': ledger['new_hard_cap_gpu_hours'], 'official_score': None, 'holdout_used': False,
              'human_action_validation': False, 'quality_protocol': protocols[0][0],
              'limitations': ['One seed, eight dev actors in acted talking-head domain; not SOTA or final base.',
                              'Gray motion and local mouth flow do not prove speech, expressions or natural action.',
                              'Native fusion strength, not new training or LoRA versus full-finetune comparison.',
                              'Missing CurricularFace/GME/Story/human and complete official MM2026 scorer.']}
    write_json(args.output, output)
    print('Audit completed:', decision, 'GPUh', hours)


if __name__ == '__main__': main()
