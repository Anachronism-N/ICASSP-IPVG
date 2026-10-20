"""Audit finite dense-caption training and same-input development quality, not official scores."""
import argparse
from pathlib import Path
import numpy as np
from ipvg.common import load_json, sha256_file, write_json
from ipvg.data import read_manifest

ARMS = ('baseline', 'lr1', 'lr5')


def training_summary(folder, expected_steps):
    result = load_json(folder / 'result.json')
    config = load_json(folder / 'config.json')
    if result['status'] != 'completed' or result['task'] != 'training':
        raise ValueError('Training did not complete')
    if sha256_file(folder / 'manifest.jsonl') != result['selected_manifest_sha256']:
        raise ValueError('Training manifest drift')
    ranks = [load_json(folder / f'rank_{i}.json') for i in range(config['resources']['gpus'])]
    initial, final, epoch_losses, peaks = set(), set(), [], []
    for index, rank in enumerate(ranks):
        if (rank['rank'] != index or rank['world_size'] != len(ranks) or rank['status'] != 'passed'
                or rank['actual_optimizer_steps'] != expected_steps or rank['backward_calls'] != expected_steps
                or not rank['rank_parameters_synchronized'] or not rank['parameters_finite']
                or rank['changed_trainable_tensors'] <= 0 or rank['parameter_max_abs_change'] <= 0
                or len(rank['steps']) != expected_steps or len(rank['losses']) != expected_steps):
            raise ValueError('Invalid rank training evidence')
        for step, record in enumerate(rank['steps'], 1):
            lr = config['parameters']['learning_rate']
            expected_lr = lr / 3 if step <= (3 if len(ranks) == 2 else 5) else lr
            if (record['optimizer_step'] != step or not record['completed'] or not record['gradients_finite']
                    or record['nonzero_gradient_tensors'] <= 0
                    or len(record['learning_rates']) != 1
                    or not np.isclose(record['learning_rates'][0], expected_lr, rtol=1e-7, atol=0)):
                raise ValueError('Optimizer gradients, steps or actual learning rate differ')
        values = [record['value'] for record in rank['losses']]
        if any(not record['finite'] for record in rank['losses']) or not np.isfinite(values).all():
            raise ValueError('Nonfinite observed losses')
        epoch_losses.append([float(np.mean(part)) for part in np.array_split(values, config['parameters']['num_epochs'])])
        initial.add(rank['initial_trainable_sha256'])
        final.add(rank['final_trainable_sha256'])
        peaks.append(rank['resource_end']['peak_allocated_bytes'] / 1e9)
    if len(initial) != 1 or len(final) != 1 or initial == final:
        raise ValueError('Ranks diverged or parameters unchanged')
    save_steps = config['parameters']['save_steps']
    expected_artifacts = {f'step-{i}.safetensors' for i in range(save_steps, expected_steps + 1, save_steps)}
    expected_artifacts.add(f'step-{expected_steps}.safetensors')
    if set(result['artifact_sha256']) != expected_artifacts:
        raise ValueError('Checkpoint coverage differs from save schedule')
    for name, digest in result['artifact_sha256'].items():
        if sha256_file(folder / 'outputs' / name) != digest:
            raise ValueError('Checkpoint hash drift')
    hours = result['estimated_backend_gpu_hours']
    if not np.isclose(hours, len(ranks) * result['backend_elapsed_seconds'] / 3600) or hours > config['resources']['max_gpu_hours']:
        raise ValueError('Training budget inconsistency')
    return {'run_id': result['run_id'], 'status': 'passed', 'gpus': len(ranks),
            'actual_optimizer_steps_per_rank': expected_steps,
            'initial_trainable_sha256': next(iter(initial)), 'final_trainable_sha256': next(iter(final)),
            'trainable_parameter_count': ranks[0]['trainable_parameter_count'],
            'changed_trainable_tensors_per_rank': [r['changed_trainable_tensors'] for r in ranks],
            'nominal_learning_rate': config['parameters']['learning_rate'],
            'first_step_learning_rate': ranks[0]['steps'][0]['learning_rates'][0],
            'last_step_learning_rate': ranks[0]['steps'][-1]['learning_rates'][0],
            'epoch_mean_loss_all_ranks': np.mean(epoch_losses, axis=0).tolist(),
            'peak_allocated_vram_gb_per_rank': peaks,
            'backend_elapsed_seconds': result['backend_elapsed_seconds'], 'allocated_gpu_hours': hours,
            'checkpoint_sha256': result['artifact_sha256'],
            'code_commit': result['code_commit'], 'source_sha256': result['source_sha256']}


def indexed(records, expected):
    result = {r['sample_id']: r for r in records}
    if len(result) != len(records) or set(result) != set(expected):
        raise ValueError('Score records missing, extra or duplicate')
    return result


def main():
    from compare_anyid_context import summarize_pairs
    from compare_anyid_multiscale import motion_summary
    from ipvg.evaluate import evaluate_outputs
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs-dir', 'scores-dir', 'train-manifest', 'eval-manifest', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose fresh aggregate output')
    train, dev = read_manifest(args.train_manifest), read_manifest(args.eval_manifest)
    if (len(train) != 32 or len(dev) != 16 or any(r['split'] != 'train' for r in train)
            or any(r['split'] != 'dev' for r in dev) or len({r['identity_id'] for r in dev}) != 8
            or {r['identity_id'] for r in train} & {r['identity_id'] for r in dev}
            or {r['source_video_id'] for r in train} & {r['source_video_id'] for r in dev}):
        raise ValueError('Unexpected cohort or split leakage')
    training = [training_summary(args.runs_dir / f'real_vace_{name}_20261020_r001', steps)
                for name, steps in [('single_pilot', 2), ('ddp_pilot_lr1', 16), ('ddp_pilot_lr5', 16), ('main_lr1', 64), ('main_lr5', 64)]]
    if (len({r['initial_trainable_sha256'] for r in training}) != 1
            or len({(r['code_commit'], r['source_sha256']) for r in training}) != 1):
        raise ValueError('Training initialization or source differs')
    for stage in ('ddp_pilot', 'main'):
        configs = [load_json(args.runs_dir / f'real_vace_{stage}_{arm}_20261020_r001/config.json') for arm in ('lr1', 'lr5')]
        left, right = [dict(c['parameters']) for c in configs]
        left.pop('learning_rate'); right.pop('learning_rate')
        if left != right or configs[0]['model_revisions'] != configs[1]['model_revisions']:
            raise ValueError('Learning rate is not the only training factor')
    quality, quality_protocol, quality_weights, quality_hours = {}, None, None, 0
    for shard in (0, 1):
        report = load_json(args.scores_dir / f'real_vace_quality_shard{shard}_20261020.json')
        if report['status'] != 'completed':
            raise ValueError('Quality shard incomplete')
        if quality_protocol is not None and (quality_protocol != report['protocol'] or quality_weights != report['weight_hashes']):
            raise ValueError('Quality protocols differ')
        quality_protocol, quality_weights = report['protocol'], report['weight_hashes']
        quality_hours += report['resource']['allocated_gpu_hours']
        for record in report['records']:
            if record['sample_id'] in quality or not np.isfinite(list(record['scores'].values())).all():
                raise ValueError('Duplicate or nonfinite quality score')
            quality[record['sample_id']] = record
    identities, mouths, panels, profiles, generation_hours = {}, {}, {}, {}, 0
    protocols = {}
    ids = {r['sample_id'] for r in dev}
    actors = sorted({r['identity_id'] for r in dev})
    for arm in ARMS:
        run = args.runs_dir / f'real_vace_eval_{arm}_20261020_r001'
        result, profile = load_json(run / 'result.json'), load_json(run / 'backend_metrics.json')
        if result['status'] != 'completed' or sha256_file(run / 'manifest.jsonl') != sha256_file(args.eval_manifest):
            raise ValueError('Generation incomplete or input differs')
        media = evaluate_outputs(args.eval_manifest, run / 'outputs', expected_width=480, expected_height=352,
                                 expected_fps=16, expected_duration=49 / 16, fail_on_extra=True)
        if not media['ok'] or any(r['media']['decoded_frames'] != 49 for r in media['samples']):
            raise ValueError('Generation media differs')
        config = load_json(run / 'config.json')
        settings = {k: config['parameters'][k] for k in ('width', 'height', 'num_frames', 'fps', 'num_inference_steps', 'cfg_scale', 'seed', 'reference_index')}
        if 'generation' in protocols and protocols['generation'] != settings:
            raise ValueError('Generation settings differ')
        protocols['generation'] = settings
        profiles[arm] = indexed(profile['samples'], ids)
        reports = {kind: load_json(args.scores_dir / f'real_vace_{arm}_{kind}_20261020.json') for kind in ('identity', 'clip', 'mouth')}
        for kind, report in reports.items():
            frozen = (report['protocol'], report.get('weight_hashes'),
                      report.get('threshold') if kind == 'mouth' else None)
            if kind in protocols and protocols[kind] != frozen:
                raise ValueError('Evaluator protocol or weights differ')
            protocols[kind] = frozen
        identity, clip, mouth = [indexed(reports[k]['records'], ids) for k in ('identity', 'clip', 'mouth')]
        for sample in ids:
            path = run / 'outputs' / (sample + '.mp4')
            digest = sha256_file(path)
            q = quality[arm + '__' + sample]
            if not all(h == digest for h in (result['output_sha256'][path.name], identity[sample]['output_sha256'],
                                            clip[sample]['video_sha256'], mouth[sample]['video_sha256'], q['video_sha256'])):
                raise ValueError('Evaluation output hashes differ')
        if arm != 'baseline':
            reload = profile['lora_reload']
            trained = next(r for r in training if r['run_id'] == f'real_vace_main_{arm}_20261020_r001')
            if reload['matched_pairs'] != 150 or reload['changed_base_tensors'] <= 0 or reload['checkpoint_sha256'] != trained['checkpoint_sha256']['step-64.safetensors']:
                raise ValueError('Real checkpoint reload not proven')
        identities[arm] = [dict(r, seed_offset=42) for r in identity.values()]
        mouths[arm] = mouth
        panels[arm] = {metric: float(np.mean([quality[arm + '__' + s]['scores'][metric] for s in ids])) for metric in ('imaging_quality', 'motion_smoothness')}
        panels[arm]['clip_frame_text_cosine_mean'] = float(np.mean([clip[s]['scores']['clip_frame_text_cosine_mean'] for s in ids]))
        panels[arm]['reload'] = profile.get('lora_reload')
        generation_hours += result['estimated_backend_gpu_hours']
    if set(quality) != {arm + '__' + s for arm in ARMS for s in ids}:
        raise ValueError('Quality panel differs from all48 inputs')
    for sample in ids:
        if len({(profiles[a][sample]['seed'], profiles[a][sample]['reference_pixels_sha256']) for a in ARMS}) != 1:
            raise ValueError('Generation seed or reference differs')
    comparisons = {}
    for arm in ('lr1', 'lr5'):
        identity = summarize_pairs(identities['baseline'], identities[arm], actors, seed=42)
        identity['baseline'] = identity.pop('narrow'); identity['candidate'] = identity.pop('wide')
        local = motion_summary(mouths['baseline'], mouths[arm], reports['mouth']['threshold'])
        local['baseline'] = local.pop('single'); local['candidate'] = local.pop('dual')
        comparisons[arm] = {'identity': identity, 'mouth': local,
                            'quality_mean_deltas': {k: panels[arm][k] - panels['baseline'][k] for k in ('imaging_quality', 'motion_smoothness', 'clip_frame_text_cosine_mean')}}
    output = {'status': 'completed', 'decision': 'small_domain_development_comparison_not_final_base',
              'training': training, 'quality_panel': panels, 'comparisons': comparisons,
              'train_manifest_sha256': sha256_file(args.train_manifest), 'eval_manifest_sha256': sha256_file(args.eval_manifest),
              'train_videos': 32, 'train_actors': 16, 'dev_videos_per_arm': 16, 'dev_actors': 8,
              'training_gpu_hours': sum(r['allocated_gpu_hours'] for r in training),
              'generation_gpu_hours': generation_hours, 'quality_profile_gpu_hours': quality_hours,
              'quality_protocol': quality_protocol, 'quality_weight_hashes': quality_weights,
              'official_score': None, 'holdout_used': False, 'human_validation_passed': False,
              'limitations': ['Small acted talking-head domain, one seed; not KeyID/AnyID training or SOTA.',
                              'Stochastic flow-matching timestep losses are not held-out quality or comparable fixed-target losses.',
                              'Missing faces stay missing; paired identities and valid-video means have different denominators.',
                              'AMT smoothness can favor little motion; mouth control is not speech, expression or lip-sync truth.',
                              'Missing CurricularFace, GME, Story and official human scores; no complete MM2026 final score.',
                              'Quality profile GPUh excludes import/preflight; managed command wall upper bound recorded separately.']}
    write_json(args.output, output)
    print('Audited five training runs and48 same-input development videos')


if __name__ == '__main__':
    main()
