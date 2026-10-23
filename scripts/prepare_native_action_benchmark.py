"""Freeze a new reference-only development cohort; not official action ground truth."""
import argparse
from collections import Counter
import copy
import hashlib
from pathlib import Path
from ipvg.common import load_json, write_json, sha256_file
from ipvg.data import read_manifest, write_manifest
from ipvg.experiments import plan_experiment

PROMPTS = {
    'portrait_turn': 'A continuous close-up portrait of the reference person against a plain light-gray background. The person starts facing the camera, slowly turns their head to their left, then returns to face the camera. The camera stays fixed, with soft even lighting.',
    'halfbody_raise_hand': 'A continuous waist-up shot of the reference person in a simple room. The person starts with both hands lowered, raises their right hand to shoulder height, then lowers it again. The camera stays fixed, with soft even lighting.',
    'fullbody_walk': 'A continuous full-body shot of the reference person in a spacious room. The person takes a few natural steps from left to right across the frame. Their body remains within the frame. The camera stays fixed, with soft even lighting.',
    'fullbody_sit': 'A continuous full-body shot of the reference person standing beside a simple chair in a spacious room. The person turns toward the chair and sits down on it. Their body and the chair remain within the frame. The camera stays fixed, with soft even lighting.',
}


def selected_rows(original, prior):
    forbidden = {r['identity_id'] for r in prior} | {r['identity_id'] for r in original if r.get('split') != 'dev'}
    actors = sorted({r['identity_id'] for r in original if r.get('split') == 'dev'} - forbidden,
                    key=lambda a: hashlib.sha256(('native-action-v1:' + a).encode()).hexdigest())[:8]
    if len(actors) != 8: raise ValueError('Need eight additional development actors, distinct from old/train/holdout')
    by_actor = {}
    for r in sorted(original, key=lambda r: r['sample_id']):
        if r['identity_id'] in actors and r['split'] == 'dev': by_actor.setdefault(r['identity_id'], r)
    result = []
    for i, actor in enumerate(actors):
        source = by_actor[actor]
        for action in (('portrait_turn', 'halfbody_raise_hand')[i % 2], ('fullbody_walk', 'fullbody_sit')[(i // 2) % 2]):
            result.append({'sample_id': 'action_' + hashlib.sha256((actor + ':' + action).encode()).hexdigest()[:20],
                           'identity_id': actor, 'source_video_id': source['source_video_id'],
                           'reference_images': list(source['reference_images']), 'reference_sha256': source['reference_sha256'],
                           'prompt': PROMPTS[action], 'track': 1, 'split': 'dev', 'timeline': [], 'action_category': action,
                           'framing': 'portrait' if action == 'portrait_turn' else 'halfbody' if action == 'halfbody_raise_hand' else 'fullbody',
                           'cohort_notice': 'New reference-only challenge-inspired prompts; source references are from talking-head recordings, no action GT or target video consumed'})
    if Counter(r['action_category'] for r in result) != Counter({k: 4 for k in PROMPTS}): raise ValueError('Action allocation must be balanced')
    return result


ACTION_MANIFEST_SHA256 = '146a769beb10e330aa062784a761be7e12fd3a63b775b89221ede2f489245d39'
BASELINE_HISTORY_RUN_ID = 'action_standin_main_20261023_r001'
PARTITION_GATES = {'gme_aesr_min_delta': .02, 'cur_score_min_delta': -.02,
                   'arc_score_min_delta': -.02, 'imaging_quality_min_delta': -.02,
                   'identity_coverage_must_not_decrease': True, 'max_missing_identity_videos': 2}


def partition_rows(manifest, expected_sha256=None):
    if sha256_file(manifest) != (ACTION_MANIFEST_SHA256 if expected_sha256 is None else expected_sha256):
        raise ValueError('The original frozen action manifest changed')
    rows = read_manifest(manifest)
    if (len(rows) != 16 or len({r['sample_id'] for r in rows}) != 16
            or len({r['identity_id'] for r in rows}) != 8
            or Counter(r['action_category'] for r in rows) != Counter({k: 4 for k in PROMPTS})):
        raise ValueError('Expected the same sixteen inputs/eight actors/four balanced actions')
    for row in rows:
        if (row.get('split') != 'dev' or row.get('track') != 1 or len(row['reference_images']) != 1
                or any(k in row for k in ('video_path', 'media_sha256', 'control_video_path'))
                or row['prompt'] != PROMPTS[row['action_category']]):
            raise ValueError('Reference-only original development inputs required')
        if sha256_file(row['reference_images'][0]) != row['reference_sha256']:
            raise ValueError('Original reference changed')
    pilots = {name: min((r for r in rows if r['action_category'] == action),
                       key=lambda r: hashlib.sha256(r['sample_id'].encode()).hexdigest())
              for name, action in (('pilot_portrait', 'portrait_turn'), ('pilot_fullbody', 'fullbody_sit'))}
    return rows, pilots


def prepare_standin_partition(args):
    if args.output_dir.exists():
        raise FileExistsError('Choose a fresh partition experiment directory')
    if not args.output_dir.parent.is_dir():
        raise FileNotFoundError('Output parent must exist')
    rows, pilots = partition_rows(args.manifest)
    prior_path = args.prior_work_dir / 'ledger.json'
    prior = load_json(prior_path)
    matches = [e for e in prior['runs'] if e['arm'] == 'standin' and e['phase'] == 'main']
    if len(matches) != 1 or matches[0]['run_id'] != BASELINE_HISTORY_RUN_ID:
        raise ValueError('Use the frozen historical action Stand-In main template')
    parent = matches[0]; folder = args.runs_dir / parent['run_id']
    template = load_json(folder / 'config.json'); historical = load_json(folder / 'result.json')
    if load_json(folder / 'backend_metrics.json').get('attention_backend') != 'FA2':
        raise ValueError('Historical native attention backend must be FA2')
    if (historical['status'] != 'completed' or historical['source_sha256'] != parent['source_sha256']
            or sha256_file(folder / 'config.json') != parent['config_sha256']
            or sha256_file(folder / 'manifest.jsonl') != sha256_file(args.manifest)
            or parent['manifest_sha256'] != sha256_file(args.manifest)
            or read_manifest(folder / 'manifest.jsonl') != rows
            or Path(template['runs_dir']).resolve() != args.runs_dir.resolve()
            or template['parameters'].get('attention_mode', 'native') != 'native'):
        raise ValueError('Historical native template/data/source drift')
    output_hashes = {r['sample_id'] + '.mp4': sha256_file(folder / 'outputs' / (r['sample_id'] + '.mp4')) for r in rows}
    if historical['output_sha256'] != output_hashes:
        raise ValueError('Historical native outputs changed')
    for k, v in (('width', 480), ('height', 352), ('num_frames', 49), ('fps', 16),
                 ('num_inference_steps', 20), ('cfg_scale', 5), ('seed', 42)):
        value = template['parameters'].get(k)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != v:
            raise ValueError('Historical Stand-In specification differs: ' + k)
    run_ids = [f'standin_partition_{arm}_{phase}_20261023_r001'
               for arm in ('native', 'partition') for phase in (*pilots, 'main')]
    if any((args.runs_dir / run_id).exists() for run_id in run_ids):
        raise FileExistsError('Partition run IDs must be fresh')
    history_locks = {name: sha256_file(folder / name)
                     for name in ('config.json', 'manifest.jsonl', 'result.json', 'backend_metrics.json')}
    args.output_dir.mkdir()
    manifest = args.output_dir / 'manifest.jsonl'
    manifest.write_bytes(args.manifest.read_bytes())
    pilot_paths = {}
    for phase, row in pilots.items():
        pilot_paths[phase] = args.output_dir / (phase + '.jsonl')
        write_manifest(pilot_paths[phase], [row])
    configs = args.output_dir / 'configs'; configs.mkdir()
    entries = []
    # Freeze every pilot before either main; execution must respect this order too.
    for phase in (*pilots, 'main'):
        for condition in ('native', 'partition'):
            c = copy.deepcopy(template)
            c['parameters']['attention_mode'] = 'native' if condition == 'native' else 'partition_balanced'
            c['parameters']['expected_attention_backend'] = 'FA2'
            c.update(run_id=f'standin_partition_{condition}_{phase}_20261023_r001',
                     manifest=str(manifest if phase == 'main' else pilot_paths[phase]),
                     hypothesis='Same frozen action cohort, original reference/prompt/seed; native versus partition-balanced attention only')
            timeout = 7200 if phase == 'main' else 900
            c['resources'] = {'gpus': 1, 'timeout_s': timeout, 'max_gpu_hours': timeout / 3600}
            path = configs / (c['run_id'] + '.json'); write_json(path, c)
            plan = plan_experiment(path, root=args.repo_root)
            entries.append({'arm': 'standin', 'condition': condition, 'phase': phase, 'run_id': c['run_id'],
                            'config_sha256': sha256_file(path), 'manifest_sha256': sha256_file(c['manifest']),
                            'source_sha256': plan['source_sha256'], 'count': 16 if phase == 'main' else 1,
                            'reserved_gpu_hours': timeout / 3600})
    ledger = {'schema_version': 1, 'benchmark_version': 'standin_partition_20261023_r001',
              'runs': entries, 'eval_manifest': str(manifest), 'eval_manifest_sha256': sha256_file(manifest),
              'original_manifest': str(args.manifest), 'original_manifest_sha256': sha256_file(args.manifest),
              'prior_ledger_sha256': sha256_file(prior_path), 'prior_work_dir': str(args.prior_work_dir),
              'baseline_history_run_id': BASELINE_HISTORY_RUN_ID,
              'baseline_history_entry': parent, 'baseline_history_file_sha256': history_locks,
              'baseline_history_output_sha256': output_hashes,
              'pilot_sample_ids': {k: r['sample_id'] for k, r in pilots.items()},
              'pilot_historical_output_sha256': {k: output_hashes[r['sample_id'] + '.mp4'] for k, r in pilots.items()},
              'pilot_before_main_required': True, 'cohort': 'same_frozen_development_action_framing_v1',
              'prompt_templates': PROMPTS, 'gates': PARTITION_GATES, 'generation_reserved_gpu_hours': 5,
              'evaluation_reserved_gpu_hours': 2, 'single_engineering_repair_allowance_gpu_hours': .5,
              'new_hard_cap_gpu_hours': 8, 'original_overall_cap_gpu_hours': 51.5,
              'prior_gpu_hours_upper': 5.31, 'expected_main_videos': 32,
              'action_ground_truth_available': False, 'training_performed': False, 'retry_count': 0}
    write_json(args.output_dir / 'ledger.json', ledger)
    return ledger


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('manifest', 'prior-manifest', 'prior-work-dir', 'output-dir', 'runs-dir', 'repo-root'):
        p.add_argument('--' + n, type=Path, required=True)
    p.add_argument('--structure-mode', choices=('native', 'standin_partition'), default='native')
    args = p.parse_args()
    if args.structure_mode == 'standin_partition':
        prepare_standin_partition(args)
        return
    if args.output_dir.exists(): raise FileExistsError('Choose a fresh action experiment directory')
    if not args.output_dir.parent.is_dir(): raise FileNotFoundError('Output parent must exist')
    rows = selected_rows(read_manifest(args.manifest), read_manifest(args.prior_manifest))
    for r in rows:
        if sha256_file(r['reference_images'][0]) != r['reference_sha256']: raise ValueError('Original reference changed')
    args.output_dir.mkdir()
    manifest = args.output_dir / 'manifest.jsonl'; pilot = args.output_dir / 'pilot.jsonl'
    write_manifest(manifest, rows); write_manifest(pilot, rows[:1])
    prior_ledger = load_json(args.prior_work_dir / 'ledger.json')
    configs = args.output_dir / 'configs'; configs.mkdir()
    entries = []
    for arm in ('vace', 'anyid', 'standin'):
        parent = next(e for e in prior_ledger['runs'] if e['arm'] == arm and e['phase'] == 'main')
        template = load_json(args.runs_dir / parent['run_id'] / 'config.json')
        for phase in ('pilot', 'main'):
            c = copy.deepcopy(template)
            timeout = 600 if phase == 'pilot' else 7200 if arm == 'standin' else 2400
            c.update(run_id=f'action_{arm}_{phase}_20261023_r001', manifest=str(pilot if phase == 'pilot' else manifest),
                     hypothesis='Independent additional development identities and four fixed action/framing prompts, same native systems and evaluation, no model changes or action-ground-truth claims')
            c['resources'] = {'gpus': 1, 'timeout_s': timeout, 'max_gpu_hours': timeout / 3600}
            config_path = configs / (c['run_id'] + '.json')
            write_json(config_path, c)
            plan = plan_experiment(config_path, root=args.repo_root)
            entries.append({'arm': arm, 'phase': phase, 'run_id': c['run_id'], 'config_sha256': sha256_file(config_path),
                            'manifest_sha256': sha256_file(c['manifest']), 'source_sha256': plan['source_sha256'],
                            'count': 1 if phase == 'pilot' else 16, 'reserved_gpu_hours': timeout / 3600})
    write_json(args.output_dir / 'ledger.json', {'runs': entries, 'eval_manifest_sha256': sha256_file(manifest),
      'eval_manifest': str(manifest), 'cohort': 'additional_development_action_framing_v1', 'prompt_templates': PROMPTS,
      'prior_manifest_sha256': sha256_file(args.prior_manifest), 'original_manifest_sha256': sha256_file(args.manifest),
      'generation_reserved_gpu_hours': sum(e['reserved_gpu_hours'] for e in entries), 'evaluation_reserved_gpu_hours': 2,
      'single_engineering_repair_allowance_gpu_hours': .5, 'new_hard_cap_gpu_hours': 7, 'original_overall_cap_gpu_hours': 51.5,
      'prior_gpu_hours_upper': 4.35, 'score_protocol': 'ipvg_cross_model_comparison_v1', 'expected_main_videos': 48,
      'action_ground_truth_available': False, 'new_training_or_model_structure_changes': False})
    print('Frozen eight additional dev actors,16 reference-only inputs,balanced four action categories; native budget <=7GPUh')


if __name__ == '__main__': main()
