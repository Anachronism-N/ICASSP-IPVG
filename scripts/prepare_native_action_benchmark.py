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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('manifest', 'prior-manifest', 'prior-work-dir', 'output-dir', 'runs-dir', 'repo-root'):
        p.add_argument('--' + n, type=Path, required=True)
    args = p.parse_args()
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
