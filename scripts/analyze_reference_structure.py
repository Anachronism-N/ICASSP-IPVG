"""Audit reference-difference routing, preserving frozen original quality protocols."""
import argparse
from pathlib import Path
import numpy as np
from ipvg.common import load_json, write_json, sha256_file
from ipvg.data import read_manifest


def routing_check(observation, gate):
    if (observation['mode'] != 'reference_difference_temporal_v1' or observation['gate'] != gate
            or observation['reference_frames'] != 1 or observation['outer_calls'] != 40 or observation['native_calls'] != 80
            or observation['hint_layers'] != 15 or observation['grid'] != [14, 22, 30]
            or not observation['reference_prefix_exact']):
        raise ValueError('Incorrect structural routing observation')
    for k in ('body_max_abs_change', 'reference_temporal_centered_max_abs', 'body_temporal_mean_max_abs_error'):
        v = observation[k]
        if isinstance(v, bool) or not isinstance(v, (float, int)) or not np.isfinite(v) or v < 0:
            raise ValueError('Invalid routing numerical evidence')
    if gate == 0 and observation['body_max_abs_change'] <= 0:
        raise ValueError('Shared-reference routing never changed video hints')
    if gate == 1 and (observation['body_max_abs_change'] != 0 or observation['body_temporal_mean_max_abs_error'] != 0):
        raise ValueError('Control routing changed original hints')


def run_check(folder, entry, checkpoint, expected_count):
    from ipvg.evaluate import evaluate_outputs
    run, config, profile = [load_json(folder / f'{n}.json') for n in ('result', 'config', 'backend_metrics')]
    if (run['status'] != 'completed' or run['run_id'] != entry['run_id']
            or run['source_sha256'] != entry['source_sha256']
            or sha256_file(folder / 'config.json') != entry['config_sha256']
            or sha256_file(folder / 'manifest.jsonl') != entry['manifest_sha256']):
        raise ValueError('Structural run source/input/config drift')
    reload = profile['lora_reload']
    if reload['checkpoint_sha256'] != checkpoint or reload['alpha'] != 1 or reload['matched_pairs'] != 150 or reload['changed_base_tensors'] <= 0:
        raise ValueError('Structural checkpoint reload not proven')
    media = evaluate_outputs(folder / 'manifest.jsonl', folder / 'outputs', expected_width=480, expected_height=352,
                             expected_fps=16, expected_duration=49 / 16, fail_on_extra=True)
    if (not media['ok'] or len(media['samples']) != expected_count or len(profile['samples']) != expected_count
            or any(r['media']['decoded_frames'] != 49 for r in media['samples'])):
        raise ValueError('Structural media incomplete or specification differs')
    for s in profile['samples']:
        routing_check(s['reference_routing'], entry['gate'])
        path = folder / 'outputs' / (s['sample_id'] + '.mp4')
        if sha256_file(path) != run['output_sha256'][path.name]:
            raise ValueError('Actual structural video hash drift')
    hours = run['resources']['gpus'] * run['backend_elapsed_seconds'] / 3600
    if (not np.isfinite(hours) or hours < 0 or hours > entry['reserved_gpu_hours']
            or not np.isclose(hours, run['estimated_backend_gpu_hours'])):
        raise ValueError('Structural run budget inconsistent')
    return run, config, profile, hours


def main():
    from analyze_lora_alpha import repeat_checks, candidate_checks
    from compare_anyid_context import summarize_pairs
    from compare_anyid_multiscale import motion_summary
    from ipvg.evaluate import evaluate_outputs
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('ledger', 'runs-dir', 'scores-dir', 'output'): p.add_argument('--' + n, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError('Fresh audit output required')
    ledger = load_json(args.ledger)
    expected = {'gain_retention': .6, 'gray_motion_min': .8, 'mouth_rate_drop_max': .05,
                'mouth_paircoverage_min': .95, 'mouth_flow_min': .8, 'imaging_drop_max': .01, 'clip_drop_max': .005,
                'repeat_identity_max': .02, 'repeat_imaging_max': .01, 'repeat_clip_max': .005,
                'repeat_motion_min': .9, 'repeat_motion_max': 1.1}
    if ledger['thresholds'] != expected: raise ValueError('Predeclared structural thresholds changed')
    manifest = Path(ledger['eval_manifest'])
    rows = read_manifest(manifest); actor_by_id = {r['sample_id']: r['identity_id'] for r in rows}
    if len(rows) != 16 or len(set(actor_by_id.values())) != 8 or sha256_file(manifest) != ledger['eval_manifest_sha256']:
        raise ValueError('Frozen structural development cohort differs')
    from mm2026_quality_eval import PROTOCOL as QUALITY_PROTOCOL, AMT_SHA256, MUSIQ_SHA256
    ids = set(actor_by_id); quality = {}; quality_protocol = None; cost = 0
    for shard in (0, 1):
        report = load_json(args.scores_dir / f'reference_quality_shard{shard}_20261021.json')
        wall = load_json(args.scores_dir / f'reference_quality_shard{shard}_20261021_wall.json')
        current = (report['protocol'], report['weight_hashes'])
        if (report['protocol'] != QUALITY_PROTOCOL or report['weight_hashes'].get('amt') != AMT_SHA256
                or report['weight_hashes'].get('musiq') != MUSIQ_SHA256
                or report['weight_hashes'].get('config') != 'a77746791b1d02c839f40791fb82adb28c5058cf3735b2657a4a965a789ce0a8'):
            raise ValueError('Quality scorer differs from frozen original protocol and weights')
        if (report['status'] != 'completed' or wall['returncode'] != 0 or wall['gpus'] != 1
                or not np.isfinite(wall['wall_seconds']) or not 0 <= wall['wall_seconds'] <= 1830
                or not np.isclose(wall['allocated_gpu_hours'], wall['wall_seconds'] / 3600)
                or (quality_protocol is not None and current != quality_protocol)):
            raise ValueError('Quality protocol/weights or allocation differs')
        quality_protocol = current; cost += wall['allocated_gpu_hours']
        for r in report['records']:
            if r['sample_id'] in quality or not np.isfinite(list(r['scores'].values())).all():
                raise ValueError('Duplicate/nonfinite structural quality score')
            quality[r['sample_id']] = r
    names = ledger['run_names']
    if set(quality) != {a + '__' + s for a in names for s in ids}:
        raise ValueError('Structural quality cohort incomplete')
    data = {}; run_evidence = []
    for arm, run_id in names.items():
        folder = args.runs_dir / run_id
        if arm.startswith('gate'):
            entry = next(r for r in ledger['entries'] if r['name'] == arm)
            run, config, profile, hours = run_check(folder, entry, ledger['checkpoint_sha256'], 16)
            cost += hours
            run_evidence.append({'run_id':run_id, 'status':run['status'], 'code_commit':run['code_commit'],
                                 'source_sha256':run['source_sha256'], 'actual_gpu_hours':hours,
                                 'reload':profile['lora_reload'], 'routing_observations':[r['reference_routing'] for r in profile['samples']]})
            prefix = 'reference_' + arm; date = '20261021'
        else:
            run, config, profile = [load_json(folder / f'{n}.json') for n in ('result', 'config', 'backend_metrics')]
            if run['status'] != 'completed' or sha256_file(folder / 'manifest.jsonl') != ledger['eval_manifest_sha256']:
                raise ValueError('Historical control incomplete or input differs')
            if arm == 'historical1':
                reload = profile['lora_reload']
                if (reload['checkpoint_sha256'] != ledger['checkpoint_sha256'] or reload.get('alpha', 1) != 1
                        or reload['matched_pairs'] != 150 or reload['changed_base_tensors'] <= 0):
                    raise ValueError('Historical structural checkpoint differs')
            prefix = 'real_vace_baseline' if arm == 'baseline' else 'lora_alpha1'
            date = '20261020' if arm == 'baseline' else '20261021'
        reports = {k: load_json(args.scores_dir / f'{prefix}_{k}_{date}.json') for k in ('identity', 'clip', 'mouth')}
        index = {}
        for kind, report in reports.items():
            records = {r['sample_id']:r for r in report['records']}
            if len(records) != 16 or len(report['records']) != 16 or set(records) != ids:
                raise ValueError('Structural scorer coverage differs')
            if any(r['identity_id'] != actor_by_id[s] for s,r in records.items()):
                raise ValueError('Scorer actor mapping differs')
            if kind in ('clip', 'mouth') and report['manifest_sha256'] != ledger['eval_manifest_sha256']:
                raise ValueError('Scorer manifest hash drift')
            if kind == 'mouth' and report['calibration_status'] != 'passed_local_activity_controls':
                raise ValueError('Mouth control calibration not passed')
            if kind == 'clip' and any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) for r in records.values() for v in r['scores'].values()):
                raise ValueError('Malformed CLIP score')
            index[kind] = records
        observations = {r['sample_id']:r for r in profile['samples']}
        for s in ids:
            path = folder / 'outputs' / (s + '.mp4'); digest = sha256_file(path)
            q = quality[arm + '__' + s]
            if any(h != digest for h in (run['output_sha256'][path.name], index['identity'][s]['output_sha256'],
                                         index['clip'][s]['video_sha256'], index['mouth'][s]['video_sha256'], q['video_sha256'])):
                raise ValueError('Structural evaluator/media hashes differ')
        panel = {k:float(np.mean([quality[arm+'__'+s]['scores'][k] for s in sorted(ids)])) for k in ('imaging_quality','motion_smoothness')}
        panel['clip_frame_text_cosine_mean'] = float(np.mean([index['clip'][s]['scores']['clip_frame_text_cosine_mean'] for s in sorted(ids)]))
        data[arm] = {'run':run,'config':config,'reports':reports,'identity':[dict(r,seed_offset=42) for r in index['identity'].values()],
                     'mouth':index['mouth'],'observations':observations,'panel':panel}
    base = data['baseline']
    for arm,item in data.items():
        a,b = dict(base['config']['parameters']),dict(item['config']['parameters'])
        for d in (a,b):
            for k in ('lora_path','lora_alpha','reference_routing','reference_temporal_gate'): d.pop(k,None)
        if a != b or item['config']['model_revisions'] != base['config']['model_revisions']:
            raise ValueError('Nonstructural generation settings differ')
        for kind in ('identity','clip','mouth'):
            if any(item['reports'][kind].get(k) != base['reports'][kind].get(k) for k in ('protocol','weight_hashes','threshold')):
                raise ValueError('Frozen evaluator protocol or weights changed')
        for s in ids:
            if any(item['observations'][s][k] != base['observations'][s][k] for k in ('seed','reference_pixels_sha256','generated_frames')):
                raise ValueError('Structural sample seed/reference differed')
    actors=sorted(set(actor_by_id.values()))
    pair=lambda a,b:summarize_pairs(data[a]['identity'],data[b]['identity'],actors,seed=42)
    exact=sum(data['gate1']['run']['output_sha256'][s+'.mp4']==data['historical1']['run']['output_sha256'][s+'.mp4'] for s in ids)
    repeat=repeat_checks(pair('historical1','gate1'),data['historical1']['panel'],data['gate1']['panel'],exact,16,ledger['thresholds'])
    full=pair('baseline','gate1'); candidate=pair('baseline','gate0')
    mouth=motion_summary(base['mouth'],data['gate0']['mouth'],base['reports']['mouth']['threshold'])
    deltas={k:data['gate0']['panel'][k]-base['panel'][k] for k in base['panel']}
    checks=candidate_checks(candidate,mouth,deltas,full['identity_actor_mean_delta'],ledger['thresholds'])
    checks['sixteen_scoreable_mouth_videos'] = mouth['dual']['scoreable_videos'] == 16
    checks['sixteen_paired_mouth_videos'] = mouth['paired_videos'] == 16
    repeat['passed']=repeat['passed'] and full['paired_videos']==16 and full['paired_actors']==8 and full['identity_actor_mean_delta'] is not None and full['identity_actor_mean_delta']>0
    pilots=[]
    for entry in ledger['entries']:
        if not entry['name'].startswith('pilot'):continue
        run,config,profile,hours=run_check(args.runs_dir/entry['run_id'],entry,ledger['checkpoint_sha256'],1)
        cost+=hours; pilots.append({'run_id':entry['run_id'],'status':run['status'],'actual_gpu_hours':hours,'routing':profile['samples'][0]['reference_routing']})
    if cost>ledger['new_hard_cap_gpu_hours']:raise ValueError('Structural hard budget exceeded')
    output={'status':'completed','decision':'local_reference_routing_candidate_only' if repeat['passed'] and all(checks.values()) else 'reference_routing_not_promoted',
        'repeat':repeat,'checks':checks,'quality_panel':{a:r['panel'] for a,r in data.items()},'identity_comparisons':{'gate1':full,'gate0':candidate,'gate0_vs_gate1':pair('gate1','gate0')},
        'gate0_mouth_comparison':mouth,'quality_deltas_vs_baseline':deltas,'runs':run_evidence,'pilots':pilots,'actual_new_gpu_hours':cost,
        'checkpoint_sha256':ledger['checkpoint_sha256'],'eval_manifest_sha256':ledger['eval_manifest_sha256'],'thresholds':ledger['thresholds'],
        'official_score':None,'holdout_used':False,'human_action_validation':False,'quality_protocol':quality_protocol[0],
        'limitations':['Fixed inference residual architecture, not trained gate or pure identity disentanglement.',
          'Null-reference condition may be out of distribution; mean-sharing can suppress necessary action.',
          'Small talking-head cohort/one seed, gray and mouth diagnostics are not natural-action truth.',
          'CurricularFace/GME/Story/human/full official MM2026 score remain missing.']}
    write_json(args.output,output);print('Structural audit',output['decision'],'actualGPUh',cost)


if __name__=='__main__': main()
