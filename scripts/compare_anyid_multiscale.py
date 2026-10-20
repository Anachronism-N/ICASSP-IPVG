"""Recompute dual-scale development gates from fixed identity and local-motion reports."""
import argparse
from pathlib import Path
import numpy as np
from ipvg.common import load_json, sha256_file, write_json
from ipvg.data import read_manifest


def motion_summary(single, dual, threshold):
    if not single or set(single)!=set(dual):
        raise ValueError('Motion sample/seed sets must match and be nonempty')
    if isinstance(threshold,bool) or not isinstance(threshold,(int,float)) or not np.isfinite(threshold) or threshold<0:
        raise ValueError('Require finite nonnegative frozen control threshold')
    for key in single:
        if single[key]['identity_id']!=dual[key]['identity_id']:
            raise ValueError('Motion actors differ')
    result={}
    for name,records in [('single',single),('dual',dual)]:
        for row in records.values():
            metrics=row['source']
            pairs=metrics['pair_scores']
            if len(pairs)!=16 or any(v is not None and (isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<0) for v in pairs):
                raise ValueError('Require sixteen finite nonnegative or missing motion pairs')
            valid=[v for v in pairs if v is not None]
            if metrics['pair_coverage']!=len(valid)/16:
                raise ValueError('Inconsistent motion coverage')
            if len(valid)>=12:
                score=metrics.get('mouth_residual_flow_mean')
                if isinstance(score,bool) or not isinstance(score,(int,float)) or not np.isfinite(score) or not np.isclose(score,np.mean(valid),rtol=0,atol=1e-7):
                    raise ValueError('Motion score differs from observed pairs')
            elif 'mouth_residual_flow_mean' in metrics:
                raise ValueError('Low-coverage motion score must remain missing')
        values=[r['source']['mouth_residual_flow_mean'] for r in records.values() if 'mouth_residual_flow_mean' in r['source']]
        active=sum(v>threshold for v in values)
        result[name]={'videos':len(records),'scoreable_videos':len(values),
                      'mean_pair_coverage':float(np.mean([r['source']['pair_coverage'] for r in records.values()])),
                      'above_control_videos':active,'activity_video_rate_all':active/len(records)}
    common=[(single[k]['source']['mouth_residual_flow_mean'],dual[k]['source']['mouth_residual_flow_mean']) for k in sorted(single)
            if 'mouth_residual_flow_mean' in single[k]['source'] and 'mouth_residual_flow_mean' in dual[k]['source']]
    result['paired_videos']=len(common)
    result['paired_flow_ratio']=float(np.mean([b for a,b in common])/max(np.mean([a for a,b in common]),1e-9)) if common else None
    return result


def quality_checks(cohorts,motion,calibration_status):
    group=cohorts['all']
    low=group['identity_actor_bootstrap_ci95'][0]
    p10=group['identity_p10_actor_delta']
    ratio=motion['paired_flow_ratio']
    return {'complete_candidate_media':True,
            'coverage_improves_at_least_0_02':group['coverage_delta']>=.02,
            'seven_frame_videos_not_down_by_seed_and_new_actors':all(cohorts[g]['dual']['videos_at_least_seven_frames']>=cohorts[g]['single']['videos_at_least_seven_frames'] for g in ('seed42','seed43','additional_dev')),
            'paired_actors_at_least_10':group['paired_actors']>=10,
            'identity_ci_lower_at_least_minus_0_02':low is not None and low>=-.02,
            'p10_not_degraded':p10 is not None and p10>=-.02,
            'gray_motion_protected':group['motion_ratio']>=.8 and group['static_fraction_delta']<=.10,
            'mouth_calibration_passed':calibration_status=='passed_local_activity_controls',
            'mouth_paircoverage_at_least_0_75':motion['dual']['mean_pair_coverage']>=.75,
            'mouth_activity_rate_not_down_more_than_0_05':motion['dual']['activity_video_rate_all']>=motion['single']['activity_video_rate_all']-.05,
            'paired_mouth_flow_ratio_at_least_0_8':ratio is not None and ratio>=.8}


def main():
    from compare_anyid_context import summarize_pairs
    from mouth_motion_eval import PROTOCOL as MOTION_PROTOCOL
    from ipvg.evaluate import evaluate_outputs
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('ledger','runs-dir','scores-dir','historical-scores-dir','calibration','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Choose a fresh aggregate output')
    ledger=load_json(args.ledger)
    calibration=load_json(args.calibration)
    if calibration['protocol']!=MOTION_PROTOCOL: raise ValueError('Local motion protocol differs')
    identity={'single':[],'dual':[]}
    motion={'single':{},'dual':{}}
    frozen=None
    hours=0
    for entry in ledger['runs']:
        folders={'single':args.runs_dir/entry['parent_run'],'dual':args.runs_dir/entry['run_id']}
        for arm,folder in folders.items():
            run=load_json(folder/'result.json')
            if run['status']!='completed': raise ValueError('Incomplete run')
            if sha256_file(folder/'manifest.jsonl')!=run['selected_manifest_sha256']: raise ValueError('Frozen manifest drift')
            media=evaluate_outputs(folder/'manifest.jsonl',folder/'outputs',expected_width=480,expected_height=480,
                                   expected_fps=16,expected_duration=49/16,fail_on_extra=True)
            if not media['ok'] or any(s['media']['decoded_frames']!=49 for s in media['samples']):
                raise ValueError('Media specification or full decode failed')
            rows=read_manifest(folder/'manifest.jsonl')
            if len(rows)!=entry['count'] or any(r.get('split')!='dev' for r in rows): raise ValueError('Unexpected input cohort')
            score_path=(args.historical_scores_dir/(entry['parent_name']+'.json')) if arm=='single' else args.scores_dir/(entry['name']+'_identity.json')
            report=load_json(score_path)
            current=(report['protocol'],report['weight_hashes'])
            if frozen is not None and current!=frozen: raise ValueError('Identity protocol or weights differ')
            frozen=current
            if len(report['records'])!=len(rows) or {r['sample_id'] for r in rows}!={r['sample_id'] for r in report['records']}:
                raise ValueError('Identity score inputs differ')
            for row in report['records']:
                name=row['sample_id']+'.mp4'
                if row['output_sha256']!=run['output_sha256'][name] or sha256_file(folder/'outputs'/name)!=row['output_sha256']:
                    raise ValueError('Output hash drift')
                identity[arm].append(dict(row,seed_offset=entry['seed']))
            local=load_json(args.scores_dir/(entry['name']+'_'+arm+'_motion.json'))
            if local['protocol']!=MOTION_PROTOCOL or local['weight_hashes']!=calibration['weight_hashes']:
                raise ValueError('Motion protocol or weights differ')
            if len(local['records'])!=len(rows): raise ValueError('Motion count differs')
            for row in local['records']:
                key=(row['sample_id'],entry['seed'])
                if key in motion[arm] or row['video_sha256']!=run['output_sha256'][row['sample_id']+'.mp4']:
                    raise ValueError('Motion duplicate or video mismatch')
                motion[arm][key]=row
            if arm=='dual':hours+=run['estimated_backend_gpu_hours']
    known=ledger['cohorts']['legacy_actors']
    new=ledger['cohorts']['additional_dev_actors']
    if len(set(known+new))!=12 or len(known)!=4 or len(new)!=8 or any(len(r)!=48 for r in identity.values()) or any(len(r)!=48 for r in motion.values()):
        raise ValueError('Expected48 paired inputs and12 frozen actors')
    if any({r['identity_id'] for r in rows}!=set(known+new) for rows in identity.values()):
        raise ValueError('Frozen actor set differs')
    if hours>ledger['max_gpu_hours']: raise ValueError('GPU budget exceeded')
    groups={name:summarize_pairs(identity['single'],identity['dual'],actors,seed) for name,actors,seed in
            [('all',known+new,None),('legacy',known,None),('additional_dev',new,None),('seed42',known+new,42),('seed43',known+new,43)]}
    for group in groups.values():
        group['single']=group.pop('narrow')
        group['dual']=group.pop('wide')
    local=motion_summary(motion['single'],motion['dual'],calibration['threshold'])
    checks=quality_checks(groups,local,calibration['status'])
    write_json(args.output,{'decision':'local_dualscale_candidate_only' if all(checks.values()) else 'dualscale_not_promoted',
                            'cohorts':groups,'motion':local,'checks':checks,'actual_candidate_gpu_hours':hours,
                            'official_score':None,'speech_or_expression_validation_passed':False,'holdout_used':False})
    print('Recomputed48 paired inputs with identity, coverage and local motion gates')


if __name__=='__main__':
    main()
