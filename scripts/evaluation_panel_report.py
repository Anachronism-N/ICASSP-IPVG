"""Audit a frozen evaluation panel; report missing official/semantic components explicitly."""
import argparse
from collections import defaultdict
from pathlib import Path
import math
from ipvg.common import load_json, write_json, sha256_file

METRICS=('gme','cur','story','vbench','quality')


def finite_values(values):
    values=[v for v in values if v is not None]
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in values):
        raise ValueError('Scores must be finite numeric or explicit missing')
    return {'count':len(values),'mean':math.fsum(values)/len(values) if values else None}


def report_records(report):
    result=[]
    for r in report['records']:
        if 'scores' in r:
            scores=dict(r['scores'])
        else:
            scores={k:r.get(k) for k in ('cur_score','arc_score','face_detection_coverage','cur_scoring_coverage','arc_scoring_coverage')}
        result.append((r,scores))
    return result


def job(folder):
    result=load_json(folder/'result.json')
    if result['status']!='completed':raise ValueError('Incomplete evaluation job: '+folder.name)
    score_path=folder/'scores.json'
    if sha256_file(score_path)!=result['score_sha256']:raise ValueError('Score hash changed')
    report=load_json(score_path)
    hours=result['backend_elapsed_seconds']*result['gpus']/3600
    if not math.isfinite(hours) or hours<0 or not math.isclose(hours,result['allocated_gpu_hours'],abs_tol=1e-10):
        raise ValueError('Job cost inconsistent')
    return result,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('panel','controls','jobs-root','output'):p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Fresh panel report required')
    panel=load_json(args.panel)['samples']; controls=load_json(args.controls)['samples']
    ids={r['sample_id']:r for r in panel}
    if len(ids)!=len(panel) or len(panel)!=120:raise ValueError('Expected120unique frozen panel inputs')
    metrics_by_id=defaultdict(dict); protocols={}; all_job_summaries=[]; hours=0
    for metric in METRICS:
        seen=set()
        for shard in (0,1):
            folder=args.jobs_root/f'panel_{metric}_shard{shard}_r001'
            result,report=job(folder);hours+=result['allocated_gpu_hours']
            if metric in protocols and protocols[metric]!=report['protocol']:raise ValueError('Metric protocol differs between shards')
            protocols[metric]=report['protocol']
            all_job_summaries.append({k:result[k] for k in ('run_id','status','manifest_sha256','score_sha256','allocated_gpu_hours')})
            for row,scores in report_records(report):
                sid=row['sample_id']
                if sid in seen or sid not in ids:raise ValueError('Missing/duplicate/unknown metric record')
                if row['video_sha256']!=ids[sid]['video_sha256'] or sha256_file(ids[sid]['video_path'])!=row['video_sha256']:
                    raise ValueError('Panel media/scorer hashes differ')
                seen.add(sid)
                for name,value in scores.items():
                    finite_values([value]);metrics_by_id[sid][name]=value
        if seen!=set(ids):raise ValueError('Metric input coverage incomplete: '+metric)
    groups=defaultdict(list)
    for sid,row in ids.items():groups[row['group']].append(metrics_by_id[sid])
    group_scores={}
    for group,rows in groups.items():
        names=sorted({k for r in rows for k in r})
        group_scores[group]={'videos':len(rows),'scores':{n:dict(finite_values([r.get(n) for r in rows]),coverage=sum(r.get(n) is not None for r in rows)/len(rows)) for n in names}}
    control_scores={}
    for metric in ('gme','cur','story','vbench','quality'):
        revision='r002' if metric=='gme' else 'r001'
        result,report=job(args.jobs_root/f'controls_{metric}_shard0_{revision}')
        hours+=result['allocated_gpu_hours']
        control_scores[metric]={r['sample_id']:s for r,s in report_records(report)}
    calibration_result,calibration=job(args.jobs_root/'source_cur-calibration_shard0_r001')
    hours+=calibration_result['allocated_gpu_hours']
    # Retain the first failed integration attempt in the allocation audit.
    failed=args.jobs_root/'controls_gme_shard0_r001/result.json'
    failed_record=load_json(failed);hours+=failed_record['allocated_gpu_hours']
    c=control_scores
    semantic={'gme_color_object_contrast':c['gme']['ball_source']['gme_aesr']-c['gme']['ball_wrong_prompt']['gme_aesr'],
              'gme_reverse_delta':c['gme']['ball_reverse']['gme_aesr']-c['gme']['ball_source']['gme_aesr'],
              'story_color_object_contrast':c['story']['ball_source']['local_story_similarity']-c['story']['ball_wrong_prompt']['local_story_similarity'],
              'story_motion_vs_freeze_delta':c['story']['ball_source']['local_story_similarity']-c['story']['ball_freeze']['local_story_similarity'],
              'story_original_vs_reverse_delta':c['story']['ball_source']['local_story_similarity']-c['story']['ball_reverse']['local_story_similarity'],
              'viclip_original_vs_reverse_delta':c['vbench']['ball_source']['overall_consistency']-c['vbench']['ball_reverse']['overall_consistency']}
    checks={'cur_source_calibration_passed':calibration['status']=='passed_source_identity_controls',
            'gme_content_contrast_positive':semantic['gme_color_object_contrast']>0,
            'raft_detects_known_motion':c['vbench']['ball_source']['dynamic_degree']==1,
            'raft_rejects_frozen_motion':c['vbench']['ball_freeze']['dynamic_degree']==0,
            'all120_metric_records_complete':True}
    action_checks={'local_story_distinguishes_motion_from_freeze':semantic['story_motion_vs_freeze_delta']>0,
                   'local_story_distinguishes_correct_order':semantic['story_original_vs_reverse_delta']>0,
                   'viclip_distinguishes_correct_order':semantic['viclip_original_vs_reverse_delta']>0}
    if hours>12:raise ValueError('Evaluation hard budget exceeded')
    output={'status':'completed_local_evaluation_panel','groups':group_scores,'protocols':protocols,'jobs':all_job_summaries,
            'source_identity_calibration':{k:calibration[k] for k in ('status','calibration','validation','checks','self_similarity_max_error','repeat_embedding_max_abs_error')},
            'control_scores':{metric:{sid:score for sid,score in rows.items() if sid.startswith('ball_')} for metric,rows in control_scores.items()},
            'semantic_control_deltas':semantic,'local_engineering_checks':checks,'action_semantic_control_checks':action_checks,
            'local_automatic_panel_passed':all(checks.values()),'official_evaluation_complete':False,
            'official_story_score':None,'official_human_score':None,'official_final_score':None,
            'stable_task_action_evaluator_verified':all(action_checks.values()),
            'actual_allocated_gpu_hours_including_failed_pilot':hours,
            'max_new_gpu_hours':12,'panel_manifest_sha256':sha256_file(args.panel),'effective_controls_sha256':sha256_file(args.controls),
            'cohorts':{'main_same_input':96,'archival_other_specification':24},
            'limitations':['96 same-specification videos and24 archival videos are stratified, never pooled for causal claims.',
              'Cur16 author-style zero for all-missing video is separate from strict ArcFace8 missing-score protocol.',
              'Frame-averaged GME is order-insensitive; local Story substitute and ViCLIP require control-based interpretation.',
              'High consistency or smoothness can reward static video; RAFT degree is not action correctness.',
              'No Gemini API access/external video upload; local captioner is not official or original StoryScore.',
              'No human scores manufactured, no official ranking score or gap-to-champion derived.',
              'The evaluation-first gate precedes stronger baselines and method optimization; incomplete task semantics remain a blocker.']}
    write_json(args.output,output)
    print('Audited120existing-video localpanel; automatic checks',all(checks.values()),'actioncontrols',all(action_checks.values()),'GPUh',hours)


if __name__=='__main__':main()
