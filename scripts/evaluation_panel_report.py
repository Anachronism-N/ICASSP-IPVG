"""Audit a frozen evaluation panel; report missing official/semantic components explicitly."""
import argparse
from collections import defaultdict
from pathlib import Path
import math
import hashlib
from ipvg.common import load_json, write_json, sha256_file, fingerprint

METRICS=('gme','cur','story','vbench','quality')
CORE_METRICS=('cur_score','arc_score','gme_aesr','motion_smoothness','imaging_quality')
COMPARISON_PROTOCOL={
    'version':'ipvg_cross_model_comparison_v1',
    'core_metrics':list(CORE_METRICS),
    'auxiliary_metrics':['face_detection_coverage','cur_scoring_coverage','arc_scoring_coverage',
                         'dynamic_degree','aesthetic_quality','subject_consistency','background_consistency'],
    'diagnostic_only':['local_story_similarity','overall_consistency'],
    'paired_inputs':'Same base_sample_id, original prompt/reference/actor, decoded output dimensions/frames/fps/duration within cohort',
    'aggregation':'Video-equal means plus paired deltas; paired actor-equal delta and explicit missing denominators',
    'combined_score':None,'official_final_score':None,'action_naturalness_verified':False,
    'readiness':'Audited comparable automatic metrics permit baseline benchmarking, not official ranking or natural-action claims',
}


def scorer_signature(report, metric):
    protocol=report.get('protocol')
    if not isinstance(protocol,dict) or not protocol.get('version'):
        raise ValueError('Missing scoring protocol/version')
    if metric=='gme':
        weights=report.get('model_file_sha256')
        execution=report.get('execution',{})
        precision={k:execution.get(k) for k in ('device','dtype','batch_size','attention_implementation')}
        if any(v is None for v in precision.values()):raise ValueError('Missing GME execution precision')
    elif metric=='cur':
        source=report.get('model_source',{})
        weights={'files':source.get('files'),'source_files':source.get('source_files')}
        precision={'device':report.get('device'),'torch_dtype':'float32','detector':protocol.get('detector_provider')}
        if not weights['files'] or not weights['source_files'] or not precision['device']:
            raise ValueError('Missing Cur weights/source/device')
    elif metric in ('quality','vbench'):
        weights=report.get('weight_hashes')
        precision=report.get('actual_parameter_dtypes',protocol.get('precision'))
        if not precision:raise ValueError('Missing quality execution precision')
    else:
        weights=report.get('model_lock_sha256')
        precision=protocol
    if not weights:raise ValueError('Missing scoring model identity')
    return {'protocol':protocol,'weights':weights,'precision':precision}


def paired_comparisons(panel, scores, media, signatures):
    if not panel or set(scores)!=set(media) or set(scores)!={r['sample_id'] for r in panel}:
        raise ValueError('Complete matching panel/score/media cohorts required')
    if not {'gme','cur','quality','vbench'}.issubset(signatures):
        raise ValueError('Core/auxiliary scorer signatures incomplete')
    cohorts=defaultdict(lambda:defaultdict(dict))
    seen=set()
    for row in panel:
        sid=row['sample_id']; group=row['group']; base_id=row['base_sample_id']
        if sid in seen or row.get('split')!='dev':raise ValueError('Duplicate or non-dev comparison input')
        seen.add(sid)
        cohort=row.get('cohort','main_same_input')
        if cohort not in ('main_same_input','historical_480square_not_main_comparison'):
            raise ValueError('Unknown comparison cohort')
        if base_id in cohorts[cohort][group]:raise ValueError('Duplicate group/base_sample_id')
        spec=media[sid]
        expected=(480,352) if cohort=='main_same_input' else (480,480)
        if (not spec.get('ok') or (spec.get('width'),spec.get('height'))!=expected
                or spec.get('decoded_frames')!=49 or not math.isclose(spec.get('fps',0),16,abs_tol=.01)
                or not math.isclose(spec.get('duration_s',0),49/16,abs_tol=.01)):
            raise ValueError('Comparison output specification mismatch')
        if any(k not in scores[sid] for k in CORE_METRICS):raise ValueError('Core score columns incomplete')
        for k in CORE_METRICS:finite_values([scores[sid][k]])
        cohorts[cohort][group][base_id]=row
    baseline_groups={'main_same_input':'baseline','historical_480square_not_main_comparison':'archive_vace_h1'}
    result={}
    for cohort,groups in cohorts.items():
        baseline=baseline_groups[cohort]
        if baseline not in groups:raise ValueError('Declared comparison baseline missing')
        left=groups[baseline]; comparisons={}
        for group,right in groups.items():
            if set(left)!=set(right):raise ValueError('Paired sample sets differ')
            for bid in left:
                a,b=left[bid],right[bid]
                if (a['prompt']!=b['prompt'] or a['reference_sha256']!=b['reference_sha256']
                        or a['identity_id']!=b['identity_id']):
                    raise ValueError('Paired prompt/reference/actor differs')
            stats={}
            for metric in CORE_METRICS:
                deltas=[]; by_actor=defaultdict(list)
                av=[];bv=[]
                for bid in sorted(left):
                    a,b=left[bid],right[bid]
                    x,y=scores[a['sample_id']][metric],scores[b['sample_id']][metric]
                    av.append(x);bv.append(y)
                    if x is not None and y is not None:
                        deltas.append(y-x);by_actor[a['identity_id']].append(y-x)
                stats[metric]={'baseline':dict(finite_values(av),coverage=sum(v is not None for v in av)/len(av)),
                               'candidate':dict(finite_values(bv),coverage=sum(v is not None for v in bv)/len(bv)),
                               'paired_videos':len(deltas),'unpaired_videos':len(left)-len(deltas),
                               'paired_video_mean_delta':finite_values(deltas)['mean'],
                               'paired_actors':len(by_actor),
                               'paired_actor_mean_delta':finite_values([math.fsum(v)/len(v) for v in by_actor.values()])['mean']}
            comparisons[group]={'input_videos':len(left),'core_metrics':stats}
        result[cohort]={'baseline_group':baseline,'groups':comparisons,'cross_cohort_comparison_permitted':False}
    return {'protocol':COMPARISON_PROTOCOL,'scorer_signatures_sha256':fingerprint(signatures),
            'comparable_evaluation_ready':True,'comparisons':result,
            'official_evaluation_complete':False,'action_naturalness_verified':False,
            'automatic_winner_selected':None}


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
    metrics_by_id=defaultdict(dict); protocols={}; signatures={}; all_job_summaries=[]; hours=0
    for metric in METRICS:
        seen=set()
        for shard in (0,1):
            revision = 'r002' if shard == 1 and metric in ('vbench','quality') else 'r001'
            folder=args.jobs_root/f'panel_{metric}_shard{shard}_{revision}'
            result,report=job(folder);hours+=result['allocated_gpu_hours']
            if report['manifest_sha256'] != result['manifest_sha256']:
                raise ValueError('Scorer and controller input manifest hashes differ')
            if metric in protocols and protocols[metric]!=report['protocol']:raise ValueError('Metric protocol differs between shards')
            protocols[metric]=report['protocol']
            signature=scorer_signature(report,metric)
            if metric in signatures and signatures[metric]!=signature:
                raise ValueError('Scoring weights/source or execution precision differs between shards')
            signatures[metric]=signature
            all_job_summaries.append({k:result[k] for k in ('run_id','status','manifest_sha256','score_sha256','allocated_gpu_hours')})
            for row,scores in report_records(report):
                sid=row['sample_id']
                if sid in seen or sid not in ids:raise ValueError('Missing/duplicate/unknown metric record')
                if row['video_sha256']!=ids[sid]['video_sha256'] or sha256_file(ids[sid]['video_path'])!=row['video_sha256']:
                    raise ValueError('Panel media/scorer hashes differ')
                if metric in ('gme','story','vbench') and row['prompt_sha256'] != hashlib.sha256(ids[sid]['prompt'].encode()).hexdigest():
                    raise ValueError('Exact scoring prompt differs from frozen panel')
                if metric == 'cur' and row['reference_sha256'] != ids[sid]['reference_sha256']:
                    raise ValueError('Scoring reference differs from frozen panel')
                seen.add(sid)
                for name,value in scores.items():
                    finite_values([value]);metrics_by_id[sid][name]=value
        if seen!=set(ids):raise ValueError('Metric input coverage incomplete: '+metric)
    from ipvg.evaluate import probe_video
    media={sid:probe_video(row['video_path']) for sid,row in ids.items()}
    comparison=paired_comparisons(panel,metrics_by_id,media,signatures)
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
    excluded_jobs = []
    for name in ('panel_vbench_shard1_r001','panel_quality_shard1_r001'):
        record = load_json(args.jobs_root/name/'result.json')
        hours += record['allocated_gpu_hours']
        excluded_jobs.append({'run_id':name,'status':record['status'],'allocated_gpu_hours':record['allocated_gpu_hours'],
                              'reason':'Old worker upstream had pruned tracked unused model directories; rerun on clean pinned snapshot, no score formula changes.'})
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
            'cross_model_comparison':comparison,'comparable_evaluation_ready':comparison['comparable_evaluation_ready'],
            'source_identity_calibration':{k:calibration[k] for k in ('status','calibration','validation','checks','self_similarity_max_error','repeat_embedding_max_abs_error')},
            'control_scores':{metric:{sid:score for sid,score in rows.items() if sid.startswith('ball_')} for metric,rows in control_scores.items()},
            'semantic_control_deltas':semantic,'local_engineering_checks':checks,'action_semantic_control_checks':action_checks,
            'local_automatic_panel_passed':all(checks.values()),'official_evaluation_complete':False,
            'official_story_score':None,'official_human_score':None,'official_final_score':None,
            'stable_task_action_evaluator_verified':all(action_checks.values()),
            'actual_allocated_gpu_hours_including_failed_pilot':hours,
            'excluded_previous_jobs':excluded_jobs,'max_new_gpu_hours':12,'panel_manifest_sha256':sha256_file(args.panel),'effective_controls_sha256':sha256_file(args.controls),
            'cohorts':{'main_same_input':96,'archival_other_specification':24},
            'limitations':['96 same-specification videos and24 archival videos are stratified, never pooled for causal claims.',
              'Cur16 author-style zero for all-missing video is separate from strict ArcFace8 missing-score protocol.',
              'Frame-averaged GME is order-insensitive; local Story substitute and ViCLIP require control-based interpretation.',
              'High consistency or smoothness can reward static video; RAFT degree is not action correctness.',
              'No Gemini API access/external video upload; local captioner is not official or original StoryScore.',
              'No human scores manufactured, no official ranking score or gap-to-champion derived.',
              'Fixed comparable automatic protocol permits baseline benchmarking; unvalidated task semantics block natural-action/official claims, not all model comparisons.']}
    write_json(args.output,output)
    print('Audited120existing-video localpanel; automatic checks',all(checks.values()),'actioncontrols',all(action_checks.values()),'GPUh',hours)


if __name__=='__main__':main()
