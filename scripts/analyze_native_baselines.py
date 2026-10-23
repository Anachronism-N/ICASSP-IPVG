"""Audit frozen native identity-generation baselines; no composite ranking or SOTA claim."""
import argparse
import math
import hashlib
from pathlib import Path
from ipvg.common import load_json,write_json,sha256_file
from ipvg.data import read_manifest


def generation_evidence(folder,entry):
    from ipvg.evaluate import evaluate_outputs
    result=load_json(folder/'result.json');config=load_json(folder/'config.json');profile=load_json(folder/'backend_metrics.json')
    if result['status']!='completed' or result['run_id']!=entry['run_id'] or result['source_sha256']!=entry['source_sha256']:
        raise ValueError('Native generation incomplete/source drift')
    if sha256_file(folder/'config.json')!=entry['config_sha256'] or sha256_file(folder/'manifest.jsonl')!=entry['manifest_sha256']:
        raise ValueError('Native frozen config or selected data changed')
    media=evaluate_outputs(folder/'manifest.jsonl',folder/'outputs',expected_width=480,expected_height=352,expected_fps=16,
                           expected_duration=49/16,fail_on_extra=True)
    if not media['ok'] or len(media['samples'])!=entry['count'] or any(r['media']['decoded_frames']!=49 for r in media['samples']):
        raise ValueError('Native media does not meet shared output specification')
    for name,digest in result['output_sha256'].items():
        if sha256_file(folder/'outputs'/name)!=digest:raise ValueError('Generation media SHA changed')
    observations={r['sample_id']:r for r in profile['samples']}
    rows=read_manifest(folder/'manifest.jsonl')
    if len(observations)!=entry['count'] or len(profile['samples'])!=entry['count'] or set(observations)!={r['sample_id'] for r in rows}:
        raise ValueError('Native observation count differs')
    if entry['arm']=='standin':
        if profile['matched_ip_tensors']!=240 or profile['standin_checkpoint_sha256']!='52f47630b4fefbdac841d3fd1ce146cca2a6a94ec0a6ab9407d16f2f635177e9':
            raise ValueError('Stand-In full native identity branch not loaded')
        for row in rows:
            s=observations[row['sample_id']]
            if (s['native_denoiser_calls']!=40 or len(s['ip_latent_calls'])!=1
                    or s['ip_latent_calls'][0]['ip_latent_shape']!=[1,16,1,64,64]
                    or s['ip_q_lora_calls']!=[[1,1024,5120]]
                    or sha256_file(row['reference_images'][0])!=s['reference_original_sha256']):
                raise ValueError('Stand-In native IP token/caching/control evidence differs')
    if entry['arm']=='anyid':
        for s in observations.values():
            if (s['reference_count']!=1 or s['inner_t']!=[1,13] or s['native_frame_num']!=53
                    or len(s['denoiser_calls'])!=40):raise ValueError('AnyID native conditioning not verified')
    if entry['arm']=='vace' and any(s['vace_forward_calls']!=40 for s in observations.values()):
        raise ValueError('VACE reference forward count differs')
    hours=result['resources']['gpus']*result['backend_elapsed_seconds']/3600
    if not math.isfinite(hours) or hours<0 or hours>entry['reserved_gpu_hours'] or not math.isclose(hours,result['estimated_backend_gpu_hours'],abs_tol=1e-10):
        raise ValueError('Native generation allocation cost inconsistent')
    return result,config,profile,hours


def identity_coverage_summary(records):
    from evaluation_panel_report import finite_values
    result={}
    for group in sorted({r['group'] for r in records}):
        rows=[r for r in records if r['group']==group]
        for r in rows:
            if len(r['cur_frame_cosines'])!=16 or len(r['arc_frame_cosines'])!=16:
                raise ValueError('Identity frame observations must contain sixteen entries')
            for name in ('cur','arc'):
                valid=[v for v in r[name+'_frame_cosines'] if v is not None]
                finite_values(valid)
                if len(valid)/16!=r[name+'_scoring_coverage']:raise ValueError('Identity coverage inconsistent')
                if valid and not math.isclose(math.fsum(valid)/len(valid),r[name+'_score'],abs_tol=1e-7):
                    raise ValueError('Identity video mean differs from observed frames')
        valid=[r for r in rows if r['cur_scoring_coverage']>0 and r['arc_scoring_coverage']>0]
        result[group]={'videos':len(rows),'detected_identity_videos':len(valid),
                       'all_missing_identity_videos':len(rows)-len(valid),
                       'mean_face_frame_coverage':math.fsum(r['face_detection_coverage'] for r in rows)/len(rows),
                       'valid_video_cur_mean_diagnostic_only':finite_values([r['cur_score'] for r in valid])['mean'],
                       'valid_video_arc_mean_diagnostic_only':finite_values([r['arc_score'] for r in valid])['mean'],
                       'notice':'Conditional means have method-specific denominators; never use them alone to select a winner.'}
    return result


def main():
    from evaluation_panel_report import paired_comparisons,scorer_signature,report_records,finite_values
    from ipvg.evaluate import probe_video
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('work-dir','runs-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--evaluation-lock',type=Path,default=Path(__file__).resolve().parents[1]/'resources/evaluation_panel.lock.json')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Fresh native audit path required')
    ledger=load_json(args.work_dir/'ledger.json'); panel=load_json(args.work_dir/'quality_frozen.json')['samples']
    if len(panel)!=48 or len({r['sample_id'] for r in panel})!=48:raise ValueError('Expected48complete newmedia')
    if len(ledger['runs'])!=6 or {(r['arm'],r['phase']) for r in ledger['runs']}!={(a,p) for a in ('vace','anyid','standin') for p in ('pilot','main')}:
        raise ValueError('Native campaign must contain the six frozen generation runs')
    ids={r['sample_id']:r for r in panel};scores={sid:{} for sid in ids};signatures={};hours=0;scoring=[];identity_records=[]
    evaluation_lock=load_json(args.evaluation_lock)
    for metric in ('cur','gme','quality','extended'):
        seen=set()
        for shard in (0,1):
            folder=args.work_dir/'evaluation'/f'{metric}_shard{shard}_r001'
            run=load_json(folder/'result.json');report=load_json(folder/'scores.json')
            if (run['status']!='completed' or sha256_file(folder/'scores.json')!=run['score_sha256']
                    or report['manifest_sha256']!=run['manifest_sha256']
                    or run['manifest_sha256']!=sha256_file(args.work_dir/f'quality_shard{shard}.json')):
                raise ValueError('Native scoring incomplete or report hash mismatch')
            label='vbench' if metric=='extended' else metric
            signature=scorer_signature(report,label)
            if metric=='gme' and report['model_file_sha256']!=evaluation_lock['gme']['files']:
                raise ValueError('GME weights differ from frozen evaluation lock')
            if metric=='cur' and report['model_source']['files']!=evaluation_lock['cur']['files']:
                raise ValueError('Identity weights differ from frozen evaluation lock')
            if metric=='extended' and report['weight_hashes']!=evaluation_lock['vbench_extended']['files']:
                raise ValueError('Auxiliary weights differ from frozen evaluation lock')
            if label in signatures and signatures[label]!=signature:raise ValueError('Evaluator protocol/weights/precision differ across shards')
            signatures[label]=signature
            h=run['backend_elapsed_seconds']/3600
            if not math.isfinite(h) or h<0 or run['gpus']!=1 or not math.isclose(h,run['allocated_gpu_hours'],abs_tol=1e-10) or h>run['reserved_gpu_hours']:
                raise ValueError('Native scoring cost not aligned')
            hours+=h;scoring.append({'run_id':run['run_id'],'status':run['status'],'gpu_hours':h,
                                    'score_sha256':run['score_sha256'],'evaluator_sha256':run['evaluator_sha256']})
            for row,values in report_records(report):
                sid=row['sample_id']
                if sid in seen or sid not in ids or row['video_sha256']!=ids[sid]['video_sha256']:
                    raise ValueError('Scoring inputs duplicate or changed')
                if row['group']!=ids[sid]['group']:raise ValueError('Scoring group differs')
                if metric=='cur' and (row['reference_sha256']!=ids[sid]['reference_sha256'] or row['identity_id']!=ids[sid]['identity_id']):
                    raise ValueError('Native evaluator anchor or actor changed')
                if metric in ('gme','extended') and row['prompt_sha256']!=hashlib.sha256(ids[sid]['prompt'].encode()).hexdigest():
                    raise ValueError('Native evaluator prompt changed')
                for value in values.values():finite_values([value])
                if metric=='cur':identity_records.append(row)
                seen.add(sid);scores[sid].update(values)
        if seen!=set(ids):raise ValueError('Native scorer does not cover48inputs')
    media={sid:probe_video(r['video_path']) for sid,r in ids.items()}
    comparison=paired_comparisons(panel,scores,media,signatures)
    generation=[];profiles={};configs={}
    for entry in ledger['runs']:
        result,config,profile,h=generation_evidence(args.runs_dir/entry['run_id'],entry)
        hours+=h
        generation.append({'run_id':entry['run_id'],'arm':entry['arm'],'phase':entry['phase'],'status':result['status'],
            'code_commit':result['code_commit'],'source_sha256':result['source_sha256'],'manifest_sha256':result['manifest_sha256'],
            'gpu_hours':h,'backend_elapsed_seconds':result['backend_elapsed_seconds'],
            'generation_median_seconds':__import__('statistics').median(r['generation_seconds'] for r in profile['samples']),
            'peak_allocated_vram_gb':max(r['peak_allocated_vram_gb'] for r in profile['samples']),
            'peak_reserved_vram_gb':max(r['peak_reserved_vram_gb'] for r in profile['samples'])})
        if entry['phase']=='main':profiles[entry['arm']]={r['sample_id']:r for r in profile['samples']};configs[entry['arm']]=config
    from adapters.standin import NEGATIVE
    if configs['standin']['parameters'].get('negative_prompt') is None:
        configs['standin']['parameters']['negative_prompt']=NEGATIVE
    main_entries={e['arm']:e for e in ledger['runs'] if e['phase']=='main'}
    main_rows=read_manifest(args.runs_dir/main_entries['vace']['run_id']/'manifest.jsonl')
    for row in main_rows:
        sid=row['sample_id']
        expected_seed=(42+int(hashlib.sha256(sid.encode()).hexdigest()[:8],16))%2**31
        if {profiles[a][sid]['seed'] for a in profiles}!={expected_seed}:raise ValueError('Original per-sample seeds differ')
        for arm in ('vace','anyid','standin'):
            folder=args.runs_dir/main_entries[arm]['run_id']
            original=next(r for r in read_manifest(folder/'manifest.jsonl') if r['sample_id']==sid)
            if original['prompt']!=row['prompt'] or original['reference_images']!=row['reference_images'] or original['identity_id']!=row['identity_id']:
                raise ValueError('Original generation prompt/reference/actor differs')
    if hours>ledger['new_hard_cap_gpu_hours']:raise ValueError('Native baseline hard budget exceeded')
    action_categories={r['sample_id']:r.get('action_category') for r in main_rows}
    stratified={}
    if any(action_categories.values()):
        for action in sorted(set(action_categories.values())):
            if action is None:raise ValueError('Action category missing from new cohort')
            stratified[action]={}
            for group in ('baseline','anyid','standin'):
                subset=[scores[sid] for sid,r in ids.items() if r['group']==group and action_categories[r['base_sample_id']]==action]
                if len(subset)!=4:raise ValueError('Each action category must have four inputs per model')
                stratified[action][group]={'videos':len(subset),'scores':{k:dict(finite_values([r.get(k) for r in subset]),coverage=sum(r.get(k)is not None for r in subset)/len(subset)) for k in sorted({k for r in subset for k in r})}}
    groups={}
    for group in ('baseline','anyid','standin'):
        subset=[scores[sid] for sid,r in ids.items() if r['group']==group]
        metrics=sorted({k for r in subset for k in r})
        groups[group]={'videos':len(subset),'scores':{k:dict(finite_values([r.get(k) for r in subset]),coverage=sum(r.get(k)is not None for r in subset)/len(subset)) for k in metrics}}
    output={'status':'completed_native_baseline_comparison','decision':'report_native_system_tradeoffs_no_automatic_winner',
       'comparison':comparison,'groups':groups,'action_stratified_groups':stratified,'cohort':ledger.get('cohort','original_talking_head_development'),
       'identity_coverage_diagnostics':identity_coverage_summary(identity_records),
       'generation_runs':generation,'scoring_runs':scoring,
       'actual_new_gpu_hours':hours,'new_hard_cap_gpu_hours':ledger['new_hard_cap_gpu_hours'],'original_overall_hard_cap_gpu_hours':ledger['original_overall_cap_gpu_hours'],
       'new_generated_videos':sum(e['count'] for e in ledger['runs']),'main_comparison_videos':len(panel),'training_performed':False,'new_architecture_changes':False,
       'official_final_score':None,'action_naturalness_verified':False,'retry_count':0,
       'native_system_settings':{a:{k:c['parameters'].get(k) for k in ('num_inference_steps','cfg_scale','sample_shift','negative_prompt','t5_cpu')} for a,c in configs.items()},
       'limitations':['Same original prompt/reference/seed/output specification and frozen scores; native preprocess/sampler/shift/negative prompts differ.',
          'Different model architectures do not share identical noise from a numeric seed; not a single-factor weight ablation.',
          'Small eight-actor development cohort/one seed with talking-head-source references; no broad action or SOTA claim.',
          'Cur/Arc largest-face policy can differ from strictsingle-face protocol; retain detection and score coverage.',
          'No combined metric, automatic champion or exact official score; human/natural-action validation remains missing.']}
    write_json(args.output,output);print('Native baseline audit complete,51videos core5+aux;GPUh',hours)


if __name__=='__main__':main()
