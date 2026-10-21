import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('panelreport',Path(__file__).resolve().parents[1]/'scripts/evaluation_panel_report.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_missing_scores_have_separate_denominator_not_zero():
    assert module.finite_values([.5,None,.7])=={'count':2,'mean':.6}
    assert module.finite_values([None])=={'count':0,'mean':None}
    for values in ([True],[float('nan')],['.5'],[float('inf')]):
        with pytest.raises(ValueError):module.finite_values(values)


def test_cur_missing_reference_and_all_missing_distinct_values():
    r={'records':[{'sample_id':'a','cur_score':None,'arc_score':None,'face_detection_coverage':0,
                   'cur_scoring_coverage':0,'arc_scoring_coverage':0}]}
    rows=module.report_records(r)
    assert rows[0][1]['cur_score'] is None
    r['records'][0]['cur_score']=0
    assert module.report_records(r)[0][1]['cur_score']==0


def test_score_hash_and_failed_run_rejected(tmp_path):
    from ipvg.common import write_json,sha256_file
    write_json(tmp_path/'scores.json',{'records':[]})
    write_json(tmp_path/'result.json',{'status':'completed','score_sha256':sha256_file(tmp_path/'scores.json'),
               'backend_elapsed_seconds':36,'gpus':1,'allocated_gpu_hours':.01})
    assert module.job(tmp_path)[0]['status']=='completed'
    write_json(tmp_path/'scores.json',{'records':['drift']})
    with pytest.raises(ValueError):module.job(tmp_path)


def comparison_fixture():
    rows=[];scores={};media={}
    for arm in ('baseline','candidate'):
        for i in range(2):
            sid=arm+'_'+str(i)
            rows.append({'sample_id':sid,'base_sample_id':str(i),'group':arm,'split':'dev',
                         'prompt':'A person moves.','reference_sha256':'same','identity_id':'actor'+str(i)})
            scores[sid]={k:.5 if arm=='baseline' else .7 for k in module.CORE_METRICS}
            media[sid]={'ok':True,'width':480,'height':352,'decoded_frames':49,'fps':16,'duration_s':49/16}
    signatures={k:{'protocol':'fixed','weights':'fixed'} for k in ('gme','cur','quality','vbench')}
    return rows,scores,media,signatures


def test_fixed_cohort_paired_metrics_and_missing_denominators():
    rows,scores,media,signatures=comparison_fixture()
    scores['candidate_1']['cur_score']=None
    report=module.paired_comparisons(rows,scores,media,signatures)
    assert report['comparable_evaluation_ready'] is True
    assert report['official_evaluation_complete'] is False
    core=report['comparisons']['main_same_input']['groups']['candidate']['core_metrics']
    cur=core['cur_score']
    assert cur['candidate']['coverage']==.5 and cur['paired_videos']==1 and cur['unpaired_videos']==1
    assert cur['paired_actor_mean_delta']==pytest.approx(.2)
    assert core['arc_score']['paired_videos']==2
    assert report['protocol']['combined_score'] is None
    assert 'local_story_similarity' not in report['protocol']['core_metrics']


@pytest.mark.parametrize('error',['prompt','reference','actor','samples','duplicates','width','frames','fps','duration','decode','metric','scoreset','mixed_cohort'])
def test_noncomparable_inputs_or_media_rejected(error):
    rows,scores,media,signatures=comparison_fixture()
    if error=='prompt':rows[2]['prompt']='Changed prompt'
    elif error=='reference':rows[2]['reference_sha256']='changed'
    elif error=='actor':rows[2]['identity_id']='changed'
    elif error=='samples':rows[2]['base_sample_id']='different'
    elif error=='duplicates':rows[2]['base_sample_id']=rows[3]['base_sample_id']
    elif error=='width':media['candidate_0']['width']=512
    elif error=='frames':media['candidate_0']['decoded_frames']=48
    elif error=='fps':media['candidate_0']['fps']=24
    elif error=='duration':media['candidate_0']['duration_s']=5
    elif error=='decode':media['candidate_0']['ok']=False
    elif error=='metric':scores['candidate_0'].pop('cur_score')
    elif error=='scoreset':scores.pop('candidate_0')
    elif error=='mixed_cohort':rows[2]['cohort']='historical_480square_not_main_comparison'
    with pytest.raises(ValueError):module.paired_comparisons(rows,scores,media,signatures)


def test_scorer_precision_and_weight_identity_are_part_of_signature():
    from copy import deepcopy
    report={'protocol':{'version':'native'},'model_file_sha256':{'weights':'hash'},
            'execution':{'device':'cuda','dtype':'float16','batch_size':1,'attention_implementation':'sdpa'}}
    signature=module.scorer_signature(report,'gme')
    changed=deepcopy(report);changed['execution']['dtype']='float32'
    assert signature!=module.scorer_signature(changed,'gme')
    changed=deepcopy(report);changed['model_file_sha256']['weights']='different'
    assert signature!=module.scorer_signature(changed,'gme')
    changed['model_file_sha256']={}
    with pytest.raises(ValueError):module.scorer_signature(changed,'gme')


def test_historical_groups_are_compared_only_in_their_own_cohort():
    rows,scores,media,signatures=comparison_fixture()
    for row in rows:
        row['cohort']='historical_480square_not_main_comparison'
        if row['group']=='baseline':row['group']='archive_vace_h1'
    for item in media.values():item['height']=480
    report=module.paired_comparisons(rows,scores,media,signatures)
    cohort=report['comparisons']['historical_480square_not_main_comparison']
    assert cohort['baseline_group']=='archive_vace_h1'
    assert cohort['cross_cohort_comparison_permitted'] is False
