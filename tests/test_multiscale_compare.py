import copy
import importlib.util
from pathlib import Path
import pytest
np=pytest.importorskip('numpy')
spec=importlib.util.spec_from_file_location('multiscale_compare',Path(__file__).resolve().parents[1]/'scripts/compare_anyid_multiscale.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def observation(score=1.,valid=16,actor='a'):
    data={'pair_scores':[score]*valid+[None]*(16-valid),'pair_coverage':valid/16}
    if valid>=12:data['mouth_residual_flow_mean']=score
    return {'identity_id':actor,'source':data}


def test_activity_uses_full_denominator_and_missing_not_zero():
    single={('a',42):observation(),('b',42):observation(valid=0)}
    dual={('a',42):observation(score=.2),('b',42):observation(valid=0)}
    result=module.motion_summary(single,dual,.5)
    assert result['single']['above_control_videos']==1
    assert result['single']['activity_video_rate_all']==.5
    assert result['dual']['above_control_videos']==0
    assert result['paired_videos']==1 and result['paired_flow_ratio']==pytest.approx(.2)


def test_low_coverage_remains_missing():
    a={('a',42):observation(valid=0)}
    result=module.motion_summary(a,copy.deepcopy(a),.5)
    assert result['paired_videos']==0 and result['paired_flow_ratio'] is None
    assert result['dual']['scoreable_videos']==0


@pytest.mark.parametrize('mutation',['keys','actor','length','nan','bool','coverage','mean','low_coverage_score'])
def test_malformed_reports_rejected(mutation):
    a={('a',42):observation()}
    b=copy.deepcopy(a)
    row=b[('a',42)]
    if mutation=='keys':b={('b',42):row}
    elif mutation=='actor':row['identity_id']='different'
    elif mutation=='length':row['source']['pair_scores'].pop()
    elif mutation=='nan':row['source']['pair_scores'][0]=np.nan
    elif mutation=='bool':row['source']['pair_scores'][0]=True
    elif mutation=='coverage':row['source']['pair_coverage']=0
    elif mutation=='mean':row['source']['mouth_residual_flow_mean']=2
    else:row['source'].update(pair_scores=[None]*16,pair_coverage=0)
    with pytest.raises(ValueError):module.motion_summary(a,b,.5)


@pytest.mark.parametrize('threshold',[None,True,-1.,np.nan,np.inf])
def test_bad_control_threshold_rejected(threshold):
    a={('a',42):observation()}
    with pytest.raises(ValueError):module.motion_summary(a,a,threshold)


def test_positive_identity_does_not_override_dynamic_failure():
    group={'coverage_delta':.1,'paired_actors':12,'identity_actor_bootstrap_ci95':[.01,.1],
           'identity_p10_actor_delta':.03,'motion_ratio':1.,'static_fraction_delta':0.,
           'single':{'videos_at_least_seven_frames':30},'dual':{'videos_at_least_seven_frames':31}}
    cohorts={name:copy.deepcopy(group) for name in ('all','seed42','seed43','additional_dev')}
    motion={'paired_flow_ratio':.5,'dual':{'mean_pair_coverage':1.,'activity_video_rate_all':.4},
            'single':{'activity_video_rate_all':.8}}
    result=module.quality_checks(cohorts,motion,'passed_local_activity_controls')
    assert result['identity_ci_lower_at_least_minus_0_02']
    assert not result['mouth_activity_rate_not_down_more_than_0_05']
    assert not result['paired_mouth_flow_ratio_at_least_0_8']
    assert not all(result.values())
