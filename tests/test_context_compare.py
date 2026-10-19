import copy
import importlib.util
from pathlib import Path
import pytest
np=pytest.importorskip('numpy')
spec=importlib.util.spec_from_file_location('context_compare',Path(__file__).resolve().parents[1]/'scripts/compare_anyid_context.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(sid='a',actor='actor1',values=None,seed=42):
    values=[.5]*8 if values is None else values
    return {'sample_id':sid,'identity_id':actor,'seed_offset':seed,'frame_cosines':values,
            'scores':{'identity_scoring_coverage':sum(v is not None for v in values)/8,'sampled_gray_motion_mean_abs':1.}}


def test_missing_never_imputed_and_actor_equal_weight():
    left=[row('a1'),row('a2'),row('b','actor2'),row('missing','actor3',[None]*8)]
    right=[row('a1',values=[.7]*8),row('a2',values=[.7]*8),row('b','actor2',[.1]*8),row('missing','actor3',[.9]*8)]
    result=module.summarize_pairs(left,right,['actor1','actor2','actor3'])
    assert result['narrow']['scoreable_videos']==3 and result['wide']['scoreable_videos']==4
    assert result['coverage_delta']==.25 and result['paired_videos']==3 and result['paired_actors']==2
    assert result['recovered_seven_frame_videos']==1 and result['lost_seven_frame_videos']==0
    assert result['identity_actor_mean_delta']==pytest.approx(-.1)


@pytest.mark.parametrize('mutation',['duplicate','actor','coverage','nan','length','bool','seed'])
def test_malformed_pairs_fail_closed(mutation):
    a,b=[row()],[row()]
    if mutation=='duplicate': b.append(copy.deepcopy(b[0]))
    elif mutation=='actor': b[0]['identity_id']='wrong'
    elif mutation=='coverage': b[0]['scores']['identity_scoring_coverage']=0
    elif mutation=='nan': b[0]['frame_cosines'][0]=np.nan
    elif mutation=='length': b[0]['frame_cosines'].pop()
    elif mutation=='bool': b[0]['frame_cosines'][0]=True
    else: b[0]['seed_offset']=43
    with pytest.raises(ValueError): module.summarize_pairs(a,b,['actor1'])


def test_input_order_does_not_change_bootstrap():
    a=[row('a','actor1'),row('b','actor2')]
    b=[row('a','actor1',[.6]*8),row('b','actor2',[.9]*8)]
    assert module.summarize_pairs(a,b,['actor1','actor2'])==module.summarize_pairs(a[::-1],b[::-1],['actor2','actor1'])


def test_seed_strata_not_independent_frame_observations():
    a=[row(seed=42),row(seed=43)]
    b=[row(values=[.6]*8,seed=42),row(values=[.3]*8,seed=43)]
    result=module.summarize_pairs(a,b,['actor1'],43)
    assert result['paired_actors']==1 and result['paired_videos']==1
    assert result['identity_actor_mean_delta']==pytest.approx(-.2)


@pytest.mark.parametrize('value',[True,'42',44])
def test_matching_invalid_seed_offsets_rejected(value):
    a,b=[row(seed=value)],[row(seed=value)]
    with pytest.raises(ValueError,match='seed offset'):
        module.summarize_pairs(a,b,['actor1'])


@pytest.mark.parametrize('value',[True,'1',np.nan,np.inf,-1.])
def test_invalid_motion_rejected(value):
    a,b=[row()],[row()]
    b[0]['scores']['sampled_gray_motion_mean_abs']=value
    with pytest.raises(ValueError,match='motion'):
        module.summarize_pairs(a,b,['actor1'])
