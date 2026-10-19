import importlib.util
from pathlib import Path
import pytest
np=pytest.importorskip('numpy')
pytest.importorskip('cv2')
spec=importlib.util.spec_from_file_location('mouth_motion',Path(__file__).resolve().parents[1]/'scripts/mouth_motion_eval.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_eye_alignment_maps_exactly_without_using_mouth():
    eyes=np.array([[20.,30.],[50.,40.]])
    matrix=module.eye_transform(eyes)
    mapped=np.column_stack((eyes,np.ones(2)))@matrix.T
    np.testing.assert_allclose(mapped,[[38.2946,51.6963],[73.5318,51.5014]],atol=1e-4)


@pytest.mark.parametrize('eyes',[[[1,1],[1,1]],[[np.nan,1],[2,3]],[1,2]])
def test_eye_alignment_invalid(eyes):
    with pytest.raises(ValueError): module.eye_transform(eyes)


def test_rigid_uniform_flow_is_not_local_mouth_activity():
    flow=np.tile(np.array([2.,-3.],dtype=np.float32),(112,112,1))
    assert module.residual_flow_score(flow)==0
    flow[80:106,28:84,0]+=1
    assert module.residual_flow_score(flow)==pytest.approx(1.)


def test_missing_does_not_bridge_and_static_score_zero():
    frame=np.zeros((112,112),np.uint8)
    frames=[frame.copy() for _ in range(17)]
    frames[8]=None
    result=module.sequence_scores(frames)
    assert result['pair_coverage']==14/16 and result['frame_coverage']==16/17
    assert result['pair_scores'][7:9]==[None,None]
    assert result['mouth_residual_flow_mean']==0


def test_low_coverage_omits_score():
    frame=np.zeros((112,112),np.uint8)
    result=module.sequence_scores([frame if i%2 else None for i in range(17)])
    assert result['pair_coverage']==0 and 'mouth_residual_flow_mean' not in result


@pytest.mark.parametrize('flow',[np.zeros((10,10,2)),np.full((112,112,2),np.inf)])
def test_bad_flow_rejected(flow):
    with pytest.raises(ValueError): module.residual_flow_score(flow)
