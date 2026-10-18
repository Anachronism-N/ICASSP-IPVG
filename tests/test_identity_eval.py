import importlib.util
from pathlib import Path
import pytest
np = pytest.importorskip('numpy')
pytest.importorskip('cv2')

spec=importlib.util.spec_from_file_location('identity_eval_test',Path(__file__).resolve().parents[1]/'scripts/identity_eval.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_frame_sampling_is_fixed_and_unique():
    indices=module.frame_indices(49)
    assert len(indices)==8 and len(set(indices))==8 and indices[0]==0 and indices[-1]==48
    with pytest.raises(ValueError): module.frame_indices(7)


def test_missing_identity_is_not_zero_and_negative_cosine_is_valid():
    assert module.scores([None]*8)=={'face_detection_coverage':0}
    result=module.scores([-.4,.2]+[None]*6)
    assert result['face_detection_coverage']==.25
    assert result['face_identity_cosine_mean']==pytest.approx(-.1)
    assert result['face_identity_cosine_min']==-.4


def test_positive_negative_roc_and_actor_bootstrap():
    stats=module.roc_stats([.8,.9,.7],[.1,.2,.3])
    assert stats['auc']==1 and stats['eer']==0
    interval=module.bootstrap_auc(['a','b'],{'a':[.8],'b':[.9]},{'a':[.1],'b':[.2]},count=20)
    assert interval==[1,1]


def test_video_threshold_is_not_applied_to_low_coverage_frames():
    assert 'face_identity_video_below_threshold' not in module.scores([.1]+[None]*7,threshold=.2)
    assert module.scores([.1]*8,threshold=.2)['face_identity_video_below_threshold']==1


def test_interpolated_eer_not_nearest_roc_point_average():
    stats=module.roc_stats([.9,.4,.3,.2],[.8,.7,.6,.1])
    assert 0<=stats['eer']<=1 and stats['auc']<1


def test_pair_protocol_mismatch_rejected(tmp_path):
    import json
    left=tmp_path/'left.json'
    right=tmp_path/'right.json'
    left.write_text(json.dumps({'protocol':{},'weight_hashes':{},'records':[]}))
    right.write_text(json.dumps({'protocol':{'version':'other'},'weight_hashes':{},'records':[]}))
    with pytest.raises(ValueError,match='protocol'):
        module.paired_report(left,right,tmp_path/'result.json')


def test_empty_pairs_rejected_without_nan_output(tmp_path):
    import json
    left=tmp_path/'left.json'
    right=tmp_path/'right.json'
    value=json.dumps({'protocol':{},'weight_hashes':{},'records':[]})
    left.write_text(value)
    right.write_text(value)
    with pytest.raises(ValueError,match='no samples'):
        module.paired_report(left,right,tmp_path/'result.json')


@pytest.mark.parametrize('values',[[.2]*7,[.2]*9,[float('nan')]*8])
def test_pair_rejects_wrong_frame_count_and_nonfinite(values,tmp_path):
    import json
    record={'sample_id':'sample','identity_id':'actor','frame_cosines':values,'scores':{'face_detection_coverage':1,'sampled_gray_motion_mean_abs':1}}
    report={'protocol':{},'weight_hashes':{},'records':[record]}
    left,right=tmp_path/'left.json',tmp_path/'right.json'
    left.write_text(json.dumps(report))
    right.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='exactly eight'):
        module.paired_report(left,right,tmp_path/'result.json')


def test_no_encoder_auto_download(tmp_path):
    pytest.importorskip('onnxruntime')
    pytest.importorskip('insightface')
    with pytest.raises(FileNotFoundError,match='automatic downloads disabled'):
        module.OfflineFaceEncoder(tmp_path)


def test_pairs_use_target_actor_and_cross_actor_only():
    refs={'a':{'face':{'embedding':np.array([1.,0.])}},'b':{'face':{'embedding':np.array([0.,1.])}}}
    items=[{'actor':'a','valid':[np.array([1.,0.])]*8},{'actor':'b','valid':[np.array([0.,1.])]*8}]
    p,n,_,_=module.split_pairs(['a','b'],items,refs)
    assert p==[1,1] and n==[0,0]
