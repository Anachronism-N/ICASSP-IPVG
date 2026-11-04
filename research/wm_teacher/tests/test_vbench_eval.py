"""CPU-only official-coverage and result-schema checks; not video quality evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import pytest

S = Path(__file__).resolve().parents[1] / 'scripts/vbench_eval.py'
spec = importlib.util.spec_from_file_location('vbench_score_contract', S)
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def full_info():
    core = [dict(prompt_en='synthetic scene '+str(i), dimension=v.DIMS, image_name='image'+str(i)+'.jpg') for i in range(355)]
    camera = [dict(prompt_en='synthetic scene '+str(i)+', '+term, dimension=['camera_motion'], image_name='image'+str(i)+'.jpg')
              for i in range(109) for term in v.CAMERA_TYPES]
    return core + camera


def camera_result():
    rows = [dict(video_path=term+'-'+str(i)+'.mp4', video_results=1.0, prompt_type=target, predict_type=[target])
            for term, target in v.CAMERA_TYPES.items() for i in range(5)]
    return [1.0, {target:1.0 for target in v.CAMERA_TYPES.values()}, rows]


def test_complete_camera_and_core_each_have_five_samples():
    names, expected = v.official_expectations(full_info())
    assert len(names)==5590 and len(expected['camera_motion'])==3815
    assert len(names-expected['camera_motion'])==1775 and len(expected)==9
    assert all('synthetic scene 0-'+str(i)+'.mp4' in names for i in range(5))
    with pytest.raises(ValueError):v.official_expectations(full_info()+[full_info()[0]])
    with pytest.raises(ValueError):v.official_expectations(full_info()[:355])


def test_camera_three_part_result_is_not_legacy_two_part_result():
    result=camera_result();expected={row['video_path'] for row in result[2]}
    assert v.valid_official_result(result,expected,'camera_motion')
    assert not v.valid_result(result,expected)
    assert not v.valid_official_result([result[0],result[2]],expected,'camera_motion')


@pytest.mark.parametrize('corruption',['missing','duplicate','wrong_type','wrong_score','wrong_prediction','wrong_mean','wrong_type_mean','nan','string_score'])
def test_camera_result_rejects_partial_or_inconsistent_evidence(corruption):
    result=camera_result();expected={row['video_path'] for row in result[2]}
    if corruption=='missing':result[2].pop()
    elif corruption=='duplicate':result[2][-1]=copy.deepcopy(result[2][0])
    elif corruption=='wrong_type':result[2][0]['prompt_type']='static'
    elif corruption=='wrong_score':result[2][0]['video_results']=0.0
    elif corruption=='wrong_prediction':result[2][0]['predict_type']=[]
    elif corruption=='wrong_mean':result[0]=0.0
    elif corruption=='wrong_type_mean':result[1]['pan_left']=0.0
    elif corruption=='nan':result[2][0]['video_results']=float('nan')
    else:result[2][0]['video_results']='1'
    assert not v.valid_official_result(result,expected,'camera_motion')


def test_dynamic_booleans_preserved_other_dimensions_require_numbers():
    result=[0.5,[dict(video_path='a.mp4',video_results=True),dict(video_path='b.mp4',video_results=False)]]
    assert v.valid_official_result(result,{'a.mp4','b.mp4'},'dynamic_degree')
    assert not v.valid_official_result(result,{'a.mp4','b.mp4'},'imaging_quality')


def test_official_mode_rejects_incomplete_videos_before_scoring(tmp_path,monkeypatch):
    repo=tmp_path/'repo';(repo/'vbench2_beta_i2v').mkdir(parents=True)
    (repo/'vbench2_beta_i2v/vbench2_i2v_full_info.json').write_text(json.dumps(full_info()))
    source=tmp_path/'eval/vbench_i2v/videos_official5/candidate';source.mkdir(parents=True)
    scorer=SimpleNamespace(VBenchI2V=lambda *args: (_ for _ in ()).throw(AssertionError('GPU scoring must not start')))
    monkeypatch.setattr(v,'VBREPO',str(repo));monkeypatch.setattr(v,'EV',str(tmp_path/'eval'))
    monkeypatch.setitem(sys.modules,'vbench2_beta_i2v',scorer)
    monkeypatch.setattr(sys,'argv',[str(S),'--suite','vbench','--model','candidate','--official-five'])
    with patch.object(v.os,'chdir'),pytest.raises(RuntimeError,match='coverage'):
        v.main()
    assert not (tmp_path/'eval/vbench_i2v/scores_official5').exists()


def test_official_mode_does_not_overwrite_existing_invalid_scores(tmp_path,monkeypatch):
    repo=tmp_path/'repo';(repo/'vbench2_beta_i2v').mkdir(parents=True)
    info=full_info();(repo/'vbench2_beta_i2v/vbench2_i2v_full_info.json').write_text(json.dumps(info))
    names,_=v.official_expectations(info)
    staged=tmp_path/'staged/videos';staged.mkdir(parents=True)
    out=tmp_path/'eval/vbench_i2v/scores_official5';out.mkdir(parents=True)
    result_path=out/'candidate__camera_motion_eval_results.json';original=json.dumps({'camera_motion':[1,{},[]]});result_path.write_text(original)
    scorer=SimpleNamespace(VBenchI2V=lambda *args:SimpleNamespace(evaluate=lambda **kw: (_ for _ in ()).throw(AssertionError('No automatic rescore'))))
    monkeypatch.setattr(v,'VBREPO',str(repo));monkeypatch.setattr(v,'EV',str(tmp_path/'eval'))
    monkeypatch.setitem(sys.modules,'vbench2_beta_i2v',scorer)
    monkeypatch.setattr(v,'prepare_vbench',lambda *args,**kw:(str(staged),'metadata',5590))
    monkeypatch.setattr(sys,'argv',[str(S),'--suite','vbench','--model','candidate','--dims','camera_motion','--official-five'])
    with patch.object(v.os,'chdir'),patch.object(v.os,'listdir',return_value=list(names)),patch.object(v.os.path,'getsize',return_value=1),patch.object(v,'bind_official_score_inputs'),pytest.raises(RuntimeError,match='no automatic retry'):
        v.main()
    assert result_path.read_text()==original
    assert not (tmp_path/'eval/vbench_i2v/scores').exists()


def test_wrong_suite_is_rejected_before_official_import(monkeypatch):
    monkeypatch.setattr(sys,'argv',[str(S),'--suite','wm','--model','candidate','--official-five'])
    with patch.object(v.os,'chdir') as cd,pytest.raises(SystemExit):v.main()
    cd.assert_not_called()


def test_official_staging_rejects_stale_same_named_video_bytes(tmp_path,monkeypatch):
    source=tmp_path/'eval/vbench_i2v/videos_official5/candidate';source.mkdir(parents=True)
    staged=tmp_path/'work/vbench_official5/candidate/videos';staged.mkdir(parents=True)
    (source/'scene-0.mp4').write_bytes(b'new');(staged/'scene-0.mp4').write_bytes(b'old')
    monkeypatch.setattr(v,'EV',str(tmp_path/'eval'));monkeypatch.setattr(v,'WORK',str(tmp_path/'work'))
    with pytest.raises(RuntimeError,match='bytes differ'):v.prepare_vbench('candidate',official_five=True)
    assert (staged/'scene-0.mp4').read_bytes()==b'old' and (source/'scene-0.mp4').read_bytes()==b'new'


def test_official_binding_refuses_changed_video_or_metadata(tmp_path):
    out=tmp_path/'scores';out.mkdir();videos=tmp_path/'videos';videos.mkdir()
    metadata=tmp_path/'metadata.json';metadata.write_text('{}');clip=videos/'scene-0.mp4';clip.write_bytes(b'original')
    args=(str(out),'candidate',str(videos),str(metadata),{'scene-0.mp4'})
    binding=v.bind_official_score_inputs(*args)
    assert binding==v.bind_official_score_inputs(*args)
    original=(out/'candidate__score_inputs.json').read_bytes();clip.write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='inputs changed'):v.bind_official_score_inputs(*args)
    clip.write_bytes(b'original');metadata.write_text('{"changed":true}')
    with pytest.raises(RuntimeError,match='inputs changed'):v.bind_official_score_inputs(*args)
    assert (out/'candidate__score_inputs.json').read_bytes()==original


def test_unbound_official_scores_never_reused_or_overwritten(tmp_path):
    out=tmp_path/'scores';out.mkdir();videos=tmp_path/'videos';videos.mkdir()
    metadata=tmp_path/'metadata.json';metadata.write_text('{}');(videos/'scene-0.mp4').write_bytes(b'original')
    old=out/'candidate__camera_motion_eval_results.json';old.write_text('{"unbound":true}')
    with pytest.raises(RuntimeError,match='lack input byte binding'):
        v.bind_official_score_inputs(str(out),'candidate',str(videos),str(metadata),{'scene-0.mp4'})
    assert old.read_text()=='{"unbound":true}' and not (out/'candidate__score_inputs.json').exists()


def test_input_bindings_are_separate_for_different_models(tmp_path):
    out=tmp_path/'scores';out.mkdir();metadata=tmp_path/'metadata.json';metadata.write_text('{}')
    a=tmp_path/'a';a.mkdir();(a/'scene-0.mp4').write_bytes(b'a')
    b=tmp_path/'b';b.mkdir();(b/'scene-0.mp4').write_bytes(b'b')
    left=v.bind_official_score_inputs(str(out),'model_a',str(a),str(metadata),{'scene-0.mp4'})
    right=v.bind_official_score_inputs(str(out),'model_b',str(b),str(metadata),{'scene-0.mp4'})
    assert left!=right and len(list(out.glob('*__score_inputs.json')))==2


def test_legacy_result_contract_unchanged():
    result=[0.5,[dict(video_path='scene-0.mp4',video_results=0.5)]]
    assert v.valid_result(result,{'scene-0.mp4'})
    assert not v.valid_result(result,{'scene-1.mp4'})
