import importlib.util
from pathlib import Path
import pytest
np=pytest.importorskip('numpy')
cv2=pytest.importorskip('cv2')
spec=importlib.util.spec_from_file_location('window_probe',Path(__file__).resolve().parents[1]/'scripts/evaluate_window_probe.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_color_centroid_has_no_identity_face_assumption():
    image=np.full((100,200,3),230,dtype=np.uint8)
    image[40:60,60:80]=[0,0,255]
    point=module.color_centroid(image,'red_ball')
    assert point==[69.5,49.5]
    assert module.color_centroid(image,'blue_square') is None


def test_event_direction_and_time_window():
    points=[[i*2,50] for i in range(33)]
    event={'start_s':0,'end_s':2,'action':'moves to the right'}
    result=module.event_score(points,8,event,320,192)
    assert result['direction_pass'] and result['coverage']==1
    event['action']='moves to the left'
    assert not module.event_score(points,8,event,320,192)['direction_pass']


def test_probe_generator_recreates_fixed_pixel_inputs_and_refuses_overwrite(tmp_path,monkeypatch):
    import sys
    spec=importlib.util.spec_from_file_location('prepare_window_probe',Path(__file__).resolve().parents[1]/'scripts/prepare_window_probe.py')
    prepare=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    output=tmp_path/'probe'
    monkeypatch.setattr(sys,'argv',['prepare_window_probe','--output-dir',str(output)])
    prepare.main()
    from ipvg.data import read_manifest
    from ipvg.common import load_json
    rows=read_manifest(output/'manifest.jsonl',check_paths=True)
    assert len(rows)==4 and [r['seed'] for r in rows]==[42,43,42,43]
    assert len(read_manifest(output/'pilot.jsonl'))==1 and len(read_manifest(output/'batch.jsonl'))==3
    assert all(r['timeline'][1]['end_s']==4.125 for r in rows)
    image=cv2.imread(str(output/'red_ball.png'))
    assert image.shape==(192,320,3) and image[96,160].tolist()==[20,20,240]
    assert load_json(output/'preparation_report.json')['status']=='passed'
    with pytest.raises(FileExistsError): prepare.main()


def test_missing_object_is_failure_not_zero_similarity():
    result=module.event_score([None]*33,8,{'start_s':0,'end_s':2,'action':'moves right'},320,192)
    assert result['signed_normalized_displacement'] is None and not result['direction_pass']
