import importlib.util
from pathlib import Path
import pytest
np=pytest.importorskip('numpy')
pytest.importorskip('cv2')
spec=importlib.util.spec_from_file_location('keyframe_analysis',Path(__file__).resolve().parents[1]/'scripts/analyze_keyframe_events.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_common_nonanchors_not_selected_anchor_scores():
    source=np.zeros((33,192,320,3),np.uint8)
    candidate=source.copy()
    candidate[1]=10
    scores=module.error_summary(source,candidate,[0,8,16,24,32],[1,2])
    assert scores['common_nonkey_mae']==5
    assert scores['fullframe_mae']==pytest.approx(10/33)
    assert scores['nonkey_mae']==pytest.approx(10/28)
    assert scores['maximum_anchor_gap']==8


@pytest.mark.parametrize('common',[[],[0],[1,1],[-1],[33],[True]])
def test_invalid_common_frame_set_rejected(common):
    source=np.zeros((33,192,320,3),np.uint8)
    with pytest.raises(ValueError):module.error_summary(source,source,[0,8,16,24,32],common)


def test_invalid_shapes_or_keys_rejected():
    source=np.zeros((33,192,320,3),np.uint8)
    with pytest.raises(ValueError):module.error_summary(source[:32],source,[0,8,16,24,32],[1])
    with pytest.raises(ValueError):module.error_summary(source,source,[0,8,15,24,32],[1])
