import importlib.util
from pathlib import Path
import pytest
np=pytest.importorskip('numpy')
pytest.importorskip('cv2')
spec=importlib.util.spec_from_file_location('keyframe_event',Path(__file__).resolve().parents[1]/'scripts/keyframe_event_ablation.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_uniform_and_event_preserve_boundary_and_fixed_budget():
    frames=[np.full((32,32,3),i,np.uint8) for i in range(33)]
    assert module.select_keyframes(frames)==[0,8,16,24,32]
    keys=module.select_keyframes(frames,mode='event_error')
    assert len(keys)==5 and {0,16,32}<=set(keys)
    schedule=module.reconstruction_schedule(keys)
    assert len(schedule)==28 and len({r[0] for r in schedule})==28
    assert {r[0] for r in schedule}|set(keys)==set(range(33))
    assert all(0<t<1 and t==pytest.approx((i-l)/(r-l)) for i,l,r,t in schedule)


def test_event_selection_reduces_frozen_linear_reconstruction_cost():
    frames=[np.full((32,32,3),min(i,5)*10 if i<=16 else (i-16)*4,np.uint8) for i in range(33)]
    chosen=module.select_keyframes(frames,mode='event_error')
    def cost(keys):
        return sum(module.interval_error(np.stack(frames).astype(np.float32),a,b) for a,b in zip(keys[:-1],keys[1:]))
    assert cost(chosen)<=cost([0,8,16,24,32]) and chosen==module.select_keyframes(frames,mode='event_error')


@pytest.mark.parametrize('keys',[[0,8,15,24,32],[0,8,16,32],[0,8,8,16,32],[8,16,24,30,32]])
def test_malformed_schedule_rejected(keys):
    with pytest.raises(ValueError):module.reconstruction_schedule(keys)


def test_wrong_source_and_mode_rejected():
    frames=[np.zeros((32,32,3),np.uint8)]*33
    with pytest.raises(ValueError):module.select_keyframes(frames[:32])
    with pytest.raises(ValueError):module.select_keyframes(frames,boundary=17)
    with pytest.raises(ValueError):module.select_keyframes(frames,mode='unknown')
