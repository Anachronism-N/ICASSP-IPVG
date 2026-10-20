import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('dense_training',Path(__file__).resolve().parents[1]/'scripts/prepare_dense_training.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_dense_caption_bounds_and_refusal():
    assert module.validate_caption(' '.join(['word']*130))==130
    assert module.validate_caption(' '.join(['word']*188))==188
    assert module.validate_caption(' '.join(['word']*200))==200
    for count in (99,201):
        with pytest.raises(ValueError):module.validate_caption(' '.join(['word']*count))
    with pytest.raises(ValueError):module.validate_caption('I cannot view '+' '.join(['word']*120))


def test_sampling_covers_full_source_and_fixed_frame_rule():
    pytest.importorskip('numpy')
    for count in (51,86):
        indices=module.sampling_indices(count)
        assert len(indices)==49 and indices[0]==0 and indices[-1]==count-1
        assert indices==sorted(set(indices))
    assert len(module.sampling_indices(51,17))==17
    for count in (True,48,51.0):
        with pytest.raises(ValueError):module.sampling_indices(count)
