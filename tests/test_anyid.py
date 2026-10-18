import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('anyid_adapter',Path(__file__).resolve().parents[1]/'scripts/adapters/anyid.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_anyid_keeps_one_reference_prefix_separate_from_output():
    assert module.reference_latent_lengths(49)==[1,13]
    assert module.reference_latent_lengths(17)==[1,5]


@pytest.mark.parametrize('value',[0,48,-1,True,49.0])
def test_anyid_frame_constraint(value):
    with pytest.raises(ValueError): module.reference_latent_lengths(value)
