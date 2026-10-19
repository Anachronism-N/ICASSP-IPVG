import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('anyid_adapter',Path(__file__).resolve().parents[1]/'scripts/adapters/anyid.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_anyid_keeps_one_reference_prefix_separate_from_output():
    assert module.reference_latent_lengths(49)==[1,13]
    assert module.reference_latent_lengths(17)==[1,5]


def test_text_device_default_and_opt_in():
    assert module.text_encoder_on_cpu({}) is True
    assert module.text_encoder_on_cpu({'t5_cpu':False}) is False
    for invalid in ('false',0,1,None):
        with pytest.raises(ValueError,match='JSON boolean'):
            module.text_encoder_on_cpu({'t5_cpu':invalid})


def test_timed_text_encoder_delegates_without_changing_outputs():
    calls=[]
    synchronizations=[]
    marker=object()
    class Context:
        shape=(512,4096)
    context=Context()
    class Encoder:
        model=marker
        def __call__(self,texts,device):
            calls.append((texts,device))
            return [context]
    wrapper=module.TimedTextEncoder(Encoder(),lambda:synchronizations.append(True))
    result=wrapper(['unchanged prompt'],'cpu')
    assert result[0] is context and wrapper.model is marker
    assert calls==[(['unchanged prompt'],'cpu')]
    assert len(synchronizations)==2 and len(wrapper.calls)==1
    assert wrapper.calls[0]['device']=='cpu'
    assert wrapper.calls[0]['context_shapes']==[[512,4096]]
    assert wrapper.calls[0]['elapsed_seconds']>=0


@pytest.mark.parametrize('value',[0,48,-1,True,49.0])
def test_anyid_frame_constraint(value):
    with pytest.raises(ValueError): module.reference_latent_lengths(value)
