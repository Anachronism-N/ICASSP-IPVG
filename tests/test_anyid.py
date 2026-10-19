import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('anyid_adapter',Path(__file__).resolve().parents[1]/'scripts/adapters/anyid.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_anyid_keeps_one_reference_prefix_separate_from_output():
    assert module.reference_latent_lengths(49)==[1,13]
    assert module.reference_latent_lengths(17)==[1,5]
    assert module.reference_latent_lengths(49,2)==[2,13]
    assert module.reference_latent_lengths(49,5)==[5,13]


@pytest.mark.parametrize('count',[0,6,True,2.0,'2',None])
def test_invalid_reference_counts_fail(count):
    with pytest.raises(ValueError,match='one to five'):
        module.reference_latent_lengths(49,count)


def test_seed_override_keeps_sample_id_and_legacy_seed():
    import hashlib
    offset=int(hashlib.sha256(b'sample_01').hexdigest()[:8],16)
    assert module.sample_seed({'sample_id':'sample_01'},{})==(42+offset)%2**31
    assert module.sample_seed({'sample_id':'sample_01','generation_seed':43},{'seed':42})==(43+offset)%2**31


@pytest.mark.parametrize('value',[True,-1,2**31,42.0,'42',None])
def test_seed_override_rejects_invalid_values(value):
    with pytest.raises(ValueError,match='Generation seed'):
        module.sample_seed({'sample_id':'sample_01','generation_seed':value},{})


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
