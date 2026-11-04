"""CPU input and protocol contracts; never semantic or GPU quality certification."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
import pytest
S=Path(__file__).resolve().parents[1]/'scripts/run_teacher_temporal_labels.py'
spec=importlib.util.spec_from_file_location('temporal_labels',S)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

SENTENCES=[
'A person in a dark shirt stands beside a table in a plain indoor room.',
'Another person wearing a light shirt nearby holds a small dark object in one hand.',
'The first person turns toward the table while the second person raises their visible arm.',
'The table stays in place, and both people finish standing beside it in the room.']

def test_import_and_help_no_torch_or_gpu():
    code='import runpy,sys;runpy.run_path('+repr(str(S))+',run_name="not_main");assert "torch" not in sys.modules'
    assert subprocess.run([sys.executable,'-c',code],capture_output=True).returncode==0
    assert subprocess.run([sys.executable,str(S),'--help'],capture_output=True).returncode==0

def test_json_renderer_keeps_exact_sentences_no_padding_or_shortening():
    assert a.parse_caption(json.dumps(SENTENCES))==' '.join(SENTENCES)

@pytest.mark.parametrize('value',[SENTENCES[:3],SENTENCES+['extra'],[1,*SENTENCES[1:]],
    [' padded ',*SENTENCES[1:]],['line\nline',*SENTENCES[1:]],['short.']*4,
    [' '.join(['visible']*40)]*4,dict(sentences=SENTENCES)])
def test_json_renderer_rejects_invalid_without_repair(value):
    with pytest.raises((AssertionError,ValueError)):a.parse_caption(json.dumps(value))

def test_native_input_fake_contract_exact16_source_timestamps_no_resampling():
    import torch
    from PIL import Image
    class Processor:
        video_processor=SimpleNamespace(temporal_patch_size=2)
        def apply_chat_template(self,messages,**kw):
            assert messages[0]['content'][0]=={'type':'video'} and kw['tokenize'] is False
            return 'synthetic native video prompt'
        def __call__(self,**kw):
            assert kw['videos'][0].shape==(16,352,640,3) and kw['do_sample_frames'] is False
            assert kw['video_metadata'][0]['frames_indices']==list(range(0,121,8))
            assert kw['video_metadata'][0]['fps']==24
            return {'video_grid_thw':torch.tensor([[8,22,40]])}
    images=[Image.new('RGB',(640,352)) for _ in range(16)]
    assert a.native_inputs(Processor(),images,'visible scene')['video_grid_thw'][0,0]==8
    with pytest.raises(AssertionError):a.native_inputs(Processor(),images[:-1],'visible scene')

@pytest.mark.parametrize('complete',[True,False])
def test_generation_single_greedy_call_requires_eos(complete):
    import torch
    class Inputs(dict):
        def to(self,device):return self
    model=SimpleNamespace(get_input_embeddings=lambda:SimpleNamespace(weight=SimpleNamespace(device='cpu')),
                          generation_config=SimpleNamespace(eos_token_id=[7]))
    calls=[]
    def generate(**kw):
        calls.append(kw);return torch.tensor([[1,2,5,7 if complete else 5]])
    model.generate=generate
    processor=SimpleNamespace(batch_decode=lambda *args,**kw:['candidate'])
    trace={}
    with patch.object(a,'native_inputs',return_value=Inputs(input_ids=torch.tensor([[1,2]]))):
        if complete:assert a.generate(model,processor,[],'prompt',384,trace)=='candidate'
        else:
            with pytest.raises(AssertionError):a.generate(model,processor,[],'prompt',384,trace)
    assert len(calls)==1 and calls[0]['do_sample'] is False and calls[0]['max_new_tokens']==384
    assert trace['new_tokens']==2

def test_unauthorized_run_no_allocation(tmp_path):
    with patch.object(a,'E',tmp_path),pytest.raises(AssertionError):a.run(SimpleNamespace(gpu_authorized=False))
    assert not list(tmp_path.iterdir())


def test_cost_clock_precedes_prelaunch_hashes_and_claim():
    import ast
    tree=ast.parse(S.read_text())
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    started=next(n for n in ast.walk(run) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='started' for t in n.targets))
    verify=next(n for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='verify')
    claim=next(n for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='save')
    assert started.lineno<verify.lineno<claim.lineno
    assert 'CAP-60-(time.monotonic()-started)' in S.read_text()


def test_schema_allocation_is_separate_and_preserves_original32_and_word_gates(monkeypatch):
    for k in ('E','AUTH_SHA','SCHEMA_DECODE'):
        monkeypatch.setattr(a,k,getattr(a,k))
    monkeypatch.setattr(sys,'path',list(sys.path))
    a.configure_schema()
    cfg=a.config()
    assert a.E.name=='allocation005_schema' and cfg['source_frames']==16
    assert cfg['source_indices']==list(range(0,121,8)) and cfg['caption_words']==[40,120]
    assert cfg['facts_max_tokens']==1024 and cfg['caption_max_tokens']==384
    assert cfg['do_sample'] is False and cfg['formal_training_allowed'] is False
    assert cfg['decoding']=='JSON_schema_and_original_deny_vocabulary'


def test_original_caption_validator_not_relaxed_in_schema_mode(monkeypatch):
    monkeypatch.setattr(a,'SCHEMA_DECODE',True)
    assert a.parse_caption(json.dumps(SENTENCES))==' '.join(SENTENCES)
    with pytest.raises(ValueError):a.parse_caption(json.dumps(['short.']*4))


def test_boundary_mode_is_independent_preserves_budget_and_stops_decoder_failures(monkeypatch):
    for k in ('E','AUTH_SHA','SCHEMA_DECODE','BOUNDARY_DECODE'):
        monkeypatch.setattr(a,k,getattr(a,k))
    monkeypatch.setattr(sys,'path',list(sys.path))
    a.configure_boundary()
    cfg=a.config()
    assert a.E.name=='allocation006_boundary'
    assert cfg['source_frames']==16 and cfg['caption_words']==[40,120]
    assert cfg['facts_max_tokens']==1024 and cfg['caption_max_tokens']==384
    assert cfg['stop_on_generation_without_EOS_or_AssertionError'] is True
    assert cfg['decoder_boundary_tokens']=='character_validated_quote_crossing_trie_v1'
    assert 'decoder_admission_failed_stop_without_retry' in S.read_text()
    assert 'if decoder_stop:stop=' in S.read_text()
