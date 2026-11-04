"""CPU schema and token-mask tests; actual decoder GPU path is separately gated."""
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import teacher_structured_decoder as decoder


def test_schema_covers_original_facts_and_caption_structure():
    import run_teacher_qwen3vl32b_recovery as original
    facts=dict(scene='A red disk rests on a white table.',rendered_or_screen=False,
        entities=[dict(id='E1',description='A red disk',held_object=None)],events=[],
        start_end_difference='The red disk stays at the same position.')
    assert original.validate_facts(facts)==facts
    assert set(decoder.FACTS_SCHEMA['required'])==set(facts)
    assert decoder.FACTS_SCHEMA['properties']['events']['items']['properties']['frame_end']['maximum']==15
    assert decoder.CAPTION_SCHEMA['minItems']==decoder.CAPTION_SCHEMA['maxItems']==4
    assert decoder.FACTS_SCHEMA['additionalProperties'] is False


def test_import_does_not_load_torch_or_decoder_dependency():
    import subprocess
    source=Path(decoder.__file__)
    code='import runpy,sys;runpy.run_path('+repr(str(source))+',run_name="not_main");assert "torch" not in sys.modules;assert "lmformatenforcer" not in sys.modules'
    assert subprocess.run([sys.executable,'-c',code],capture_output=True).returncode==0


def walk(parser,text):
    for char in text:
        if char not in parser.get_allowed_characters():return False
        parser=parser.add_character(char)
    return parser.can_end()


def test_actual_JSON_grammar_keeps_fixed4_and_typed16_frame_facts():
    lm=pytest.importorskip('lmformatenforcer',reason='Actual grammar dependency must be supplied by frozen isolated runtime')
    valid='["one","two","three","four"]'
    assert walk(lm.JsonSchemaParser(decoder.CAPTION_SCHEMA),valid)
    for value in ('[]','["one","two","three"]','["one","two","three","four","five"]',str({'wrong':'object'})):
        assert not walk(lm.JsonSchemaParser(decoder.CAPTION_SCHEMA),value)
    facts=dict(scene='A red disk.',rendered_or_screen=False,entities=[dict(id='E1',description='A red disk',held_object=None)],
        events=[dict(entity_id='E1',frame_start=0,frame_end=15,action='Stays in place')],start_end_difference='No visible change.')
    assert walk(lm.JsonSchemaParser(decoder.FACTS_SCHEMA),json.dumps(facts))
    facts['events'][0]['frame_end']=16
    assert not walk(lm.JsonSchemaParser(decoder.FACTS_SCHEMA),json.dumps(facts))


def test_constraints_use_existing_vocabulary_and_fresh_JSON_parsers():
    from types import SimpleNamespace
    import run_teacher_qwen3vl32b_recovery as original
    tok=SimpleNamespace(eos_token_id=7,encode=lambda text,**kw:[len(text)])
    lm=SimpleNamespace(JsonSchemaParser=lambda schema:schema)
    integration=SimpleNamespace(build_token_enforcer_tokenizer_data=lambda tok:object(),
        build_transformers_prefix_allowed_tokens_fn=lambda data,schema:SimpleNamespace(schema=schema))
    with patch.dict(sys.modules,{'lmformatenforcer':lm,'lmformatenforcer.integrations.transformers':integration}),patch.object(original,'vectorized_bad_words',return_value=lambda ids,scores:scores) as mask:
        model=decoder.StructuredDecoder(tok,[7,8]);facts=model.constraints(True);caption=model.constraints(False)
        assert facts['prefix_allowed_tokens_fn'].schema==decoder.FACTS_SCHEMA
        assert caption['prefix_allowed_tokens_fn'].schema==decoder.CAPTION_SCHEMA
        assert mask.call_count==2 and mask.call_args.args[1]==[7,8]
        with pytest.raises(ValueError):decoder.StructuredDecoder(tok,[8])


def test_model_loading_follows_actual_CUDA_admission_in_new_mode():
    import ast
    source=Path(__file__).resolve().parents[1]/'scripts/run_teacher_temporal_labels.py'
    worker=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='worker')
    calls=[n for n in ast.walk(worker) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    gate=next(n.lineno for n in calls if n.func.attr=='cuda_admission')
    load=next(n.lineno for n in calls if n.func.attr=='from_pretrained' and isinstance(n.func.value,ast.Name) and n.func.value.id=='Qwen3VLForConditionalGeneration')
    assert gate<load
