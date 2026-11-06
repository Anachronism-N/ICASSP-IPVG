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
    import transformers
    from types import SimpleNamespace
    import run_teacher_qwen3vl32b_recovery as original
    tok=SimpleNamespace(eos_token_id=7,encode=lambda text,**kw:[len(text)])
    lm=SimpleNamespace(JsonSchemaParser=lambda schema:schema)
    integration=SimpleNamespace(build_token_enforcer_tokenizer_data=lambda tok:SimpleNamespace(regular_tokens=[]),
        build_transformers_prefix_allowed_tokens_fn=lambda data,schema:SimpleNamespace(schema=schema,token_enforcer=object()))
    with patch.dict(sys.modules,{'lmformatenforcer':lm,'lmformatenforcer.integrations.transformers':integration}),patch.object(original,'vectorized_bad_words',return_value=lambda ids,scores:scores) as mask:
        model=decoder.StructuredDecoder(tok,[7,8]);facts=model.constraints(True);caption=model.constraints(False)
        assert callable(facts['prefix_allowed_tokens_fn'])
        assert callable(caption['prefix_allowed_tokens_fn'])
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


@pytest.mark.parametrize('facts_stage', [True, False])
def test_real_Qwen_boundary_tokens_keep_complete_JSON_and_original_denials(facts_stage):
    pytest.importorskip('lmformatenforcer')
    model = Path('/opt/models/Qwen3-VL-32B-Instruct')
    if not (model/'tokenizer.json').exists():
        pytest.skip('Frozen local tokenizer is required for actual CPU contract')
    import torch
    from transformers import AutoTokenizer, PrefixConstrainedLogitsProcessor
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    eos = json.loads((model/'generation_config.json').read_bytes())['eos_token_id']
    instance = decoder.StructuredDecoder(tokenizer, eos)
    facts = dict(scene='A red disk.', rendered_or_screen=False,
        entities=[dict(id='E1', description='A red disk', held_object=None)],
        events=[dict(entity_id='E1', frame_start=0, frame_end=15, action='Stays in place.')],
        start_end_difference='No visible change.')
    text = json.dumps(facts) if facts_stage else json.dumps(['one.', 'two.', 'three.', 'four.'])
    constraints = instance.constraints(facts_stage)
    prefix = constraints['prefix_allowed_tokens_fn']
    mask = PrefixConstrainedLogitsProcessor(prefix, 1)
    sent = tokenizer.encode('Synthetic JSON:', add_special_tokens=False)
    prefix(0, torch.tensor(sent))
    for token in tokenizer.encode(text, add_special_tokens=False):
        inputs = torch.tensor([sent])
        scores = torch.zeros((1, len(tokenizer)), dtype=torch.bfloat16)
        values = constraints['logits_processor'](inputs, mask(inputs, scores))
        assert torch.isfinite(values[0, token])
        assert all(torch.isneginf(values[0, stop]) for stop in eos)
        sent.append(token)
    assert tokenizer.eos_token_id in prefix(0, torch.tensor(sent))
    assert not torch.cuda.is_initialized()


def test_boundary_trie_uses_character_grammar_not_unfiltered_quote_allowlist():
    pytest.importorskip('lmformatenforcer')
    import torch
    from types import SimpleNamespace
    from lmformatenforcer import JsonSchemaParser
    instance = object.__new__(decoder.StructuredDecoder)
    instance.boundary_tree = {'.': {'"': {',': {None: [10]}, '}': {None: [11]}}}}
    parser = JsonSchemaParser(decoder.FACTS_SCHEMA)
    for char in '{"scene":"disk':
        parser = parser.add_character(char)
    class Prefix:
        token_enforcer = SimpleNamespace(prefix_states={(3,): SimpleNamespace(parser=parser)})
        def __call__(self, batch, sent): return [7]
    assert instance.boundary_prefix(Prefix())(0, torch.tensor([3])) == [7, 10]


def test_bounded_facts_is_separate_keeps_original_structure_and_caption_gate():
    schema = decoder.bounded_facts_schema()
    assert 'maxLength' not in decoder.FACTS_SCHEMA['properties']['scene']
    assert schema['required'] == decoder.FACTS_SCHEMA['required']
    assert schema['properties']['entities']['maxItems'] == 32
    assert schema['properties']['events']['maxItems'] == 128
    assert schema['properties']['scene']['maxLength'] == 160
    assert schema['properties']['events']['items']['properties']['action']['maxLength'] == 120
    assert decoder.CAPTION_SCHEMA['minItems'] == decoder.CAPTION_SCHEMA['maxItems'] == 4
    lm = pytest.importorskip('lmformatenforcer')
    prefix = '{"scene":"' + 'a' * 160
    parser = lm.JsonSchemaParser(schema)
    for char in prefix:
        assert char in parser.get_allowed_characters()
        parser = parser.add_character(char)
    assert parser.get_allowed_characters() == '"'
    assert not parser.can_end()


def test_actual_Qwen_bounded_string_forbids_adversarial_continuation_without_EOS_repair():
    pytest.importorskip('lmformatenforcer')
    model = Path('/opt/models/Qwen3-VL-32B-Instruct')
    if not (model/'tokenizer.json').exists():
        pytest.skip('Frozen local tokenizer is required for actual CPU contract')
    import torch
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    eos = json.loads((model/'generation_config.json').read_bytes())['eos_token_id']
    sample = dict(scene='a' * 160, rendered_or_screen=False,
        entities=[dict(id='E1', description='A red disk', held_object=None)], events=[],
        start_end_difference='No visible change.')
    import run_teacher_qwen3vl32b_recovery as original
    assert original.validate_facts(sample) == sample
    for bounded in (False, True):
        instance = decoder.StructuredDecoder(tokenizer, eos, bounded=bounded)
        constraints = instance.constraints(True)
        fn = constraints['prefix_allowed_tokens_fn']
        prompt = tokenizer.encode('Synthetic facts:', add_special_tokens=False)
        sent = list(prompt)
        fn(0, torch.tensor(sent))
        for token in tokenizer.encode('{"scene":"', add_special_tokens=False):
            assert token in fn(0, torch.tensor(sent))
            sent.append(token)
        token = tokenizer.encode('a', add_special_tokens=False)
        assert len(token) == 1
        for _ in range(160):
            assert token[0] in fn(0, torch.tensor(sent))
            sent.append(token[0])
        choices = fn(0, torch.tensor(sent))
        assert (token[0] in choices) is (not bounded)
        assert tokenizer.eos_token_id not in choices
        assert tokenizer.encode('"', add_special_tokens=False)[0] in choices
    fn = decoder.StructuredDecoder(tokenizer, eos, bounded=True).constraints(True)['prefix_allowed_tokens_fn']
    sent = list(prompt)
    fn(0, torch.tensor(sent))
    for token in tokenizer.encode(json.dumps(sample), add_special_tokens=False):
        assert token in fn(0, torch.tensor(sent))
        sent.append(token)
    assert tokenizer.eos_token_id in fn(0, torch.tensor(sent))
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('words', [1, 11, 12, 22, 23, 30])
def test_caption_word_schema_matches_original_counter_bounds(words):
    import re
    lm = pytest.importorskip('lmformatenforcer')
    sentence = ' '.join(['visible'] * words) + '.'
    assert len(re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", sentence)) == words
    assert walk(lm.JsonSchemaParser(decoder.caption_word_schema()), json.dumps([sentence] * 4)) == (12 <= words <= 22)
    assert walk(lm.JsonSchemaParser(decoder.CAPTION_SCHEMA), json.dumps([sentence] * 4))
    if 12 <= words <= 22:
        import run_teacher_qwen3vl32b_recovery as original
        assert original.validate_caption(' '.join([sentence] * 4))


@pytest.mark.parametrize('value', [
    ['short.'] * 4,
    [' '.join(['visible'] * 12) + '.'] * 3,
    [' '.join(['visible'] * 12) + '.'] * 5,
    [' ' + ' '.join(['visible'] * 12) + '.'] * 4,
    [' '.join(['visible'] * 12) + '. extra.'] * 4,
    [' '.join(['visible'] * 12) + '\n'] * 4,
])
def test_caption_word_schema_rejects_invalid_sentences_without_repair(value):
    lm = pytest.importorskip('lmformatenforcer')
    assert not walk(lm.JsonSchemaParser(decoder.caption_word_schema()), json.dumps(value))


def test_caption_word_schema_contractions_hyphens_and_numbers_match_counter():
    import re
    lm = pytest.importorskip('lmformatenforcer')
    sentence = "A red-brown disk can't leave its small table, while 2 nearby objects remain still."
    count = len(re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", sentence))
    assert 12 <= count <= 22
    assert walk(lm.JsonSchemaParser(decoder.caption_word_schema()), json.dumps([sentence] * 4))


def test_real_Qwen_word_count_masks_early_close_and_extra_word_without_model_load():
    pytest.importorskip('lmformatenforcer')
    model = Path('/opt/models/Qwen3-VL-32B-Instruct')
    if not (model/'tokenizer.json').exists():
        pytest.skip('Frozen local tokenizer is required for actual CPU contract')
    import torch
    from transformers import AutoTokenizer, PrefixConstrainedLogitsProcessor
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    eos = json.loads((model/'generation_config.json').read_bytes())['eos_token_id']
    instance = decoder.StructuredDecoder(tokenizer, eos, bounded=True, word_count=True)
    for words in (11, 12, 22):
        constraints = instance.constraints(False)
        fn = constraints['prefix_allowed_tokens_fn']
        prompt = tokenizer.encode('Synthetic JSON:', add_special_tokens=False)
        sent = list(prompt)
        fn(0, torch.tensor(sent))
        text = '["' + ' '.join(['visible'] * words)
        for token in tokenizer.encode(text, add_special_tokens=False):
            assert token in fn(0, torch.tensor(sent))
            sent.append(token)
        ids = torch.tensor([sent])
        scores = torch.zeros((1, len(tokenizer)), dtype=torch.bfloat16)
        mask = PrefixConstrainedLogitsProcessor(fn, 1)
        values = constraints['logits_processor'](ids, mask(ids, scores))
        period = tokenizer.encode('.', add_special_tokens=False)
        extra_word = tokenizer.encode(' visible', add_special_tokens=False)
        assert len(period) == len(extra_word) == 1
        assert bool(torch.isfinite(values[0, period[0]])) == (words >= 12)
        assert bool(torch.isfinite(values[0, extra_word[0]])) == (words < 22)
        assert all(torch.isneginf(values[0, stop]) for stop in eos)
    proof = instance.cuda_admission('cpu')
    assert proof['caption_word_count_path_pass'] and proof['bounded_string_CPU_input_score_path_pass']
    assert not proof['CPU_input_GPU_score_path_pass']
    assert not torch.cuda.is_initialized()
