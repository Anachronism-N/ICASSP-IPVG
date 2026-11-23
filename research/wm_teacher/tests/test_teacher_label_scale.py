import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_teacher_label_scale as ls  # noqa: E402


def test_text_integrity_rejects_glued_cjk_and_hyphen_chains_without_editing():
    clean = ('A person in a blue shirt stands in a bright hallway and lifts both arms. '
             "The camera slowly zooms in while the person's hands open outward.")
    assert ls.text_integrity(clean) == clean
    assert ls.text_integrity('It’s a quiet room.') == 'It’s a quiet room.'
    for bad in ('A small glass container ofmintrestsleftoftheglass.',
                'A cake covered with a thick layer厚层 of frosting.',
                'He stands at-a-slight-angle near the door.'):
        with pytest.raises(ls.IntegrityError):
            ls.text_integrity(bad)


def test_short_glued_tokens_are_a_documented_limit_not_claimed_detected():
    assert ls.text_integrity('A man sits on abench.') == 'A man sits on abench.'


def test_prompts_extend_original_007_and_008_text_exactly():
    r, tl, _ = ls.modules()
    cap = ls.caption_prompt()
    assert cap.startswith(tl.CAPTION_PROMPT[:-len('Grounded facts:\n')]) and cap.endswith('Grounded facts:\n')
    assert ls.CAMERA_CAPTION in cap and 'frame' not in ls.CAMERA_CAPTION.lower()
    facts = ls.facts_prompt()
    assert facts == r.FACTS_PROMPT_V2 + tl.ACTOR_PROMPT + ls.CAMERA_PROMPT


def test_camera_facts_require_enum_and_keep_original_validator():
    base = dict(scene='A kitchen table.', rendered_or_screen=False,
                entities=[dict(id='E1', description='A white plate', held_object=None)], events=[],
                start_end_difference='No visible change.')
    assert ls.validate_camera_facts(dict(base, camera_motion='zoom_in_or_push_in'))['camera_motion'] == 'zoom_in_or_push_in'
    for bad in (dict(base), dict(base, camera_motion='dolly_zoom')):
        with pytest.raises(ls.IntegrityError):
            ls.validate_camera_facts(bad)
    with pytest.raises(ValueError):
        ls.validate_camera_facts(dict(base, camera_motion='static', extra='x'))


def test_config_has_no_word_regex_and_records_camera_schema():
    c = ls.config()
    assert c['caption_word_count_grammar'] is False and 'pattern' not in json.dumps(c['caption_schema'])
    assert 'camera_motion' in c['facts_schema']['required'] and c['stop_failure_rate'] == 0.10
    assert c == ls.config()


def test_stop_rule_matches_approved_failure_cap():
    ok, bad = dict(success=True), dict(success=False)
    assert ls.stop_decision([bad] * 3 + [ok] * 17, 1856) is None
    assert ls.stop_decision([bad] * 12 + [ok] * 116, 1856) is None
    assert ls.stop_decision([bad] * 13 + [ok] * 115, 1856) == 'failure_rate_exceeded'
    assert ls.stop_decision([ok, dict(success=False, decoder_stop=True)], 1856) == 'decoder_admission_failed_stop_without_retry'
    assert ls.stop_decision([bad] * 4 + [ok] * 28, 32) == 'failure_rate_exceeded'
    assert ls.pilot_pass([bad] * 3 + [ok] * 29, 32) and not ls.pilot_pass([bad] * 4 + [ok] * 28, 32)
    assert not ls.pilot_pass([ok] * 31, 32)


def test_claims_are_exclusive(tmp_path):
    (tmp_path / 'claims').mkdir()
    assert ls.claim(tmp_path, 7) and not ls.claim(tmp_path, 7) and ls.claim(tmp_path, 8)


@pytest.mark.skipif(not ls.POOL.is_file(), reason='private pool manifest not published')
def test_pilot_selection_is_seeded_and_disjoint():
    old = [r['video'] for r in ls.pool_rows()[:32]]
    a = ls.select_pilot(20261121, old); b = ls.select_pilot(20261121, old)
    assert a == b and len(a) == 32 and len({w['sample_id'] for w in a}) == 32
    assert not {w['video'] for w in a} & set(old)
    rest = ls.scale_windows([w['sample_id'] for w in a])
    assert len(rest) == 1856 - 32 and not {w['sample_id'] for w in rest} & {w['sample_id'] for w in a}


def test_summary_reports_exclusions_by_dataset_and_camera():
    outs = [dict(dataset='hoigen1m', success=True, facts_json=dict(camera_motion='static')),
            dict(dataset='hoigen1m', success=False, error_type='DenyMatch', failure_reason='x'),
            dict(dataset='openvidgen', success=True, facts_json=dict(camera_motion='pan_left'))]
    s = ls.summarize(outs, dict(jobs=[0, 1, 2], mode='pilot'))
    assert s['by_dataset']['hoigen1m'] == dict(windows=2, labels=1, excluded=1)
    assert s['failure_reasons'] == {'x': 1} and s['camera_motion_counts'] == {'static': 1, 'pan_left': 1}
    assert s['pilot_failure_rule_pass'] is False  # 1/3 failures exceeds the 10% cap


def test_009b_caption_word_range_and_compact_facts_arrays():
    c = ls.config()
    assert c['caption_words'] == [40, 150] and c['token_limit_without_EOS'] == 'window_excluded'
    assert c['facts_schema']['properties']['entities']['maxItems'] == 8
    assert c['facts_schema']['properties']['events']['maxItems'] == 12
    sentence = 'The person in a blue shirt slowly lifts a white cup from the wooden table and drinks.'
    ok = json.dumps([sentence] * 4)
    assert len(ls.parse_caption(ok).split()) == 68
    long = json.dumps([sentence + ' ' + ' '.join(['then the person keeps looking ahead calmly'] * 4)] * 4)
    with pytest.raises(ls.IntegrityError):
        ls.parse_caption(long)
    mid = json.dumps([sentence + ' Then the person sets the white cup down and smiles at the table again.'] * 4)
    assert 120 < len(ls.parse_caption(mid).split()) <= 150
    with pytest.raises(ValueError):
        ls.parse_caption(json.dumps([sentence + ' The frame shows a cup.'] * 4))


def test_only_token_limit_without_eos_is_excluded_not_stopped(monkeypatch):
    r, tl, _ = ls.modules()
    job = dict(index=0, sample_id='s', dataset='d', video=dict(path='/x', bytes=1, sha256='0' * 64))
    monkeypatch.setattr(r, 'check', lambda *a, **k: None)
    monkeypatch.setattr(r, 'sample_images', lambda *a, **k: ([], dict(source_indices=list(range(0, 121, 8)))))
    def runaway(model, processor, images, prompt, limit, trace):
        trace.update(new_tokens=limit, ended_with_EOS=False); raise AssertionError()
    for flag in ('SCHEMA_DECODE', 'BOUNDARY_DECODE', 'BOUNDED_DECODE'):
        monkeypatch.setattr(tl, flag, True)
    monkeypatch.setattr(tl, 'generate', runaway)
    v = ls.label_one(None, None, job, {}, None)
    assert not v['success'] and v['failure_reason'] == ls.LENGTH_REASON and v['decoder_stop'] is False
    def broken(model, processor, images, prompt, limit, trace):
        trace.update(new_tokens=5, ended_with_EOS=True); raise AssertionError()
    monkeypatch.setattr(tl, 'generate', broken)
    v = ls.label_one(None, None, job, {}, None)
    assert v['decoder_stop'] is True

