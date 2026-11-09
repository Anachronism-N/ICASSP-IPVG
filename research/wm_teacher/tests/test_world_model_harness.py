import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import world_model_harness as h


@pytest.fixture
def contract():
    return dict(schema='world_model_evaluation_contract_v1',
        protocol=dict(prompt_variant='raw', input_mode='I2V', samples_per_pair=1,
                      selection='single', checkpoint_sha256='a'*64, scorer_sha256='b'*64,
                      inputs_sha256='c'*64, media=dict(width=16, height=16, frames=2, fps=24),
                      sampler=dict(steps=50, cfg=5.0, shift=5.0)),
        jobs=[dict(id='x', pair='p', sample=0, seed=0, output='x.mp4')],
        expected_scores=dict(quality=['x']))


def test_comparison_different_checkpoint_but_identical_experiment(contract):
    other = copy.deepcopy(contract); other['protocol']['checkpoint_sha256'] = 'd'*64
    assert h.comparable_protocols(contract, other)
    for key, value in [('prompt_variant', 'po'), ('input_mode', 'V2V')]:
        changed = copy.deepcopy(other); changed['protocol'][key] = value
        assert not h.comparable_protocols(contract, changed)
    changed = copy.deepcopy(other); changed['protocol']['sampler']['cfg'] = 3
    assert not h.comparable_protocols(contract, changed)
    changed = copy.deepcopy(other); changed['jobs'][0]['seed'] = 1
    assert not h.comparable_protocols(contract, changed)


@pytest.mark.parametrize('mutation', [
    lambda c: c['jobs'].append(dict(c['jobs'][0])),
    lambda c: c['protocol'].update(samples_per_pair=5),
    lambda c: c['protocol'].update(selection='unknown'),
    lambda c: c['protocol']['sampler'].update(cfg=float('nan')),
    lambda c: c['protocol'].pop('sampler'),
    lambda c: c['protocol']['media'].update(frames=True),
    lambda c: c['expected_scores'].update(quality=['missing']),
    lambda c: c['expected_scores'].update(quality=['x', 'x']),
    lambda c: c['jobs'][0].update(sample=True),
    lambda c: c['jobs'][0].update(output='../bad.mp4'),
    lambda c: c['jobs'][0].update(seed=True),
])
def test_refuse_incomplete_or_invalid_protocol(contract, mutation):
    mutation(contract)
    with pytest.raises(ValueError): h.validate_evaluation_contract(contract)


def test_all_samples_protocol_keeps_every_sample(contract):
    contract['protocol'].update(samples_per_pair=5, selection='all_samples')
    contract['jobs'] = [dict(id=str(i), pair='p', sample=i, seed=i, output=f'{i}.mp4') for i in range(5)]
    contract['expected_scores'] = dict(quality=[str(i) for i in range(5)])
    h.validate_evaluation_contract(contract)
    contract['jobs'][4]['sample'] = 3
    with pytest.raises(ValueError): h.validate_evaluation_contract(contract)


@pytest.fixture
def artifacts(contract, tmp_path):
    video = tmp_path / 'x.mp4'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi', '-i',
                    'color=c=red:s=16x16:r=24', '-frames:v', '2', '-an', str(video)],
                   check=True, capture_output=True, timeout=30)
    sha = h.validate_evaluation_contract(contract)
    videos = [dict(id='x', protocol_sha256=sha, full_decode_pass=True, file=h.file_record(video))]
    score = tmp_path / 'scores.json'
    payload = dict(protocol_sha256=sha, dimension='quality',
                   video_bytes_sha256=h.digest({'x': videos[0]['file']['sha256']}),
                   rows=[dict(id='x', value=0.5)])
    score.write_text(json.dumps(payload))
    scores = dict(quality=dict(protocol_sha256=sha, file=h.file_record(score)))
    return videos, scores, payload


def test_actual_CPU_video_decode_and_complete_bound_score_file(contract, artifacts):
    videos, scores, _ = artifacts
    result = h.audit_evaluation(contract, videos, scores)
    assert result['declared_coverage_verified'] and result['videos_verified'] == 1
    assert not any(result[key] for key in ('official_coverage_proven', 'metric_correctness_proven',
                                         'model_quality_measured', 'sota_proven'))


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'decode', 'protocol', 'byte_drift', 'media'])
def test_bad_video_or_media_is_not_accepted(contract, artifacts, change):
    videos, scores, _ = artifacts
    if change == 'missing': videos.clear()
    elif change == 'duplicate': videos.append(dict(videos[0]))
    elif change == 'decode': videos[0]['full_decode_pass'] = False
    elif change == 'protocol': videos[0]['protocol_sha256'] = 'f'*64
    elif change == 'byte_drift': Path(videos[0]['file']['path']).write_bytes(b'broken')
    else:
        spec = dict(contract['protocol']['media'], frames=3)
        with pytest.raises(ValueError): h.validate_video(videos[0]['file']['path'], spec)
        return
    with pytest.raises(ValueError): h.audit_evaluation(contract, videos, scores)


@pytest.mark.parametrize('change', ['missing_dim', 'file_drift', 'wrong_video', 'nan', 'duplicate_rows', 'bool'])
def test_bad_scores_are_rejected_even_with_new_file_binding(contract, artifacts, change):
    videos, scores, payload = artifacts
    path = Path(scores['quality']['file']['path'])
    if change == 'missing_dim': scores.clear()
    elif change == 'file_drift': path.write_text('{}')
    else:
        if change == 'wrong_video': payload['video_bytes_sha256'] = 'd'*64
        elif change == 'nan': payload['rows'][0]['value'] = float('nan')
        elif change == 'bool': payload['rows'][0]['value'] = True
        else: payload['rows'].append(dict(payload['rows'][0]))
        path.write_text(json.dumps(payload)); scores['quality']['file'] = h.file_record(path)
    with pytest.raises(ValueError): h.audit_evaluation(contract, videos, scores)


def test_symlink_is_not_an_accepted_file_record(tmp_path):
    original = tmp_path / 'original'; original.write_text('safe')
    link = tmp_path / 'link'; link.symlink_to(original)
    with pytest.raises(ValueError): h.file_record(link)


def test_CPU_import_does_not_load_torch_and_CLI_does_not_authorize_GPU(contract, tmp_path):
    code = 'import runpy,sys;runpy.run_path(' + repr(str(Path(h.__file__))) + ');assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True)
    p = tmp_path/'contract.json'; p.write_text(json.dumps(contract))
    r = subprocess.run([sys.executable, h.__file__, 'validate', '--contract', str(p)], check=True, capture_output=True)
    assert json.loads(r.stdout)['execution_authorized'] is False
