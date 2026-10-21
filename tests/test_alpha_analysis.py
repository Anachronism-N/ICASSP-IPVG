import copy
import importlib.util
import os
from pathlib import Path
import pytest
pytest.importorskip('numpy')
path = Path(os.environ.get('IPVG_ALPHA_ANALYZER', str(Path(__file__).resolve().parents[1] / 'scripts/analyze_lora_alpha.py')))
spec = importlib.util.spec_from_file_location('alpha_analysis', path)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

T = {'gain_retention': .6, 'gray_motion_min': .8, 'mouth_rate_drop_max': .05,
     'mouth_paircoverage_min': .95, 'mouth_flow_min': .8, 'imaging_drop_max': .01, 'clip_drop_max': .005,
     'repeat_identity_max': .02, 'repeat_imaging_max': .01, 'repeat_clip_max': .005,
     'repeat_motion_min': .9, 'repeat_motion_max': 1.1}


def identity():
    arm = {'identity_coverage': 1, 'videos_at_least_seven_frames': 16, 'valid_video_identity_mean': .6}
    return {'narrow': dict(arm), 'wide': dict(arm), 'paired_actors': 8, 'paired_videos': 16, 'identity_actor_mean_delta': .3,
            'identity_actor_bootstrap_ci95': [.1, .4], 'identity_p10_actor_delta': .1, 'motion_ratio': .8}


def mouth():
    arm = {'activity_video_rate_all': 1, 'mean_pair_coverage': 1}
    return {'single': dict(arm), 'dual': dict(arm), 'paired_flow_ratio': .8}


def test_preregistered_candidate_boundaries():
    result = module.candidate_checks(identity(), mouth(), {'imaging_quality': -.01, 'clip_frame_text_cosine_mean': -.005}, .5, T)
    assert all(result.values())


@pytest.mark.parametrize('failure', ['gain', 'motion', 'active', 'coverage', 'ci', 'p10', 'paircoverage', 'flow', 'fullgain', 'imaging', 'clip'])
def test_each_quality_guard_rejects_real_tradeoff(failure):
    a, m = identity(), mouth()
    delta = {'imaging_quality': 0, 'clip_frame_text_cosine_mean': 0}
    full = .5
    if failure == 'gain': a['identity_actor_mean_delta'] = .299
    elif failure == 'motion': a['motion_ratio'] = .799
    elif failure == 'active': m['dual']['activity_video_rate_all'] = 15 / 16
    elif failure == 'coverage': a['wide']['identity_coverage'] = .99
    elif failure == 'ci': a['identity_actor_bootstrap_ci95'][0] = 0
    elif failure == 'p10': a['identity_p10_actor_delta'] = -.001
    elif failure == 'paircoverage': m['dual']['mean_pair_coverage'] = .94
    elif failure == 'flow': m['paired_flow_ratio'] = None
    elif failure == 'fullgain': full = 0
    elif failure == 'imaging': delta['imaging_quality'] = -.011
    elif failure == 'clip': delta['clip_frame_text_cosine_mean'] = -.006
    assert not all(module.candidate_checks(a, m, delta, full, T).values())


def test_repeat_consistency_gate_and_exact_hashes():
    pair = identity(); pair['motion_ratio'] = 1
    q = {'imaging_quality': .7, 'clip_frame_text_cosine_mean': .3}
    assert module.repeat_checks(pair, q, q, 0, 16, T)['passed']
    drift = dict(q, imaging_quality=.8)
    assert not module.repeat_checks(pair, q, drift, 15, 16, T)['passed']
    assert module.repeat_checks(pair, q, drift, 16, 16, T)['passed']
    pair['wide']['identity_coverage'] = .9
    assert not module.repeat_checks(pair, q, q, 0, 16, T)['passed']
