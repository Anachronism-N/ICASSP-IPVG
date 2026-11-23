import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import teacher_dev_generate as g  # noqa: E402


def test_negative_prompt_and_sampler_match_gen_jobs_defaults():
    src = (Path(__file__).resolve().parents[1] / 'scripts/gen_jobs.py').read_text()
    assert g.negative_prompt() in src.replace('"\n       "', '')
    assert g.SAMPLER == dict(cfg_scale=5.0, sigma_shift=5.0, num_inference_steps=50, tiled=True)
    assert (g.W, g.H, g.NUM_FRAMES, g.FPS) == (1280, 704, 121, 24)


def test_job_schema_and_claims(tmp_path):
    j = dict(id='a', model='base', ckpt='NONE', image='/x.png', prompt='p', out=str(tmp_path / 'a.mp4'), seed=0)
    p = tmp_path / 'jobs.jsonl'; p.write_text(json.dumps(j) + '\n')
    assert g.load_jobs(p)[0]['id'] == 'a'
    p.write_text(json.dumps(dict(j, extra=1)) + '\n')
    with pytest.raises(AssertionError):
        g.load_jobs(p)
    assert g.claim(j['out']) and not g.claim(j['out'])


def test_stale_claims_of_dead_owners_are_cleared_live_and_finished_kept(tmp_path):
    import os
    mk = lambda n: dict(id=n, model='base', ckpt='NONE', image='/x.png', prompt='p', out=str(tmp_path / (n + '.mp4')), seed=0)
    dead, live, done = mk('dead'), mk('live'), mk('done')
    open(dead['out'] + '.claim', 'w').write('999999999')
    open(live['out'] + '.claim', 'w').write(str(os.getpid()))
    open(done['out'] + '.claim', 'w').write('999999999'); open(done['out'], 'w').write('v')
    assert g.clear_stale_claims([dead, live, done]) == ['dead']
    assert os.path.exists(live['out'] + '.claim') and os.path.exists(done['out'] + '.claim')


def test_foreign_host_claims_are_never_cleared(tmp_path):
    j = dict(id='x', model='base', ckpt='NONE', image='/x.png', prompt='p', out=str(tmp_path / 'x.mp4'), seed=0)
    open(j['out'] + '.claim', 'w').write('other-node:999999999')
    assert g.clear_stale_claims([j]) == [] and (tmp_path / 'x.mp4.claim').exists()
    assert g.claim(str(tmp_path / 'y.mp4')) and open(str(tmp_path / 'y.mp4.claim')).read().startswith(__import__('os').uname().nodename + ':')
