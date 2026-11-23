import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import official_wan22_generate as g  # noqa: E402


def test_job_schema_frames_and_host_claims(tmp_path):
    j = dict(id='a', image='/x.jpg', prompt='p', out=str(tmp_path / 'a.mp4'), seed=42, frame_num=121, fps=24, ckpt_dir='/m')
    p = tmp_path / 'jobs.jsonl'; p.write_text(json.dumps(j) + '\n')
    assert g.load_jobs(p)[0]['frame_num'] == 121
    p.write_text(json.dumps(dict(j, frame_num=120)) + '\n')
    with pytest.raises(AssertionError):
        g.load_jobs(p)
    assert g.claim(j['out']) and not g.claim(j['out'])
    open(j['out'] + '.claim', 'w').write('elsewhere:1')
    assert g.clear_stale_claims([j]) == []
    open(j['out'] + '.claim', 'w').write(os.uname().nodename + ':999999999')
    assert g.clear_stale_claims([j]) == ['a']
