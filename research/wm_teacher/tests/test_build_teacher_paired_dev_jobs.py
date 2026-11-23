import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import build_teacher_paired_dev_jobs as b  # noqa: E402


def test_paired_jobs_cover_94_cases_two_seeds_and_match_scorer_layout(tmp_path):
    if not (b.E1 / 'cases.jsonl').is_file():
        return
    out = b.build(tmp_path / 'dev', [('base', 'NONE')], [0, 1])
    jobs = [json.loads(l) for l in (tmp_path / 'dev/jobs.jsonl').read_text().splitlines()]
    assert out['jobs'] == len(jobs) == 188 and {j['seed'] for j in jobs} == {0, 1}
    cases = [json.loads(l) for l in (tmp_path / 'dev/cases.jsonl').read_text().splitlines()]
    assert all(j['out'].endswith(f"videos/base_s{j['seed']}/" + j['id'].split('__base__')[0] + '.mp4') for j in jobs)
    assert {c['id'] for c in cases} == {j['id'].split('__base__')[0] for j in jobs}


def test_append_model_reuses_frozen_cases_and_never_overwrites(tmp_path):
    import pytest
    if not (b.E1 / 'cases.jsonl').is_file():
        return
    b.build(tmp_path / 'dev', [('base', 'NONE')], [0, 1])
    out = b.add_model(tmp_path / 'dev', 'T1', 'NONE', [0, 1])
    jobs = [json.loads(l) for l in (tmp_path / 'dev/jobs_T1.jsonl').read_text().splitlines()]
    assert out['jobs'] == len(jobs) == 188 and all('/videos/T1_s' in j['out'] for j in jobs)
    with pytest.raises(FileExistsError):
        b.add_model(tmp_path / 'dev', 'T1', 'NONE', [0, 1])
