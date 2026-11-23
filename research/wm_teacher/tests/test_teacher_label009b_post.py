import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import teacher_label009b_post as post  # noqa: E402


def _plan(tmp_path, outs):
    run = tmp_path / 'run01'; (run / 'labels').mkdir(parents=True)
    jobs = []
    for i, (ds, o) in enumerate(outs):
        jobs.append(dict(index=i, sample_id=f's{i}', dataset=ds, video=dict(path=f'/v{i}.mp4', bytes=1, sha256='0' * 64)))
        if o is not None:
            (run / 'labels' / f'{i}.json').write_text(json.dumps(dict(o, index=i, sample_id=f's{i}', dataset=ds, video=dict(path=f'/v{i}.mp4'))))
    p = tmp_path / 'plan.json'; p.write_text(json.dumps(dict(output_root=str(run), jobs=jobs)))
    return p


def test_summary_counts_excluded_and_unattempted(tmp_path):
    cap = 'A person lifts a white cup from the table and drinks slowly. ' * 6
    p = _plan(tmp_path, [('a', dict(success=True, caption=cap, facts_json=dict(camera_motion='static'))),
                         ('a', dict(success=False, failure_reason='x')), ('b', None)])
    s = post.summary(p)
    assert s['labels'] == 1 and s['excluded'] == 1 and s['not_attempted'] == 1
    assert s['by_dataset']['a'] == dict(windows=2, labels=1, excluded=1, not_attempted=0)
    assert s['failure_reasons'] == {'x': 1} and s['camera_motion'] == {'static': 1}
