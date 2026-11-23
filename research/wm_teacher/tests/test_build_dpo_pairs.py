import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import build_dpo_pairs as b  # noqa: E402


def test_pairs_use_combined_reward_and_motion_floor():
    jobs = [dict(id=f'p{p}_s{s}', image=f'img{p}', prompt=f'pr{p}', out=f'/v/p{p}_s{s}.mp4') for p in (1, 2) for s in (1, 2, 3)]
    sc = {'p1_s1': (0.30, 2.0, 5.0), 'p1_s2': (0.20, 1.0, 5.0), 'p1_s3': (0.10, 3.0, 5.0),   # s3 best (low surprise, high MQ)
          'p2_s1': (0.05, 0.0, 0.5), 'p2_s2': (0.40, -1.0, 6.0), 'p2_s3': (0.30, 0.0, 6.0)}   # p2 winner nearly static
    scores = [dict(id=k, ok=True, vjepa_surprise=v[0], vr_MQ=v[1], motion=v[2]) for k, v in sc.items()]
    pairs, rej, _ = b.build(scores, jobs, 0.5, 0.8)
    assert [(p['winner'], p['loser']) for p in pairs] == [('p1_s3', 'p1_s1')]
    assert rej == {'winner_too_static': 1}
