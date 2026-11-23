import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import teacher_paired_dev_summary as t  # noqa: E402


def test_cluster_bootstrap_keeps_seeds_together_and_detects_sign():
    pos = [{i: 0.01 + 0.001 * (i % 3) for i in range(50)}, {i: 0.012 for i in range(50)}]
    r = t.cluster_bootstrap(pos)
    assert r['n_cases'] == 50 and r['seeds'] == 2 and r['ci95'][0] > 0 and abs(r['mean'] - 0.0115) < 1e-3
    mixed = [{i: (0.01 if i % 2 else -0.01) for i in range(50)}, {i: 0.0 for i in range(50)}]
    r = t.cluster_bootstrap(mixed)
    assert r['ci95'][0] < 0 < r['ci95'][1]
