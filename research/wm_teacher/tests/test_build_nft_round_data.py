import hashlib
import json
import statistics as st
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import build_nft_round_data as b  # noqa: E402


class SerialPool:
    def __init__(self, *a):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def map(self, fn, items, chunksize=1):
        return [fn(x) for x in items]


class AdvantageTests(unittest.TestCase):
    def test_group_of(self):
        self.assertEqual(b.group_of('t2c_abc_s101'), 't2c_abc')
        with self.assertRaises(ValueError):
            b.group_of('t2c_abc')

    def test_per_prompt_mean_global_std(self):
        rows = [dict(group=g, physics=p, mq=m, motion=o) for g, p, m, o in
                [('a', 1, 0, 0), ('a', 2, 1, 1), ('a', 3, 0, 2), ('b', 0, 5, 1), ('b', 1, 1, 0), ('b', 9, 2, 2)]]
        b.advantages(rows, dict(physics=1.0, mq=0.0, motion=0.0))
        for g in 'ab':
            self.assertAlmostEqual(sum(r['advantage'] for r in rows if r['group'] == g), 0.0, places=9)
        std_all = st.pstdev([r['reward'] for r in rows]) + 1e-4
        ra = [r for r in rows if r['group'] == 'a']
        self.assertAlmostEqual(ra[2]['advantage'], (ra[2]['reward'] - st.mean(x['reward'] for x in ra)) / std_all)
        self.assertAlmostEqual(st.mean(r['z_physics'] for r in rows), 0.0, places=9)


class BuildTests(unittest.TestCase):
    def write(self, root, n_groups=2, seeds=(101, 102, 103), tamper=None):
        seeds_rows, jobs = [], []
        (root / 'rw').mkdir()
        for k in range(n_groups):
            video = f'/data/gt_{k}.mp4'
            key = hashlib.sha256(video.encode()).hexdigest()[:16]
            seeds_rows.append(dict(video=video, prompt=f'p{k}', parent=f'ds:{k}', dataset='ds', diff_median=1.0))
            for s in seeds:
                jid = f't2c_{key}_s{s}'
                jobs.append(dict(id=jid, image='x.png', prompt=f'p{k}', out=f'/gen/{jid}.mp4', seed=s, ckpt='NONE', model='m'))
                rec = dict(id=jid, video=f'/gen/{jid}.mp4', ok=True, vjepa_surprise=0.4 + 0.01 * s % 7 + k * 0.001,
                           vr_MQ=float(s % 5) - k, motion=float(s % 3) + 1)
                if tamper == 'failed' and s == seeds[0] and k == 0:
                    rec['ok'] = False
                (root / 'rw' / (jid + '.json')).write_text(json.dumps(rec))
        if tamper == 'prompt':
            jobs[0]['prompt'] = 'other'
        (root / 'seeds.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in seeds_rows))
        (root / 'jobs.jsonl').write_text(''.join(json.dumps(j) + '\n' for j in jobs))
        return types.SimpleNamespace(jobs=str(root / 'jobs.jsonl'), seeds=str(root / 'seeds.jsonl'),
                                     rewards_dir=str(root / 'rw'), out_dir=str(root / 'out'), w_physics=1.0,
                                     w_mq=1.0, w_motion=1.0, workers=1)

    def test_build_outputs_and_traceability(self):
        fake = lambda pair: 0.1 * (int(pair[0].split('_s')[-1].split('.')[0]) % 4)  # noqa: E731
        with tempfile.TemporaryDirectory() as d, patch.object(b, 'Pool', SerialPool), patch.object(b, 'motion_error', fake):
            root = Path(d)
            s = b.build(self.write(root))
            self.assertEqual((s['n'], s['groups'], s['group_sizes']), (6, 2, [3]))
            self.assertFalse(s['eval_inputs_used'])
            rows = [json.loads(l) for l in (root / 'out/samples.jsonl').read_text().splitlines()]
            man = [json.loads(l) for l in (root / 'out/cache_manifest.jsonl').read_text().splitlines()]
            self.assertEqual([r['sample_id'] for r in rows], [m['sample_id'] for m in man])
            self.assertTrue(all(set(m['captions']) == {'nft'} for m in man))
            self.assertTrue(all(isinstance(r['advantage'], float) for r in rows))
            again = self.write(Path(tempfile.mkdtemp(dir=d)))
            again.out_dir = str(root / 'out')  # never overwrite an existing round directory
            with self.assertRaises(FileExistsError):
                b.build(again)

    def test_refuses_failed_reward_or_untraceable_prompt(self):
        fake = lambda pair: 0.1  # noqa: E731
        for tamper in ('failed', 'prompt'):
            with tempfile.TemporaryDirectory() as d, patch.object(b, 'Pool', SerialPool), patch.object(b, 'motion_error', fake):
                with self.subTest(tamper=tamper), self.assertRaises(ValueError):
                    b.build(self.write(Path(d), tamper=tamper))

    def test_refuses_singleton_groups(self):
        with tempfile.TemporaryDirectory() as d, patch.object(b, 'Pool', SerialPool), patch.object(b, 'motion_error', lambda p: 0.1):
            with self.assertRaises(ValueError):
                b.build(self.write(Path(d), seeds=(101,)))


if __name__ == '__main__':
    unittest.main()
