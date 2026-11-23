#!/usr/bin/env python3
"""DiffusionNFT round data (CPU): rewards -> per-prompt advantages + cache manifest, training data only.

Per generated candidate (training-pool first frame + caption, never evaluation inputs):
  physics   = -V-JEPA2 surprise (official WMReward score, lower surprise = better)
  mq        = VideoReward motion quality (published normalisation)
  motion    = -|log(motion_gen / motion_GT)|, the same estimator as the dev-set guard metric
              (gt_ref_metrics: K=12 matched time points, 256x144, 64x36 gray frame differences) against the real
              continuation of the same training window. Counteracts the static bias of physics/mq.
Each component is z-scored over the round pool; R = sum_k w_k z_k. Advantages follow the official
PerPromptStatTracker with global std (DiffusionNFT config.base: sample.global_std=True):
  A = (R - mean_group(R)) / (std_all(R) + 1e-4)
Outputs: samples.jsonl, cache_manifest.jsonl (caption version 'nft'), summary.json.
"""
import argparse
import hashlib
import json
import math
from multiprocessing import Pool
import os
from pathlib import Path
import statistics as st
import sys

SCRIPTS = Path(__file__).resolve().parent
COMPONENTS = ('physics', 'mq', 'motion')


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def group_of(sample_id):
    head, _, seed = sample_id.rpartition('_s')
    require(head and seed.isdigit(), 'Candidate id must end with _s<seed>: ' + sample_id)
    return head


def seed_key(video):
    return hashlib.sha256(video.encode()).hexdigest()[:16]


def motion_error(pair):
    gen, gt = pair
    sys.path.insert(0, str(SCRIPTS))
    import gt_ref_metrics as g
    gf, gfps = g.read(gt)
    vf, vfps = g.read(gen)
    dur = min(len(gf) / gfps, len(vf) / vfps) - 1e-3
    a = [x.resize((256, 144)) for x in g.sample(gf, gfps, dur)]
    b = [x.resize((256, 144)) for x in g.sample(vf, vfps, dur)]
    return abs(math.log(g.motion(b) / g.motion(a)))


def zscores(values):
    m, s = st.mean(values), st.pstdev(values)
    require(s > 0, 'Degenerate reward component')
    return [(v - m) / s for v in values]


def corr(a, b):
    ma, mb, sa, sb = st.mean(a), st.mean(b), st.pstdev(a), st.pstdev(b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / len(a) / sa / sb


def advantages(rows, weights):
    for k in COMPONENTS:
        for r, z in zip(rows, zscores([r[k] for r in rows])):
            r['z_' + k] = z
    for r in rows:
        r['reward'] = sum(weights[k] * r['z_' + k] for k in COMPONENTS)
    std_all = st.pstdev([r['reward'] for r in rows]) + 1e-4
    groups = {}
    for r in rows:
        groups.setdefault(r['group'], []).append(r['reward'])
    for r in rows:
        r['advantage'] = (r['reward'] - st.mean(groups[r['group']])) / std_all
    return rows


def build(a):
    jobs = [json.loads(l) for l in Path(a.jobs).read_text().splitlines() if l.strip()]
    seeds = {seed_key(s['video']): s for s in (json.loads(l) for l in Path(a.seeds).read_text().splitlines() if l.strip())}
    rows = []
    for j in jobs:
        rec = json.loads((Path(a.rewards_dir) / (j['id'] + '.json')).read_text())
        require(rec.get('ok') is True and rec['id'] == j['id'] and rec['video'] == j['out'], 'Reward missing/failed: ' + j['id'])
        grp = group_of(j['id'])
        key = grp.split('_', 1)[1]
        require(key in seeds and seeds[key]['prompt'] == j['prompt'], 'Candidate not traceable to its training seed: ' + j['id'])
        rows.append(dict(sample_id=j['id'], group=grp, video=j['out'], prompt=j['prompt'], gt=seeds[key]['video'],
                         parent=seeds[key]['parent'], dataset=seeds[key]['dataset'], seed=j['seed'],
                         vjepa_surprise=rec['vjepa_surprise'], vr_MQ=rec['vr_MQ'], motion_magnitude=rec['motion']))
    require(rows and len({r['sample_id'] for r in rows}) == len(rows), 'Empty/duplicate candidates')
    sizes = {}
    for r in rows:
        sizes[r['group']] = sizes.get(r['group'], 0) + 1
    require(min(sizes.values()) >= 2, 'Every prompt group needs >= 2 candidates')
    with Pool(a.workers) as pool:
        errs = pool.map(motion_error, [(r['video'], r['gt']) for r in rows], chunksize=4)
    for r, e in zip(rows, errs):
        r.update(motion_error=e, physics=-r['vjepa_surprise'], mq=r['vr_MQ'], motion=-e)
    weights = dict(physics=a.w_physics, mq=a.w_mq, motion=a.w_motion)
    advantages(rows, weights)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=False)
    keep = ('sample_id', 'group', 'parent', 'dataset', 'seed', 'vjepa_surprise', 'vr_MQ', 'motion_magnitude',
            'motion_error', 'z_physics', 'z_mq', 'z_motion', 'reward', 'advantage')
    (out / 'samples.jsonl').write_text(''.join(json.dumps({k: r[k] for k in keep}, sort_keys=True) + '\n' for r in rows))
    (out / 'cache_manifest.jsonl').write_text(''.join(
        json.dumps(dict(sample_id=r['sample_id'], video=r['video'], captions={'nft': r['prompt']}), sort_keys=True) + '\n'
        for r in rows))
    adv = [r['advantage'] for r in rows]
    summary = dict(n=len(rows), groups=len(sizes), group_sizes=sorted(set(sizes.values())), weights=weights,
                   advantage=dict(mean=st.mean(adv), std=st.pstdev(adv), min=min(adv), max=max(adv),
                                  frac_abs_gt_5=sum(abs(x) > 5 for x in adv) / len(adv)),
                   corr_advantage=dict(motion_magnitude=corr(adv, [r['motion_magnitude'] for r in rows]),
                                       motion_error=corr(adv, [r['motion_error'] for r in rows]),
                                       vjepa_surprise=corr(adv, [r['vjepa_surprise'] for r in rows]),
                                       vr_MQ=corr(adv, [r['vr_MQ'] for r in rows])),
                   motion_error_mean=st.mean(errs),
                   samples_sha256=hashlib.sha256((out / 'samples.jsonl').read_bytes()).hexdigest(),
                   manifest_sha256=hashlib.sha256((out / 'cache_manifest.jsonl').read_bytes()).hexdigest(),
                   training_data_only=True, eval_inputs_used=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, sort_keys=True))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--jobs', required=True); ap.add_argument('--seeds', required=True)
    ap.add_argument('--rewards-dir', required=True); ap.add_argument('--out-dir', required=True)
    ap.add_argument('--w-physics', type=float, default=1.0); ap.add_argument('--w-mq', type=float, default=1.0)
    ap.add_argument('--w-motion', type=float, default=1.0)
    ap.add_argument('--workers', type=int, default=max(1, min(24, (os.cpu_count() or 2) // 2)))
    print(json.dumps(build(ap.parse_args()), indent=1))


if __name__ == '__main__':
    main()
