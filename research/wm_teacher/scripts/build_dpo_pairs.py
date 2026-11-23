#!/usr/bin/env python3
"""Build Flow-DPO pairs from reward scores (CPU). Training-pool candidates only.

Per prompt with >=2 scored candidates: score = z(-vjepa_surprise) + z(vr_MQ) (z over all scored candidates);
winner = argmax, loser = argmin; keep the pair only if score gap >= min_gap and
motion(winner) >= motion_floor * motion(loser) (prevents the physics reward from collapsing into 'no motion').
Writes pairs.jsonl and a cache manifest (one sample per used candidate video, caption = the shared prompt).
"""
import argparse
import hashlib
import json
import statistics as st
from collections import defaultdict
from pathlib import Path


def zscores(vals):
    m, s = st.mean(vals), st.pstdev(vals) or 1.0
    return [(v - m) / s for v in vals]


def build(scores, jobs, min_gap, motion_floor):
    ok = [s for s in scores if s.get('ok')]
    za = zscores([-s['vjepa_surprise'] for s in ok]); zb = zscores([s['vr_MQ'] for s in ok])
    comb = {s['id']: a + b for s, a, b in zip(ok, za, zb)}
    byid = {s['id']: s for s in ok}
    groups = defaultdict(list)
    for j in jobs:
        if j['id'] in comb:
            groups[(j['image'], j['prompt'])].append(j)
    pairs, rejected = [], defaultdict(int)
    for (image, prompt), js in groups.items():
        if len(js) < 2:
            rejected['too_few_candidates'] += 1; continue
        js = sorted(js, key=lambda j: comb[j['id']])
        lo, hi = js[0], js[-1]
        if comb[hi['id']] - comb[lo['id']] < min_gap:
            rejected['small_gap'] += 1; continue
        if byid[hi['id']]['motion'] < motion_floor * byid[lo['id']]['motion']:
            rejected['winner_too_static'] += 1; continue
        pairs.append(dict(winner=hi['id'], loser=lo['id'], prompt=prompt, image=image, gap=comb[hi['id']] - comb[lo['id']],
                          winner_video=hi['out'], loser_video=lo['out'],
                          winner_scores=byid[hi['id']], loser_scores=byid[lo['id']]))
    return pairs, dict(rejected), comb


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--jobs', required=True); ap.add_argument('--scores-dir', required=True); ap.add_argument('--out-dir', required=True)
    ap.add_argument('--min-gap', type=float, default=0.5); ap.add_argument('--motion-floor', type=float, default=0.8)
    a = ap.parse_args()
    jobs = [json.loads(l) for l in Path(a.jobs).read_text().splitlines() if l.strip()]
    scores = [json.loads(p.read_text()) for p in Path(a.scores_dir).glob('*.json')]
    pairs, rej, comb = build(scores, jobs, a.min_gap, a.motion_floor)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=False)
    with open(out / 'pairs.jsonl', 'x') as f:
        f.write(''.join(json.dumps(p, sort_keys=True) + '\n' for p in pairs))
    used = {}
    for p in pairs:
        used[p['winner']] = (p['winner_video'], p['prompt']); used[p['loser']] = (p['loser_video'], p['prompt'])
    with open(out / 'cache_manifest.jsonl', 'x') as f:
        for sid, (v, pr) in sorted(used.items()):
            f.write(json.dumps(dict(sample_id=sid, video=v, captions=dict(dpo=pr))) + '\n')
    w = [p['winner_scores'] for p in pairs]; l = [p['loser_scores'] for p in pairs]
    summ = dict(scored=sum(1 for s in scores if s.get('ok')), failed=sum(1 for s in scores if not s.get('ok')), pairs=len(pairs),
                rejected=rej, min_gap=a.min_gap, motion_floor=a.motion_floor,
                mean_winner=dict(vjepa=st.mean(x['vjepa_surprise'] for x in w), MQ=st.mean(x['vr_MQ'] for x in w), motion=st.mean(x['motion'] for x in w)) if w else None,
                mean_loser=dict(vjepa=st.mean(x['vjepa_surprise'] for x in l), MQ=st.mean(x['vr_MQ'] for x in l), motion=st.mean(x['motion'] for x in l)) if l else None,
                pairs_sha256=hashlib.sha256((out / 'pairs.jsonl').read_bytes()).hexdigest())
    (out / 'summary.json').write_text(json.dumps(summ, indent=1)); print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
