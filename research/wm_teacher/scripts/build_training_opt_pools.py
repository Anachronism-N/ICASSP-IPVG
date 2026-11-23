#!/usr/bin/env python3
"""Training-optimization pools (CPU): T3 motion-rich SFT windows and T2 DPO prompt seeds.

Source: dataset/wan22_5s/final_5s_landscape.jsonl (121-frame 1280x704 windows with Qwen3.5 captions).
Isolation: every window whose parent key (prepare_e1_devset.parent_keys, unchanged rule) intersects the E1 dev set or
any E1 exclusion source (WM108, val120, P1/C1/M1, smoke, inference dev) is removed; unresolved parents are removed;
datasets without a parent rule are removed. Public benchmark inputs (VBench, Physics-IQ, WMB) are not in this pool.
Motion: diff_median >= q (pool quantile); no scene cut; per-parent cap; per-dataset cap; deterministic seeded order.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / 'dataset/wan22_5s/final_5s_landscape.jsonl'


def dev_module():
    spec = importlib.util.spec_from_file_location('prepare_e1_devset', ROOT / 'scripts/prepare_e1_devset.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def excluded_keys(dev, easy):
    """Parent keys of the E1 dev set and of every E1 exclusion source (paths mapped through the same parent rule)."""
    keys, unresolved = set(), 0
    for row in (json.loads(l) for l in open(ROOT / 'eval/e1_devset_20261021/cases.jsonl')):
        keys |= set(row['parent_keys'])
    audit = json.loads((ROOT / 'eval/teacher_progress_campaign_20261104/supervised_pool_parent_isolation_CPU.audit.json').read_text())
    sources = {name: files for name, files in audit['exclusion_source_files'].items()}  # migrated copies, audited 2026-11-06
    sources['wm108'] = [str(ROOT / 'eval/wm_bench/cases.jsonl')]
    for name, files in sources.items():
        assert all(os.path.isfile(x) for x in files), name
    for name, paths in dev.load_exclusions({k: v for k, v in sources.items()}).items():
        for p in paths:
            ks, ok = dev.parent_keys(p, easy)
            keys |= set(ks)
            unresolved += not ok
    return keys, unresolved


def build(seed, motion_q, per_parent, dataset_caps, n_sft, n_dpo, exclude_videos):
    dev = dev_module()
    easy = dev.easycom_index() if hasattr(dev, 'easycom_index') else None
    bad, excl_unresolved = excluded_keys(dev, easy)
    rows, stats = [], Counter()
    for line in open(FINAL):
        r = json.loads(line)
        ds = r['source_dataset']
        if ds not in dataset_caps:
            stats['dataset_not_selected'] += 1; continue
        if r.get('scene_cut') or r.get('num_frames') != 121 or not isinstance(r.get('diff_median'), (int, float)):
            stats['quality_signal'] += 1; continue
        rel = r['video_path']
        if rel in exclude_videos:
            stats['already_used'] += 1; continue
        keys, resolved = dev.parent_keys(rel, easy)
        if not resolved or not keys:
            stats['unresolved_parent'] += 1; continue
        if set(keys) & bad:
            stats['eval_parent_overlap'] += 1; continue
        rows.append(dict(video=str(ROOT / rel), rel=rel, dataset=ds, prompt=r['prompt'], diff_median=r['diff_median'],
                         parent=sorted(keys)[0]))
    dm = sorted(r['diff_median'] for r in rows)
    thr = dm[int(motion_q * (len(dm) - 1))] if dm else None
    rng = random.Random(seed)
    rng.shuffle(rows)
    by_parent, by_ds, picked = Counter(), Counter(), []
    for r in sorted(rows, key=lambda x: -x['diff_median']):
        if r['diff_median'] < thr:
            break
        if by_parent[r['parent']] >= per_parent or by_ds[r['dataset']] >= dataset_caps[r['dataset']]:
            continue
        by_parent[r['parent']] += 1; by_ds[r['dataset']] += 1; picked.append(r)
    rng.shuffle(picked)
    # T2 DPO seeds: one window per parent, parents disjoint from T3 SFT; chosen first so SFT never exhausts them
    dpo, dpo_parents = [], set()
    for r in picked:
        if len(dpo) < n_dpo and r['parent'] not in dpo_parents:
            dpo.append(r); dpo_parents.add(r['parent'])
    sft = [r for r in picked if r['parent'] not in dpo_parents][:n_sft]
    return dict(sft=sft, dpo=dpo, threshold=thr, stats=dict(stats), candidates=len(rows), excluded_parent_keys=len(bad),
                exclusion_paths_unresolved=excl_unresolved,
                picked=len(picked), by_dataset_sft=dict(Counter(r['dataset'] for r in sft)),
                by_dataset_dpo=dict(Counter(r['dataset'] for r in dpo)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', required=True); ap.add_argument('--seed', type=int, default=20261123)
    ap.add_argument('--motion-q', type=float, default=0.5); ap.add_argument('--per-parent', type=int, default=3)
    ap.add_argument('--n-sft', type=int, default=10000); ap.add_argument('--n-dpo', type=int, default=640)
    a = ap.parse_args()
    caps = dict(hoigen1m=4000, charades=2500, celebvhq=1500, direct=1200, vfhq=800, matrix=800, noxi=600, egoexo4d=0)
    used = set()
    for l in open(ROOT / 'training_metadata/l009b_20261122/pool1758_l009b_manifest.jsonl'):
        v = json.loads(l)['video']; used.add(os.path.relpath(v, ROOT))
    res = build(a.seed, a.motion_q, a.per_parent, caps, a.n_sft, a.n_dpo, used)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=False)
    with open(out / 't3_sft_manifest.jsonl', 'x') as f:
        for i, r in enumerate(res['sft']):
            sid = 't3_' + hashlib.sha256(r['rel'].encode()).hexdigest()[:16]
            f.write(json.dumps(dict(sample_id=sid, video=r['video'], captions=dict(q35=r['prompt']), dataset=r['dataset'],
                                    parent=r['parent'], diff_median=r['diff_median'])) + '\n')
    with open(out / 't2_dpo_seeds.jsonl', 'x') as f:
        for r in res['dpo']:
            f.write(json.dumps(dict(video=r['video'], prompt=r['prompt'], dataset=r['dataset'], parent=r['parent'],
                                    diff_median=r['diff_median'])) + '\n')
    summ = {k: v for k, v in res.items() if k not in ('sft', 'dpo')}
    summ.update(n_sft=len(res['sft']), n_dpo=len(res['dpo']), caps=caps, seed=a.seed, motion_q=a.motion_q,
                excluded_l009b_windows=len(used), eval_sets_excluded=['e1_devset', 'wm108', 'fixed_val120', 'P1', 'C1', 'M1',
                                                                     'training_smoke', 'inference_dev_20261017'],
                sha256={n: hashlib.sha256((out / n).read_bytes()).hexdigest() for n in ('t3_sft_manifest.jsonl', 't2_dpo_seeds.jsonl')})
    (out / 'summary.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
