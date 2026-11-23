#!/usr/bin/env python3
"""Paired dev-set scoring wrapper and preregistered T1-vs-base summary (original E1 scorers and rule).

score   : calls score_e1.cmd_vbench / cmd_gtref unchanged on <root>/videos/<config>/ (needs the scoring venv).
summary : CPU; per-seed guards (score_e1.RULE), two-seed mean Total* delta with case-cluster bootstrap CI,
          egoexo4d-excluded sensitivity. Reports only; promotion still needs user review.
"""
import argparse
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
EVAL_HOME = Path(os.environ.get('WM_EVAL_HOME', '/dockerdata/wm_eval_20261121'))


def score(kind, configs, root):
    import score_e1 as s
    s.VBREPO = str(EVAL_HOME / 'VBench')
    s.WORK_BASE = str(EVAL_HOME / 'work')
    os.environ.setdefault('VBENCH_CACHE_DIR', str(EVAL_HOME / 'vbench_cache'))
    (s.cmd_vbench if kind == 'vbench' else s.cmd_gtref)(configs, root=str(root))


def cluster_bootstrap(diffs_by_seed, b=2000, seed=20261021):
    ids = sorted(diffs_by_seed[0])
    rng = random.Random(seed)
    mean = sum(sum(d[i] for d in diffs_by_seed) / len(diffs_by_seed) for i in ids) / len(ids)
    boots = []
    for _ in range(b):
        pick = [ids[rng.randrange(len(ids))] for _ in ids]
        boots.append(sum(sum(d[i] for d in diffs_by_seed) / len(diffs_by_seed) for i in pick) / len(pick))
    boots.sort()
    return dict(mean=mean, ci95=[boots[int(0.025 * b)], boots[int(0.975 * b) - 1]], n_cases=len(ids), seeds=len(diffs_by_seed))


def summarize(root, candidate='T1', base='base', seeds=(0, 1), exclude=None):
    import score_e1 as s
    cases = s.load_cases(str(root))
    if exclude:
        cases = [c for c in cases if c['dataset'] not in exclude]
    keep = {c['id'] for c in cases}
    keys = s.DIMS + list(s.GT_KEYS) + ['total']
    per_seed, diffs = {}, []
    for sd in seeds:
        cr = {i: v for i, v in s.load_config_rows(f'{candidate}_s{sd}', s.load_cases(str(root)), str(root)).items() if i in keep}
        br = {i: v for i, v in s.load_config_rows(f'{base}_s{sd}', s.load_cases(str(root)), str(root)).items() if i in keep}
        cm, bm = s.mean_stats(cr), s.mean_stats(br)
        checks = s.guard_checks(cm, bm)
        per_seed[sd] = dict(candidate=cm, base=bm, paired=s.paired(cr, br, keys), guards=checks, guards_pass=all(checks.values()))
        diffs.append({i: cr[i]['total'] - br[i]['total'] for i in keep})
    boot = cluster_bootstrap(diffs)
    qualified = all(v['guards_pass'] for v in per_seed.values())
    return dict(protocol='T1 preregistered paired dev evaluation', n_cases=len(keep), per_seed=per_seed,
                two_seed_total_delta=boot, qualified=qualified,
                improvement_claim=qualified and boot['mean'] > 0 and boot['ci95'][0] > 0,
                not_blind_test=True, sota_proven=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('score'); p.add_argument('kind', choices=('vbench', 'gtref')); p.add_argument('--root', required=True)
    p.add_argument('configs', nargs='+')
    p = sub.add_parser('summary'); p.add_argument('--root', required=True); p.add_argument('--out', required=True)
    p.add_argument('--candidate', default='T1')
    p = sub.add_parser('score-shard'); p.add_argument('kind', choices=('vbench', 'gtref')); p.add_argument('--root', required=True)
    p.add_argument('--configs', required=True, help='comma list; shard k scores configs[k]'); p.add_argument('--shard', type=int, required=True)
    a = ap.parse_args()
    if a.cmd == 'score':
        score(a.kind, a.configs, a.root)
    elif a.cmd == 'score-shard':
        score(a.kind, [a.configs.split(',')[a.shard]], a.root)
    else:
        out = dict(candidate=a.candidate, primary=summarize(a.root, candidate=a.candidate),
                   sensitivity_excluding_egoexo4d=summarize(a.root, candidate=a.candidate, exclude={'egoexo4d'}))
        with open(a.out, 'x') as f:
            json.dump(out, f, indent=1)
        print(json.dumps(dict(qualified=out['primary']['qualified'], improvement_claim=out['primary']['improvement_claim'],
                              total_delta=out['primary']['two_seed_total_delta'])))


if __name__ == '__main__':
    main()
