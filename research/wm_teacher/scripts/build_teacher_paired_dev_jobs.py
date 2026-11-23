#!/usr/bin/env python3
"""Build paired dev-set generation jobs (base vs one frozen candidate, seeds 0 and 1) for teacher_dev_generate.py.

Layout matches score_e1.py: <root>/cases.jsonl (the original 94 e1 cases, byte copy) and
<root>/videos/<model>_s<seed>/<case id>.mp4, so the original VBench/GT-reference scorers apply unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
E1 = ROOT / 'eval/e1_devset_20261021'
CASES_SHA256 = '0873b163284a92b207ac248cc41ff934cf73615f1f9af073eca8bfb2073e5aed'


def build(root, models, seeds):
    root = Path(root)
    raw = (E1 / 'cases.jsonl').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CASES_SHA256, 'Frozen 94-case dev set drift'
    cases = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    assert len(cases) == 94
    root.mkdir(parents=True, exist_ok=False)
    (root / 'cases.jsonl').write_bytes(raw)
    jobs = []
    for name, ckpt in models:
        assert ckpt == 'NONE' or Path(ckpt).is_file(), ckpt
        for seed in seeds:
            for c in cases:
                assert Path(c['image']).is_file()
                jobs.append(dict(id=f"{c['id']}__{name}__s{seed}", model=f'{name}_s{seed}', ckpt=ckpt, image=c['image'],
                                 prompt=c['prompt'], out=str(root / 'videos' / f'{name}_s{seed}' / f"{c['id']}.mp4"), seed=seed))
    with open(root / 'jobs.jsonl', 'x') as f:
        f.write(''.join(json.dumps(j, sort_keys=True) + '\n' for j in jobs))
    return dict(root=str(root), jobs=len(jobs), sha256=hashlib.sha256((root / 'jobs.jsonl').read_bytes()).hexdigest())


def add_model(root, name, ckpt, seeds):
    """Append one more model to an existing paired root (same frozen cases), as jobs_<name>.jsonl."""
    root = Path(root)
    raw = (root / 'cases.jsonl').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CASES_SHA256, 'Paired root case file drift'
    assert ckpt == 'NONE' or Path(ckpt).is_file(), ckpt
    cases = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    jobs = [dict(id=f"{c['id']}__{name}__s{seed}", model=f'{name}_s{seed}', ckpt=ckpt, image=c['image'], prompt=c['prompt'],
                 out=str(root / 'videos' / f'{name}_s{seed}' / f"{c['id']}.mp4"), seed=seed) for seed in seeds for c in cases]
    path = root / f'jobs_{name}.jsonl'
    with open(path, 'x') as f:
        f.write(''.join(json.dumps(j, sort_keys=True) + '\n' for j in jobs))
    return dict(jobs=len(jobs), path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True)
    ap.add_argument('--model', action='append', required=True, help='name=ckpt_or_NONE')
    ap.add_argument('--seeds', default='0,1')
    ap.add_argument('--append', action='store_true', help='add one model to an existing root')
    a = ap.parse_args()
    models = [tuple(m.split('=', 1)) for m in a.model]
    seeds = [int(s) for s in a.seeds.split(',')]
    if a.append:
        assert len(models) == 1
        print(json.dumps(add_model(a.root, models[0][0], models[0][1], seeds)))
    else:
        print(json.dumps(build(a.root, models, seeds)))


if __name__ == '__main__':
    main()
