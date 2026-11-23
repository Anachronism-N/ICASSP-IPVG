#!/usr/bin/env python3
"""CPU post-processing for label009b scale outputs: summary, blind human sample pack, cache manifest.

Never edits labels; excluded (failed/unattempted) windows never enter the cache manifest.
"""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
VERSION = 'l009b'


def load(plan_path):
    p = json.loads(Path(plan_path).read_text())
    run = Path(p['output_root'])
    outs = {}
    for job in p['jobs']:
        f = run / 'labels' / f"{job['index']}.json"
        if f.exists():
            outs[job['index']] = json.loads(f.read_text())
    return p, run, outs


def summary(plan_path):
    import run_teacher_label_scale as ls
    p, run, outs = load(plan_path)
    by = {}
    for job in p['jobs']:
        d = by.setdefault(job['dataset'], dict(windows=0, labels=0, excluded=0, not_attempted=0))
        d['windows'] += 1
        o = outs.get(job['index'])
        d['not_attempted' if o is None else 'labels' if o['success'] else 'excluded'] += 1
    ok = [o for o in outs.values() if o['success']]
    for o in ok:
        ls.text_integrity(o['caption'])
    words = sorted(len(re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", o['caption'])) for o in ok)
    reasons, camera = {}, {}
    for o in outs.values():
        if not o['success']:
            reasons[o.get('failure_reason')] = reasons.get(o.get('failure_reason'), 0) + 1
        else:
            camera[o['facts_json']['camera_motion']] = camera.get(o['facts_json']['camera_motion'], 0) + 1
    return dict(windows=len(p['jobs']), attempted=len(outs), labels=len(ok), excluded=len(outs) - len(ok),
                not_attempted=len(p['jobs']) - len(outs), by_dataset=by, failure_reasons=reasons, camera_motion=camera,
                caption_words=dict(min=words[0], median=words[len(words) // 2], max=words[-1]) if words else None,
                text_integrity_recheck_pass=True)


def human_pack(plan_path, out, n, seed):
    p, run, outs = load(plan_path)
    out = Path(out)
    out.mkdir(mode=0o700); (out / 'sheets').mkdir(mode=0o700); (out / 'human_verdicts').mkdir(mode=0o700)
    ok = sorted(i for i, o in outs.items() if o['success'])
    chosen = random.Random(seed).sample(ok, n)
    items, rows = [], []
    for k, i in enumerate(chosen):
        o = outs[i]
        sheet = f'sheets/item{k:02d}.jpg'
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', o['video']['path'], '-vf',
                        "select='not(mod(n\\,8))',scale=320:-1,tile=4x4", '-frames:v', '1', str(out / sheet)], check=True)
        (out / sheet).chmod(0o600)
        lf = run / 'labels' / f'{i}.json'
        items.append(dict(item=k, label_index=i, sample_id=o['sample_id'], dataset=o['dataset'],
                          label_sha256=hashlib.sha256(lf.read_bytes()).hexdigest(), video=o['video']['path']))
        rows.append(f"<h3>Item {k:02d}</h3><p><b>camera_motion (label):</b> {html.escape(o['facts_json']['camera_motion'])}</p>"
                    f"<p style='max-width:1280px'>{html.escape(o['caption'])}</p><img src='{sheet}' width='1280'>"
                    f"<p>Original video: <code>{html.escape(o['video']['path'])}</code></p><hr>")
    page = ("<html><meta charset='utf-8'><body><h2>Label009b scaled-output blind human review (private)</h2>"
            "<p>Random sample of retained labels. Each sheet shows the 16 frames the labeler saw (0,8,...,120). "
            "For each item set overall = usable / needs_fix / reject (optional action/camera/entity notes) in "
            "human_verdicts/verdicts.json (copy verdicts.template.json). AI verdicts are hidden.</p>" + ''.join(rows) + '</body></html>')
    for name, data in (('index.html', page),
                       ('verdicts.template.json', json.dumps([dict(item=x['item'], overall='', action='', camera='', entity='', notes='')
                                                               for x in items], indent=1)),
                       ('mapping.private.json', json.dumps(dict(items=items, seed=seed, n=n, ai_verdicts_hidden=True), indent=1))):
        (out / name).write_text(data); (out / name).chmod(0o600)
    return dict(pack=str(out), items=len(items))


def cache_manifest(plan_path, out):
    p, run, outs = load(plan_path)
    pool = {json.loads(l)['sample_id']: json.loads(l) for l in
            (ROOT / 'training_metadata/e3_recaption_20261022/pool1856_dual_manifest.jsonl').read_text().splitlines()}
    lines, excluded = [], []
    for job in p['jobs']:
        o = outs.get(job['index'])
        if o is None or not o['success']:
            excluded.append(dict(sample_id=job['sample_id'], dataset=job['dataset'],
                                 reason='not_attempted' if o is None else o.get('failure_reason')))
            continue
        assert pool[job['sample_id']]['video'] == job['video']['path'] == o['video']['path']
        lines.append(json.dumps(dict(sample_id=job['sample_id'], video=job['video']['path'],
                                     captions={VERSION: o['caption']}, dataset=job['dataset'],
                                     label_sha256=hashlib.sha256((run / 'labels' / f"{job['index']}.json").read_bytes()).hexdigest()),
                                sort_keys=True))
    out = Path(out)
    with open(out, 'x') as f:
        f.write('\n'.join(lines) + '\n')
    out.chmod(0o600)
    ex = out.with_suffix('.excluded.json')
    with open(ex, 'x') as f:
        json.dump(excluded, f, indent=1)
    ex.chmod(0o600)
    return dict(manifest=str(out), rows=len(lines), excluded=len(excluded),
                sha256=hashlib.sha256(out.read_bytes()).hexdigest())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('summary', 'human-pack', 'cache-manifest'):
        s = sub.add_parser(name); s.add_argument('--label-plan', required=True)
        if name != 'summary':
            s.add_argument('--out', required=True)
        if name == 'human-pack':
            s.add_argument('--n', type=int, default=30); s.add_argument('--seed', type=int, default=20261122)
    a = ap.parse_args()
    if a.cmd == 'summary':
        print(json.dumps(summary(a.label_plan), indent=1))
    elif a.cmd == 'human-pack':
        print(json.dumps(human_pack(a.label_plan, a.out, a.n, a.seed)))
    else:
        print(json.dumps(cache_manifest(a.label_plan, a.out)))


if __name__ == '__main__':
    main()
