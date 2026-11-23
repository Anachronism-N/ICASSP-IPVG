#!/usr/bin/env python3
"""Parallel finalize for cache_e3_inputs caches: identical checks and index content, threads for file hashing.

Imports cache_e3_inputs unchanged (its binding records the cacher's own SHA), runs the same per-record checks
concurrently, then writes cache_index.json exactly as cache_e3_inputs.finalize would.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cache_e3_inputs as ce  # noqa: E402

m1, require = ce.m1, ce.require


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True); ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--framework', type=Path, default=ce.FRAMEWORK_DEFAULT); ap.add_argument('--workers', type=int, default=32)
    a = ap.parse_args()
    rows, versions = ce.read_manifest(a.manifest)
    root = Path(a.out_dir).absolute()
    binding = m1.read_json(root / 'binding.json')
    require(binding == ce.make_binding(a.manifest, rows, versions, binding['preprocessing']['frame_policy'] == 'first121',
                                       a.framework), 'Binding differs from manifest/code')

    def check(row):
        rec_path = ce.paths(root, row['sample_id'], versions)[2]
        require(rec_path.exists(), 'Missing record: ' + row['sample_id'])
        rec = m1.read_json(rec_path)
        require(rec['sample_id'] == row['sample_id'] and set(rec['contexts']) == set(versions)
                and all(rec['contexts'][v]['prompt_sha256'] == m1.text_hash(row['captions'][v]) for v in versions),
                'Record differs from manifest: ' + row['sample_id'])
        require(ce.record_ok(root, rec), 'Record files changed: ' + row['sample_id'])
        return rec
    with ThreadPoolExecutor(a.workers) as ex:
        samples = list(ex.map(check, rows))
    index = {'protocol': ce.PROTOCOL, 'complete': True, 'binding': binding, 'versions': versions, 'samples': samples,
             'x0_bytes_total': sum(s['x0']['bytes'] for s in samples),
             'context_bytes_total': sum(c['bytes'] for s in samples for c in s['contexts'].values())}
    m1.atomic_json(root / 'cache_index.json', index)
    print(json.dumps({'status': 'finalized', 'samples': len(samples), 'versions': versions, 'index': str(root / 'cache_index.json'),
                      'index_sha256': m1.sha256(root / 'cache_index.json'),
                      'bytes_total': index['x0_bytes_total'] + index['context_bytes_total']}))


if __name__ == '__main__':
    main()
