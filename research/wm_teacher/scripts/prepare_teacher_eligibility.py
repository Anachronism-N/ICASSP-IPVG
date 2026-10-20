#!/usr/bin/env python3
"""Derive a private namespace-screened candidate CSV; never approve training."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
CSV_REL = 'training_metadata/ft2/ft2_train.csv'
SCOPE_REL = 'eval/diag/namespace_inventory_scope_20261020.json'
CSV_SHA = '2fd1615c3c538d9d9462ad166b78129f8c349174bffd5ce1085888d5f001a0a6'
SCOPE_SHA = '1a874efd3253f4e1ce1b8a2696c0e4bdfe9167c38e0c59a373262a753147ace6'
EXPECTED_ROWS, EXPECTED_QUARANTINE = 32365, 19


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def dataset(video):
    require(isinstance(video, str) and chr(92) not in video, 'Invalid video path')
    path = PurePosixPath(video)
    require(not path.is_absolute() and path.as_posix() == video and '..' not in path.parts,
            'Video must be a safe canonical ROOT-relative path')
    require(len(path.parts) >= 4 and path.parts[:2] == ('dataset', 'wan22_5s')
            and path.parts[2].endswith('_5s'), 'Unexpected video layout')
    return path.parts[2][:-3]


def partition(rows, scope):
    require(scope['status'] == 'completed_metadata_potential_collision_scope_NOT_proof_of_media_overwrite',
            'Unverified namespace scope')
    require(scope['raw_data_modified'] is False and scope['GPU_used'] is False, 'Unexpected scope operation')
    coverage = scope['coverage']['ft2']
    require(len(rows) == EXPECTED_ROWS == coverage['total_records'], 'Original CSV coverage differs')
    records = scope['private_cases']['ft2']
    require(len(records) == coverage['ambiguous_namespace_records'], 'Namespace record coverage differs')
    flagged = {}
    for row in records:
        require(type(row['multiple_candidate_runs']) is bool, 'Namespace flag must be bool')
        ds = dataset(row['video'])
        require(ds == row['dataset'] and row['video'] not in flagged, 'Duplicate/inconsistent scope identity')
        flagged[row['video']] = row['multiple_candidate_runs']
    exclude = {video for video, ambiguous in flagged.items() if ambiguous}
    require(len(exclude) == EXPECTED_QUARANTINE == coverage['multiple_candidate_run_records'],
            'Unexpected quarantine count')
    videos = set()
    kept, quarantined = [], []
    for row in rows:
        require(set(row) == {'video', 'prompt'} and isinstance(row['prompt'], str) and row['prompt'],
                'Missing or malformed original CSV fields')
        dataset(row['video'])
        require(row['video'] not in videos, 'Duplicate video in original CSV')
        videos.add(row['video'])
        (quarantined if row['video'] in exclude else kept).append(row)
    require(set(flagged) <= videos, 'Scope contains videos not present in frozen CSV')
    require(len(quarantined) == len(exclude), 'Quarantine identity coverage differs')
    return kept, quarantined


def csv_bytes(rows):
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text, fieldnames=['video', 'prompt'], lineterminator=chr(10))
    writer.writeheader()
    writer.writerows(rows)
    return text.getvalue().encode('utf-8')


def prepare(root, out_dir):
    root = Path(root).resolve()
    out = Path(out_dir).absolute()
    require(not out.exists() and not out.is_symlink(), 'Output must be new; no resume or overwrite')
    out = out.resolve()
    require(out != root / 'training_metadata' and out.is_relative_to(root / 'training_metadata'),
            'Output must be inside private ROOT/training_metadata')
    csv_path, scope_path = root / CSV_REL, root / SCOPE_REL
    for path in (csv_path, scope_path):
        require(path.is_file() and path.resolve().is_relative_to(root), 'Input escapes project or is missing')
    raw, scope_raw = csv_path.read_bytes(), scope_path.read_bytes()
    require(digest(raw) == CSV_SHA and digest(scope_raw) == SCOPE_SHA, 'Frozen input SHA256 mismatch')
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig'), newline=''))
    require(reader.fieldnames == ['video', 'prompt'], 'Original CSV header differs')
    rows = list(reader)
    kept, quarantined = partition(rows, json.loads(scope_raw))
    outputs = {'namespace_candidates.csv': csv_bytes(kept), 'namespace_quarantine.csv': csv_bytes(quarantined)}
    require(csv_path.read_bytes() == raw and scope_path.read_bytes() == scope_raw,
            'Frozen inputs changed during validation')
    out.mkdir(mode=0o700, parents=False)
    for name, data in outputs.items():
        fd = os.open(out / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    require(digest(csv_path.read_bytes()) == CSV_SHA and digest(scope_path.read_bytes()) == SCOPE_SHA,
            'Inputs changed before publication; preserve incomplete output')
    for name, data in outputs.items():
        require((out / name).read_bytes() == data, 'Candidate output content differs')
    summary = {
        'status': 'namespace_screened_candidates_NOT_training_approved',
        'original_rows': len(rows), 'candidate_rows': len(kept), 'quarantined_rows': len(quarantined),
        'original_csv_sha256': CSV_SHA, 'scope_sha256': SCOPE_SHA,
        'output_sha256': {name: digest(data) for name, data in outputs.items()},
        'candidates_by_dataset': dict(Counter(dataset(row['video']) for row in kept)),
        'quarantine_by_dataset': dict(Counter(dataset(row['video']) for row in quarantined)),
        'captions_modified': False, 'original_csv_modified': False, 'media_deleted_or_modified': False,
        'quarantine_is_proof_of_label_error': False, 'training_approved': False, 'GPU_used': False,
        'remaining_gates': ['semantic-label review', 'train/development parent-source isolation',
                            'training scope/budget and gradient/save/restore smoke'],
        'human_review': 'pending', 'teacher_passed': False,
    }
    data = (json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + chr(10)).encode()
    fd = os.open(out / 'eligibility_summary.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(data)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args()
    summary = prepare(args.root, args.out_dir)
    print(json.dumps({k: summary[k] for k in ('status', 'candidate_rows', 'quarantined_rows', 'training_approved')}))


if __name__ == '__main__':
    main()
