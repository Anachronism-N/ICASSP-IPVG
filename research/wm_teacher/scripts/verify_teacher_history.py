#!/usr/bin/env python3
"""Authenticate immutable old run evidence before retiring reservations, not costs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time

ROOT = Path(__file__).resolve().parents[1]
EXECUTION_ID = 'recovery_a1'
INDEX = 'teacher_devset_original_tool_logs_20261026_1408_0f6ab742.json'
INDEX_SHA = 'cbb6069f950d07cfd67bb94b171d79c31a807e155b6ceeb1a95d4b5d26c484a7'
AUDITS = {
    'alg': ('teacher_alg_race_audit_20261026_1408_0f6ab742.json', '59657153babe30e7d90198e5e6aa69070a86bcf4e5d9fd2426f34e54f3f648d9', 5926.31142393779, {1: ('zZrWKa', 686), 2: ('ljzRXI', 452)}),
    'dymos': ('teacher_dymos_race_audit_20261026_1350_8c43f913.json', 'b764c0a4885fc7c705f6f72ebb0190c23b5dd9a62f79d8de76e63056563840fc', 9559.88809741661, {0: ('z59WNN', 682), 3: ('J1o9ym', 681)}),
}
LOG_SHAS = {
    ('zZrWKa', 'stdout'): '6920a383dfc732f004b4f354467be16cae84688db4bbd152c68497fa81144c34',
    ('zZrWKa', 'stderr'): '45a2f04c90d6a0d17d0f0368bb0aef5b9fed489bbaf0ee0fd27d7e5919eab145',
    ('ljzRXI', 'stdout'): '083068fff91f09224af8c17486a2d171e0a0b2ddd179e5380bd6a689cf6ed8c1',
    ('ljzRXI', 'stderr'): '06fa670a32aa0832cb212e132356d13651e2ac1d18049bac8cd05cce3d189c45',
    ('z59WNN', 'stdout'): '3f343ea15e8784b5e2c55d20e7c5bef13089913884fffffba3cc544913c90e72',
    ('z59WNN', 'stderr'): '5415037aacecdf9e1777593ec9301780cddabb1caa034d2f2e1bff222f528fb3',
    ('J1o9ym', 'stdout'): 'eedfa8925f5bba1984732b2bc24d24e8fea97abaa1453983872972e76c33dddf',
    ('J1o9ym', 'stderr'): '7909d8296fe0a16549d2555f6760efc0283a3d2a530de892ec12aea58db41c11',
}
OLD_SCRIPTS = {'run_teacher_devset.py', 'run_teacher_devset_node.py'}
PROC_ROOT = Path('/proc')
INVENTORY_MAX_AGE_SECONDS = 86400


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + chr(10)).encode()


def record(path):
    path = Path(path).absolute()
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    after = path.stat()
    require(path.resolve() == path and (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Evidence changed while reading or symlinked')
    return dict(path=str(path), bytes=after.st_size, mtime_ns=after.st_mtime_ns, sha256=digest.hexdigest())


def read_bound(row, pinned=None):
    actual = record(row['path'])
    require(all(actual[k] == row[k] for k in ('path', 'bytes', 'sha256')) and (pinned is None or actual['sha256'] == pinned), 'Historical evidence SHA drift')
    return Path(row['path']).read_bytes()


def inventory(rank, root=ROOT):
    require(type(rank) is int and 0 <= rank <= 3, 'Actual node rank required')
    root = Path(root).resolve()
    matches = []
    for entry in PROC_ROOT.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            raw = (entry / 'cmdline').read_bytes()
            args = [os.fsdecode(x) for x in raw.split(bytes([0])) if x]
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            if not args or fields[0] in ('Z', 'X'):
                continue
            scripts = {Path(arg).name for arg in args} & OLD_SCRIPTS
            cwd = Path(os.readlink(entry / 'cwd'))
            old_roots = {root / f'eval/teacher_{method}_devset_20261026' for method in AUDITS}
            old_root_present = any(str(old_root) in arg and EXECUTION_ID not in arg
                                   for old_root in old_roots for arg in args)
            for index, arg in enumerate(args):
                candidate = arg.split('=', 1)[1] if arg.startswith('--plan=') else (
                    args[index + 1] if arg == '--plan' and index + 1 < len(args) else None)
                if candidate is not None:
                    resolved = Path(candidate) if Path(candidate).is_absolute() else cwd / candidate
                    old_root_present = old_root_present or resolved.resolve().parent in old_roots
            if scripts and old_root_present:
                matches.append(dict(pid=int(entry.name), ppid=int(fields[1]), start_ticks=int(fields[19]),
                                    session=int(fields[3]), scripts=sorted(scripts), argv_sha256=hashlib.sha256(raw).hexdigest()))
        except (OSError, ValueError, IndexError):
            continue
    return dict(schema='teacher_old_run_inventory_v1', node_rank=rank, hostname=socket.gethostname(),
                root=str(root), checked_epoch=time.time(), matches=matches, old_scripts=sorted(OLD_SCRIPTS),
                verifier_source=record(__file__), GPU_used=False, signals_sent=False)


def verified_old_evidence(method, root=ROOT):
    require(method in AUDITS, 'One historical method required')
    root = Path(root).resolve()
    filename, audit_sha, upper, owners = AUDITS[method]
    audit_record = record(root / 'eval' / filename)
    audit = json.loads(read_bound(audit_record, audit_sha))
    require(audit['schema'] == 'teacher_devset_readonly_race_audit_v1' and audit['originals_modified'] is False
            and audit['unknown_shards'] == [] and audit['counts']['mp4'] == 0, 'Incomplete historical audit')
    old = json.loads(read_bound(audit['plan_record']))
    read_bound(audit['ledger_record'])  # Damaged bytes remain evidence, not a valid live ledger.
    require(old['method'] == method and old['nshards'] == 16 and old['video_count'] == 188 and old['seeds'] == [0, 1]
            and old['phase_seconds']['generation'] == 86400, 'Wrong historical plan/budget')
    shards = audit['shards']
    require(len(shards) == 16 and {s['shard'] for s in shards} == set(range(16)), 'All sixteen old shards must be charged')
    require(all(type(s['cost_upper_bound_seconds']) in (int, float) and math.isfinite(s['cost_upper_bound_seconds'])
                and s['cost_upper_bound_seconds'] > 0 for s in shards), 'Missing old conditional upper')
    require(math.isclose(sum(s['cost_upper_bound_seconds'] for s in shards), upper, abs_tol=1e-8), 'Old upper drift')
    index_record = record(root / 'eval' / INDEX)
    index = json.loads(read_bound(index_record, INDEX_SHA))
    require(index['schema'] == 'teacher_devset_original_tool_logs_readonly_v1' and index['originals_modified'] is False, 'Wrong original tool index')
    task_index = {r['task_id']: r for r in index['task_summaries']}
    logs = []
    for rank, (task, duration) in owners.items():
        summary = task_index[task]
        require(summary['rank'] == rank and summary['method'] == method and summary['reported_duration_seconds'] == duration, 'Task/node bound mismatch')
        for stream in ('stdout', 'stderr'):
            rows = [r for r in index['records'] if r['task_id'] == task and r['stream'] == stream]
            require(len(rows) == 1, 'Missing/duplicate original stream')
            row = record(rows[0]['path'])
            payload = read_bound(row, LOG_SHAS[(task, stream)])
            require(b'[truncated]' not in payload and b'...[truncated]' not in payload, 'Full raw log required, not index excerpt')
            if stream == 'stdout':
                decoded = []
                for line in payload.decode().splitlines():
                    try:
                        value = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(value, dict):
                        decoded.append(value)
                require(any(v.get('method') == method and (v.get('rank') == rank or
                        v.get('plan_sha256') == audit['plan_record']['sha256']) for v in decoded), 'Raw stdout is not bound to old method/plan')
            logs.append(dict(node_rank=rank, task_id=task, stream=stream, reported_duration_seconds=duration, file=row))
    return audit, dict(audit=audit_record, old_plan=audit['plan_record'], old_ledger=audit['ledger_record'],
                       raw_tool_index=index_record, raw_tool_logs=logs)


def prepare_retirement(method, inventories, output, root=ROOT):
    audit, evidence = verified_old_evidence(method, root)
    require(set(inventories) == set(AUDITS[method][3]), 'Both old owners inventory required')
    inventory_rows = []
    for rank, row in sorted(inventories.items()):
        value = json.loads(read_bound(row))
        require(value['schema'] == 'teacher_old_run_inventory_v1' and value['node_rank'] == rank
                and value['root'] == str(Path(root).resolve()) and value['matches'] == []
                and value['signals_sent'] is False and value['GPU_used'] is False
                and value['old_scripts'] == sorted(OLD_SCRIPTS), 'Old execution is not fully stopped')
        require(record(value['verifier_source']['path']) == value['verifier_source']
                and value['verifier_source']['sha256'] == record(__file__)['sha256'], 'Inventory verifier changed')
        inventory_rows.append(dict(node_rank=rank, hostname=value['hostname'], file=row))
    require(len({row['hostname'] for row in inventory_rows}) == 2, 'Actual different nodes required')
    proof = dict(schema='teacher_generation_retirement_v1', private=True, method=method, execution_id=EXECUTION_ID,
                 old_actual_seconds=None, old_generation_upper_seconds=AUDITS[method][2],
                 **evidence, termination_inventory=inventory_rows,
                 termination_verifier_source=record(__file__), source_files=[record(__file__)],
                 shards=[dict(shard=s['shard'], node_rank=s['node_rank'], retired=True, source_verified=True,
                              complete_tool_logs=True, old_cost_upper_seconds=s['cost_upper_bound_seconds'],
                              records=[r['file'] for r in evidence['raw_tool_logs'] if r['node_rank'] == s['node_rank']])
                         for s in audit['shards']],
                 caveat='Only old reservations are retired. Historical actual cost remains unknown; all16 audited upper is retained once.')
    verify_retirement(proof, {'root': str(Path(root).resolve()), 'method': method,
                             'prior_generation': dict(audit=evidence['audit'], old_plan=evidence['old_plan'], old_ledger=evidence['old_ledger'])})
    output = Path(output).absolute()
    require(not output.exists() and output.resolve() == output, 'New immutable retirement proof required')
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(encoded(proof)); handle.flush(); os.fsync(handle.fileno())
    return record(output)


def verify_retirement(proof, plan):
    method, root = plan['method'], Path(plan['root'])
    require(method in AUDITS and proof.get('schema') == 'teacher_generation_retirement_v1'
            and proof.get('execution_id') == EXECUTION_ID and proof.get('method') == method
            and proof.get('old_actual_seconds') is None
            and proof.get('old_generation_upper_seconds') == AUDITS[method][2], 'Unknown old cost cannot be changed')
    require(isinstance(proof.get('audit'), dict) and proof['audit'].get('sha256') == AUDITS[method][1],
            'Pinned historical audit required before reading private evidence')
    audit, expected = verified_old_evidence(method, root)
    for key in ('audit', 'old_plan', 'old_ledger', 'raw_tool_index', 'raw_tool_logs'):
        require(proof[key] == expected[key], 'Pinned old evidence binding drift')
    require(all(proof[k] == plan['prior_generation'][k] for k in ('audit', 'old_plan', 'old_ledger')), 'Retirement plan drift')
    expected_shards = [dict(shard=s['shard'], node_rank=s['node_rank'], retired=True, source_verified=True,
                           complete_tool_logs=True, old_cost_upper_seconds=s['cost_upper_bound_seconds'],
                           records=[r['file'] for r in expected['raw_tool_logs'] if r['node_rank'] == s['node_rank']])
                       for s in audit['shards']]
    require(proof['shards'] == expected_shards, 'All16 retirement/raw-log/accounting evidence drift')
    require(proof['termination_verifier_source'] == record(__file__) and proof['source_files'] == [record(__file__)], 'History verifier source drift')
    rows = proof['termination_inventory']
    require(len(rows) == 2 and {r['node_rank'] for r in rows} == set(AUDITS[method][3])
            and len({r['hostname'] for r in rows}) == 2, 'Actual old node inventories required')
    for row in rows:
        value = json.loads(read_bound(row['file']))
        require(value['schema'] == 'teacher_old_run_inventory_v1' and value['matches'] == []
                and value['node_rank'] == row['node_rank'] and value['hostname'] == row['hostname']
                and value['root'] == str(root.resolve()) and value['signals_sent'] is False and value['GPU_used'] is False
                and value['verifier_source'] == record(__file__) and value['old_scripts'] == sorted(OLD_SCRIPTS)
                and type(value['checked_epoch']) in (float, int) and math.isfinite(value['checked_epoch'])
                and -5 <= time.time() - value['checked_epoch'] <= INVENTORY_MAX_AGE_SECONDS,
                'Unverified old process inventory')
    return proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', action='store_true')
    parser.add_argument('--rank', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    require(args.inventory and args.rank is not None, 'Explicit CPU inventory rank required')
    value = inventory(args.rank)
    if args.output:
        require(not args.output.exists() and args.output.resolve() == args.output.absolute(), 'New inventory path required')
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as handle:
            handle.write(encoded(value)); handle.flush(); os.fsync(handle.fileno())
        print(json.dumps(record(args.output)))
    else:
        print(json.dumps(value))


if __name__ == '__main__':
    main()
