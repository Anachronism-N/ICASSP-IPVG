#!/usr/bin/env python3
"""CPU budget amendment for the user's one recovery; does not launch GPUs.

Previously generated diagnostic videos are preserved but not silently mixed
into a new strict comparison. All three methods are planned fresh under one
newly frozen supervisor version, subject to actual per-shard feasibility.
"""
import argparse
import json
from pathlib import Path

import run_teacher_generation as generation

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'teacher_one_recovery_budget_v1'
EXECUTION_ID = 'recovery_a2'
PRIOR_SECONDS = {'base': 4387.127680046484, 'alg': 5926.31142393779 + 4879.150361155625,
                 'dymos': 9559.88809741661}
CAPS = {'base': [4300] * 12 + [4000] * 4,
        'alg': [4775] * 12 + [4475] * 4,
        'dymos': [4800] * 16}
PHASES = {'base': {'generation': 72000, 'score': 18000},
          'alg': dict(generation.dev.PHASE_SECONDS), 'dymos': dict(generation.dev.PHASE_SECONDS)}
require, record, sha, encoded = generation.require, generation.record, generation.sha, generation.encoded


def audit_inputs(root):
    root = Path(root)
    folder = root / 'eval/teacher_execution_recovery_a1'
    old = {method: record(root / 'eval' / generation.recovery.PRIOR[method][0], nonempty=True)
           for method in ('alg', 'dymos')}
    require(all(old[m]['sha256'] == generation.recovery.PRIOR[m][1] for m in old), 'Pinned prior audit drift')
    recent = {method: record(folder / ('first_actual_' + method + '_attempt_audit.json'), nonempty=True)
              for method in ('base', 'alg')}
    values = {method: generation.read_bound(row) for method, row in recent.items()}
    for method, audit in values.items():
        require(audit['schema'] == 'teacher_first_actual_attempt_audit_v1' and audit['method'] == method
                and audit['completed_videos'] == 10 and audit['generation_complete'] is False
                and audit['quality_pass'] is False and audit['GPU_released_evidence'] is True,
                'Actual failed-attempt audit required; partial output is not completion')
        parent = generation.read_bound(audit['parent'])
        worker = generation.read_bound(audit['worker'])
        require(parent['method'] == method and parent['success'] is False and len(worker['cases']) == 10
                and parent['owned_tree_released'] is True and parent['GPU_released'] is True
                and parent['final_owned_processes'] == [] and parent['final_session_members'] == [],
                'Latest failed allocation release/coverage is unproven')
        measured = audit['allocated_elapsed_seconds'] if method == 'base' else audit['actual_new_allocated_seconds']
        require(measured == parent['elapsed_seconds'] and measured > 0
                and abs(parent['end_monotonic'] - parent['start_monotonic'] - measured) < .001,
                'Actual parent allocation timing mismatch')
        require(len(parent['release_queries']) >= 2 and all(q['pids'] == [] and q['returncode'] == 0
                and q['command'] == generation.QUERY_COMMAND for q in parent['release_queries'][-2:]),
                'Original raw release queries missing')
    totals = {'base': values['base']['allocated_elapsed_seconds'],
              'alg': generation.recovery.PRIOR['alg'][2] + values['alg']['actual_new_allocated_seconds'],
              'dymos': generation.recovery.PRIOR['dymos'][2]}
    require(totals == PRIOR_SECONDS, 'Pinned actual failed attempts and all16 old upper must be charged once')
    return old, recent, totals


def validate(value):
    require(value['schema'] == SCHEMA and value['execution_id'] == EXECUTION_ID
            and value['user_authorized_recovery_attempts'] == 1 and value['retry_after_failure'] is False
            and value['GPU_authorized'] is False and value['status'] == 'BLOCKED', 'One CPU amendment, not execution authorization')
    require(value['phase_seconds'] == PHASES and value['shard_caps_seconds'] == CAPS,
            'Fixed recovery phase/cap amendment drift')
    require(value['videos_per_method'] == 188 and value['seeds'] == [0, 1]
            and value['cases_sha256'] == generation.dev.CASES_SHA256 and value['call'] == generation.smoke.SPEC,
            'Same full94 x two seeds/spec required')
    require(value['old_video_reuse_for_strict_comparison'] is False
            and value['preserved_complete_diagnostic_videos'] == {'base': 10, 'alg': 10, 'dymos': 0},
            'Preserve old videos; do not pretend mixed execution versions are a strict fresh comparison')
    for method in generation.METHODS:
        prior = value['prior_generation_charge_seconds'][method]
        require(generation.recovery.finite(prior) and prior == PRIOR_SECONDS[method]
                and len(CAPS[method]) == 16 and prior + sum(CAPS[method]) <= PHASES[method]['generation'],
                'Prior failures plus new caps exceed original generation budget')
        expected = PHASES[method]['generation'] - prior
        require(value['remaining_generation_seconds'][method] == expected, 'Remaining cost cannot be reset or moved')
    require(value['quality_pass'] is False and value['sota_claim'] is False, 'No quality claim from budget planning')
    return value


def prepare(output, authorization, approved_authorization_sha, root=ROOT):
    root = Path(root).resolve()
    auth = record(authorization, nonempty=True)
    require(auth['sha256'] == approved_authorization_sha, 'Explicit approved once-only user authorization required')
    proof = generation.read_bound(auth)
    require(proof.get('schema') == 'teacher_one_recovery_user_authorization_v1'
            and proof.get('attempts') == 1 and proof.get('budget_increase') is False
            and proof.get('preserve_failed_costs') is True, 'One recovery and no added budget must be authorized')
    old, recent, prior = audit_inputs(root)
    value = dict(schema=SCHEMA, private=True, execution_id=EXECUTION_ID, user_authorized_recovery_attempts=1,
                 authorization=auth, old_prior_audits=old, recent_attempt_audits=recent,
                 prior_generation_charge_seconds=prior,
                 remaining_generation_seconds={m: PHASES[m]['generation'] - prior[m] for m in generation.METHODS},
                 shard_caps_seconds=CAPS, phase_seconds=PHASES,
                 cases_sha256=generation.dev.CASES_SHA256, call=dict(generation.smoke.SPEC), seeds=[0, 1],
                 videos_per_method=188, old_video_reuse_for_strict_comparison=False,
                 preserved_complete_diagnostic_videos={'base': 10, 'alg': 10, 'dymos': 0},
                 status='BLOCKED', GPU_authorized=False, retry_after_failure=False,
                 gates=['New execution-specific sources/models/runtime/content contract and public test/byte gates',
                        'All16 remaining budget reservations including full lifecycle and130-second tail',
                        'Controlled original12-job formal canary completes in revised cap; no quality lookahead',
                        'Real parent/worker clocks and own-local generation SHA checks before CUDA',
                        'Independent exact coverage and review; no AR/Student/distillation'],
                 source=record(__file__, nonempty=True), quality_pass=False, sota_claim=False)
    validate(value)
    output = Path(output).absolute()
    require(not output.exists() and output.resolve() == output, 'Do not overwrite any prior amendment')
    generation.dev.exclusive(output, value)
    return output, sha(output.read_bytes())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--authorization', required=True, type=Path)
    p.add_argument('--approved-authorization-sha256', required=True)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args(argv)
    path, digest = prepare(args.output, args.authorization, args.approved_authorization_sha256)
    print(json.dumps(dict(path=str(path), sha256=digest, status='BLOCKED', GPU_used=False)))


if __name__ == '__main__':
    main()
