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
RECENT_AUDIT_SHAS = {'base': '00954bb0a1d0afadbc9b96176cc1ce59290b470600357b721a40a8932dc3c0cf',
                     'alg': 'e5da64d332c164ff8e7eeba48f4d9e777d675d6eef7d05b45d763932232076a1'}
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
    require(all(recent[m]['sha256']==RECENT_AUDIT_SHAS[m] for m in recent), 'Pinned recent actual failure audit drift')
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


PLAN_SCHEMA = 'teacher_one_recovery_plan_v1'


def output_root(root, method):
    require(method in generation.METHODS, 'Unknown recovery method')
    return Path(root) / f'eval/teacher_{method}_once_recovery_20261029'


def sources(root, method):
    from run_teacher_base import sources as base_sources
    original = base_sources(Path(root)) if method == 'base' else generation.recovery.sources(Path(root))
    return sorted(set(original) | {Path(root) / name for name in (
        'scripts/prepare_teacher_one_recovery.py', 'tests/test_teacher_one_recovery.py')})


def validate_budget_record(row):
    value = generation.read_bound(row)
    validate(value)
    authorization = generation.read_bound(value['authorization'])
    require(authorization.get('schema') == 'teacher_one_recovery_user_authorization_v1'
            and authorization.get('attempts') == 1 and authorization.get('budget_increase') is False
            and authorization.get('preserve_failed_costs') is True, 'Actual one-recovery authorization is required')
    old, recent, prior = audit_inputs(ROOT)
    require(value['old_prior_audits'] == old and value['recent_attempt_audits'] == recent
            and value['prior_generation_charge_seconds'] == prior, 'Actual failed-attempt budget evidence drift')
    return value


def validate_plan(plan):
    method = plan['method']
    require(method in generation.METHODS and plan['schema'] == PLAN_SCHEMA and plan['private'] is True
            and plan['execution_id'] == EXECUTION_ID and plan['GPU_used'] is False
            and plan['GPU_authorized'] is False and plan['status'] == 'BLOCKED'
            and plan['quality_pass'] is False and plan['sota_claim'] is False, 'New once-only CPU plan required')
    budget = validate_budget_record(plan['recovery_budget'])
    require(plan['phase_seconds'] == PHASES[method]
            and plan['total_seconds'] == (90000 if method == 'base' else generation.dev.TOTAL_SECONDS)
            and plan['prior_generation']['upper_seconds'] == PRIOR_SECONDS[method]
            and plan['prior_generation']['actual_seconds'] == (PRIOR_SECONDS['base'] if method == 'base' else None),
            'New recovery cannot omit or zero previously charged failures')
    require(plan['prior_generation']['recovery_budget'] == plan['recovery_budget']
            and plan['shard_cap_seconds'] == budget['shard_caps_seconds'][method]
            and plan['nshards'] == 16 and plan['seeds'] == [0,1] and plan['video_count'] == 188,
            'Fixed all16 reservations/full94x2 protocol drift')
    require(plan['call'] == generation.smoke.SPEC and plan['fps'] == 24 and plan['checkpoint'] == 'NONE'
            and plan['parameters'] == ({} if method == 'base' else generation.dev.PARAMS[method])
            and plan['active_steps'] == (0 if method == 'base' else 10)
            and plan['method_context_installed'] == (method != 'base'), 'Recovery intervention/spec drift')
    expected_ranks = [0,3] if method in ('base','dymos') else [1,2]
    expected = [dict(shard=i*8+gpu,node_rank=rank,gpu_index=gpu,allocated_gpus=1,cap_seconds=CAPS[method][i*8+gpu])
                for i,rank in enumerate(expected_ranks) for gpu in range(8)]
    require(plan['node_ranks'] == expected_ranks and plan['allocations'] == expected
            and PRIOR_SECONDS[method] + sum(CAPS[method]) <= PHASES[method]['generation'], 'Fixed owner/caps exceed remaining generation budget')
    root=Path(plan['root'])
    require(root.resolve()==root and plan['output_root']==str(output_root(root,method))
            and plan['cases_file']['sha256']==generation.dev.CASES_SHA256 and len(plan['cases'])==94
            and len({c['id'] for c in plan['cases']})==94 and plan['common_contract_frozen'] is True,
            'Strict fresh recovery inputs/root required')
    jobs=[dict(j,output=str(output_root(root,method)/f"seed{j['seed']}/videos/{method}/{j['case_id']}.mp4"))
          for j in generation.dev.jobs_for(plan['cases'],root,method)]
    require(plan['jobs']==jobs and len(plan['source_files'])==len(sources(root,method))
            and {r['path'] for r in plan['source_files']}=={str(p) for p in sources(root,method)}, 'Recovery source/job coverage drift')
    require(plan['historical_base_reuse_allowed'] is False and plan['old_diagnostic_media_reused'] is False,
            'Old videos must not be mixed into a new strict execution version')
    if method != 'base':
        require(plan['prior_generation']['audit']['sha256']==generation.recovery.PRIOR[method][1], 'Pinned oldest audit required')
        require(generation.recovery.finite(plan['smoke_charged_seconds'])
                and 0 < plan['smoke_charged_seconds'] <= PHASES[method]['smoke'], 'Existing smoke/performance fee binding missing')
    return plan


def prepare_plan(method, budget_path, approved_budget_sha, score_bundle, approved_score_sha, node_preflight, approved_node_sha):
    import run_teacher_base as base
    import score_teacher_devset as scoring
    root=ROOT.resolve()
    budget_row=record(budget_path,nonempty=True)
    require(budget_row['sha256']==approved_budget_sha, 'Approved actual recovery budget SHA required')
    validate_budget_record(budget_row)
    folder=output_root(root,method)
    require(not folder.exists() and folder.resolve()==folder, 'Unique new recovery root required; never overwrite')
    old_paths={'base':root/'eval/teacher_base_fresh_20261028/plan.BLOCKED.json',
               'alg':generation.recovery.plan_folder(root,'alg')/'plan.blocked.json',
               'dymos':generation.recovery.plan_folder(root,'dymos')/'plan.blocked.json'}
    old_record=record(old_paths[method],nonempty=True);old_plan=generation.read_bound(old_record)
    require(old_plan['method']==method and old_plan['GPU_used'] is False, 'Original frozen plan is evidence, not live GPU state')
    scoring.verify_bundle(score_bundle,approved_score_sha,verify_content=True)
    pf_row=record(node_preflight,nonempty=True)
    require(pf_row['sha256']==approved_node_sha,'Approved node preflight required')
    pf=generation.read_bound(pf_row);models=generation.smoke.models_from_preflight(pf)
    raw,cases,inputs,manifest=generation.dev.frozen_cases(root)
    negative=generation.smoke.generator_negative(root)
    ranks=[0,3] if method in ('base','dymos') else [1,2]
    prior=dict(actual_seconds=PRIOR_SECONDS['base'] if method=='base' else None,
               upper_seconds=PRIOR_SECONDS[method],recovery_budget=budget_row)
    if method!='base':
        prior.update({k:old_plan['prior_generation'][k] for k in ('audit','old_plan','old_ledger')})
    allocations=[dict(shard=i*8+gpu,node_rank=rank,gpu_index=gpu,allocated_gpus=1,cap_seconds=CAPS[method][i*8+gpu])
                 for i,rank in enumerate(ranks) for gpu in range(8)]
    value=dict(schema=PLAN_SCHEMA,private=True,execution_id=EXECUTION_ID,root=str(root),method=method,
        status='BLOCKED',GPU_authorized=False,GPU_used=False,quality_pass=False,sota_claim=False,checkpoint='NONE',
        parameters={} if method=='base' else dict(generation.dev.PARAMS[method]),call=dict(generation.smoke.SPEC),fps=24,
        active_steps=0 if method=='base' else 10,method_context_installed=method!='base',common_contract_frozen=True,
        phase_seconds=PHASES[method],total_seconds=90000 if method=='base' else generation.dev.TOTAL_SECONDS,
        seeds=[0,1],video_count=188,nshards=16,node_ranks=ranks,allocations=allocations,shard_cap_seconds=CAPS[method],
        recovery_budget=budget_row,prior_generation=prior,old_plan_evidence=old_record,
        historical_base_reuse_allowed=False,old_diagnostic_media_reused=False,
        historical_base_reuse=generation.dev.base_reuse(root,manifest),cases_file=manifest,cases=cases,inputs=inputs,
        jobs=[dict(j,output=str(folder/f"seed{j['seed']}/videos/{method}/{j['case_id']}.mp4"))
              for j in generation.dev.jobs_for(cases,root,method)],output_root=str(folder),
        negative_prompt=negative,negative_sha256=sha(negative.encode()),node_preflight=pf_row,
        models=models,model_content_sha256=pf['model_content_sha256'],runtime=generation.smoke.runtime_records(),
        source_files=[record(p) for p in sources(root,method)],score_bundle=record(score_bundle,nonempty=True),
        source=record(__file__,nonempty=True),smoke_charged_seconds=old_plan.get('smoke_charged_seconds',0),
        admission=dict(status='BLOCKED',native_throughput_PASS=False,runnable=False,reasons=base.BLOCKERS))
    require(value['score_bundle']['sha256']==approved_score_sha,'Scoring source bytes changed')
    validate_plan(value)
    folder.mkdir(mode=0o700)
    for seed in (0,1):
        seed_root=folder/f'seed{seed}';(seed_root/'videos'/method).mkdir(mode=0o700,parents=True)
        with (seed_root/'cases.jsonl').open('xb') as handle:
            import os
            os.chmod(handle.name,0o600);handle.write(raw);handle.flush();os.fsync(handle.fileno())
    path=folder/'plan.blocked.json';generation.dev.exclusive(path,value)
    return path,sha(path.read_bytes())


def main(argv=None):
    import os
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--authorization', type=Path)
    p.add_argument('--approved-authorization-sha256')
    p.add_argument('--method', choices=generation.METHODS)
    p.add_argument('--budget', type=Path)
    p.add_argument('--approved-budget-sha256')
    p.add_argument('--score-bundle', type=Path)
    p.add_argument('--approved-score-bundle-sha256')
    p.add_argument('--node-preflight', type=Path)
    p.add_argument('--approved-node-preflight-sha256')
    p.add_argument('--output', type=Path)
    args = p.parse_args(argv)
    if args.method:
        require(all((args.budget,args.approved_budget_sha256,args.score_bundle,args.approved_score_bundle_sha256,
                     args.node_preflight,args.approved_node_preflight_sha256)) and args.output is None,
                'New method preparation requires approved budget/score/node inputs; output path is fixed')
        path,digest=prepare_plan(args.method,args.budget,args.approved_budget_sha256,args.score_bundle,
                                args.approved_score_bundle_sha256,args.node_preflight,args.approved_node_preflight_sha256)
    else:
        require(args.authorization and args.approved_authorization_sha256 and args.output,
                'CPU budget preparation requires explicit authorization SHA and new output')
        path,digest=prepare(args.output,args.authorization,args.approved_authorization_sha256)
    print(json.dumps(dict(path=str(path),sha256=digest,status='BLOCKED',GPU_used=False)))

if __name__ == '__main__':
    main()
