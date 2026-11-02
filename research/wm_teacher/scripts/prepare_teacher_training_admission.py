#!/usr/bin/env python3
"""CPU-only freeze of one Teacher candidate; never launches training or imports torch."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
AUTHORIZATION = ROOT / 'eval/diag/teacher_training_admission_authorization_20261101.json'
SCHEMA = 'teacher_training_admission_cpu_v1'
CAPS = {'label_rebuild_and_cache': 24, 'single_gpu_training_resume': 2,
        'multi_node_technical_validation': 6}
AUTH_EXPECTED = {
    'schema': 'teacher_training_admission_authorization_v1',
    'user_choice': '批准32 GPUh准入',
    'all_nodes_exclusive_to_current_experiment': True,
    'stage_caps_GPUh': CAPS, 'total_cap_GPUh': 32,
    'phase_borrowing_allowed': False, 'failed_allocations_charged': True,
    'supervised_training_data_gate_required': True, 'independent_semantic_review_required': True,
    'formal_candidate_training_budget_approved': False, 'AR_allowed': False,
    'student_or_distillation_allowed': False, 'previous_experiment_budgets_reused': False,
    'training_started': False,
}
SOURCE_FILES = ('prepare_teacher_training_admission.py', 'train_e3_lora.py',
                'run_teacher_training_smoke.py', 'run_teacher_mechanism_probe.py')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def small_record(path):
    path = Path(path).resolve(strict=True)
    require(path.is_file() and path.stat().st_size <= 1024 * 1024, 'Small proof/source file required; no model hashing')
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def read_json(path):
    record = small_record(path)
    data = Path(record['path']).read_bytes()
    require(hashlib.sha256(data).hexdigest() == record['sha256'], 'Input changed while reading')
    return json.loads(data), record


def validate_authorization(path=AUTHORIZATION):
    value, record = read_json(path)
    # Canonical byte comparison preserves bool/int distinctions; the actual raw bytes are also bound.
    require(canonical(value) == canonical(AUTH_EXPECTED), 'Authorization differs from the approved 24+2+6 contract')
    return record


def validate_candidate(candidate):
    keys = {'seed', 'max_steps', 'horizon', 'warmup', 'lr', 'weight_decay', 'clip',
            'anchor_lambda', 'anchor_weighting', 'offload', 'caption_version',
            'lora_rank', 'lora_alpha', 'per_rank_batch', 'grad_accum'}
    require(isinstance(candidate, dict) and set(candidate) == keys, 'Exactly one frozen candidate required; no arm/grid fields')
    for key in ('seed', 'max_steps', 'horizon', 'warmup', 'lora_rank', 'lora_alpha', 'per_rank_batch', 'grad_accum'):
        require(type(candidate[key]) is int, 'Integer candidate field required: ' + key)
    require(candidate['max_steps'] == candidate['horizon'] > candidate['warmup'] >= 0, 'Freeze one max_steps/horizon above warmup')
    for key in ('lr', 'weight_decay', 'clip', 'anchor_lambda'):
        require(type(candidate[key]) in (int, float) and math.isfinite(candidate[key]), 'Finite candidate field required: ' + key)
    require(candidate['lr'] > 0 and candidate['clip'] > 0 and candidate['weight_decay'] >= 0
            and candidate['anchor_lambda'] >= 0, 'Bad candidate optimizer/lambda')
    require(candidate['anchor_weighting'] in ('training_weight', 'uniform') and type(candidate['offload']) is bool,
            'Native weighting/offload required')
    require(type(candidate['caption_version']) is str and bool(candidate['caption_version']), 'One caption version required')
    require(candidate['lora_rank'] == candidate['lora_alpha'] == 32, 'Native existing LoRA rank/alpha only; no full SFT')
    require(candidate['per_rank_batch'] == candidate['grad_accum'] == 1, 'Existing bs1/accum1 trajectory only')
    return dict(candidate)


def phase_bounds(phase, gpu_hours, cards, *, resume=False):
    require(phase in CAPS, 'Unknown/legacy/formal phase is not authorized')
    require(type(gpu_hours) in (int, float) and math.isfinite(gpu_hours) and 0 < gpu_hours <= CAPS[phase],
            'Phase hard cap exceeded; no borrowing or prior budget reuse')
    require(type(cards) is int and cards > 0, 'Positive allocated-card count required')
    expected = {'single_gpu_training_resume': 1, 'multi_node_technical_validation': 32}
    if phase in expected:
        require(cards == expected[phase], 'Phase allocation topology differs')
    else:
        require(cards <= 32, 'Label/cache allocation exceeds four-node resources')
    require(not resume, 'Unsigned resume denied; new all-rank resume proof and remaining-phase authorization required')
    return {'phase': phase, 'cap_GPUh': CAPS[phase], 'reserved_GPUh': float(gpu_hours),
            'allocated_cards': cards, 'global_wall_upper_seconds': gpu_hours * 3600 / cards,
            'charged_from': 'node-local supervisor before spawn/rendezvous/model load',
            'charged_until': 'final release including checkpoint/write/verify/cleanup grace',
            'failed_allocations_charged': True, 'phase_borrowing_allowed': False}


def allocated_cost_bound(nodes, *, expected_nodes=4, local_cards=8):
    """Sum allocated cards x same-host duration; global max is a conservative upper bound."""
    require(type(expected_nodes) is int and expected_nodes > 0 and type(local_cards) is int and local_cards > 0,
            'Invalid allocation topology')
    require(isinstance(nodes, list) and len(nodes) == expected_nodes, 'Every allocated node must close its ledger')
    require({n.get('node_rank') for n in nodes} == set(range(expected_nodes)), 'Missing/duplicate/foreign node ledger')
    durations = []
    for n in nodes:
        require(type(n.get('node_rank')) is int and type(n.get('cards')) is int and n['cards'] == local_cards,
                'Node allocation differs')
        require(n.get('scope') == 'pre_spawn_through_final_release' and n.get('released') is True,
                'Spawn/model load/rendezvous/cleanup/final release must all be charged')
        require(type(n.get('clock_id')) is str and bool(n['clock_id']), 'Same-host clock identity required')
        a, b = n.get('started_monotonic'), n.get('released_monotonic')
        require(all(type(v) in (int, float) and math.isfinite(v) for v in (a, b)) and b >= a,
                'Invalid same-host duration')
        durations.append(b - a)
    exact = sum(local_cards * d for d in durations) / 3600
    upper = expected_nodes * local_cards * max(durations) / 3600
    return {'allocated_GPUh': exact, 'global_wall_upper_seconds': max(durations),
            'allocated_GPUh_upper': upper, 'cards': expected_nodes * local_cards}


def verify_frozen_inputs(plan):
    require(plan.get('schema') == SCHEMA, 'Old plan/manifest version denied')
    require(canonical(plan.get('authorization_contract')) == canonical(AUTH_EXPECTED), 'Frozen authorization contract differs')
    records = [plan['authorization'], *plan['source'].values(), *plan['evidence'].values()]
    for record in records:
        require(small_record(record['path']) == record, 'Frozen input bytes/source changed: ' + str(record.get('path')))
    require(hashlib.sha256(canonical(plan['candidate'])).hexdigest() == plan['candidate_sha256'], 'Candidate changed after freeze')
    validate_candidate(plan['candidate'])
    require(plan.get('execution_allowed') is False and plan.get('formal_training_allowed') is False,
            'CPU plan cannot authorize execution')
    return True


def prepare(candidate, *, authorization=AUTHORIZATION, label_manifest=None, cache_manifest=None,
            semantic_approval=None, model_runtime_proof=None):
    auth = validate_authorization(authorization)
    candidate = validate_candidate(candidate)
    evidence = {}
    blockers = []
    for key, path in (('label_manifest', label_manifest), ('cache_manifest', cache_manifest),
                      ('semantic_approval', semantic_approval), ('model_runtime_proof', model_runtime_proof)):
        if path is None:
            blockers.append('missing_' + key)
        else:
            value, evidence[key] = read_json(path)
            require(isinstance(value, dict) and type(value.get('schema')) is str, 'Versioned proof required: ' + key)
            if key in ('label_manifest', 'cache_manifest'):
                require(value['schema'].endswith('_v2'), 'Old v1 label/cache manifest denied')
            # Binding bytes is not semantic/model/runtime validation, so presence alone never passes a gate.
            blockers.append('independent_validation_required_' + key)
    blockers += ['missing_actual_single_gpu_four_update_checkpoint_restore_proof',
                 'missing_actual_32_gpu_nccl_validation_proof',
                 'missing_all_rank_rng_optimizer_next_order_resume_trajectory_proof',
                 'missing_pre_spawn_through_final_release_node_local_allocation_ledgers',
                 'missing_frozen_runtime_source_user_review', 'formal_candidate_training_budget_not_approved']
    phases = {p: phase_bounds(p, CAPS[p], {'label_rebuild_and_cache': 32,
                                         'single_gpu_training_resume': 1, 'multi_node_technical_validation': 32}[p])
              for p in CAPS}
    plan = {'schema': SCHEMA, 'status': 'BLOCKED', 'execution_allowed': False, 'formal_training_allowed': False,
            'authorization': auth, 'authorization_contract': AUTH_EXPECTED,
            'source': {f: small_record(Path(__file__).parent / f) for f in SOURCE_FILES},
            'evidence': evidence, 'candidate': candidate, 'candidate_sha256': hashlib.sha256(canonical(candidate)).hexdigest(),
            'phase_bounds': phases, 'blockers': blockers,
            'teacher': {'model': 'Wan2.2-TI2V-5B', 'bidirectional': True, 'single_candidate': True,
                        'AR': False, 'student_or_distillation': False, 'full_SFT': False, 'score_based_parameter_selection': False},
            'technical_validation': {'single_gpu_updates': 4, 'multi_node_updates': 4,
                                     'offload_modes': ['on' if candidate['offload'] else 'off'], 'synthetic_inputs_allowed': True,
                                     'synthetic_proves_label_quality': False, 'measured_throughput_available': False,
                                     'four_updates_fit_phase_cap_guaranteed': False,
                                     'single_gpu_effective_batch': 1, 'multi_node_effective_batch': 32,
                                     'single_vs_multi_trajectory_equality_claim': False,
                                     'comparison': 'CPU mean-gradient reference only; different effective batches do not imply equal trajectories'},
            'resume_contract': {'current_checkpoint': 'rank0-only RNG; NOT sufficient for distributed resume',
                                'required_ranks': list(range(32)),
                                'per_rank_required': ['rank_node_local_mapping', 'CPU_RNG', 'local_CUDA_RNG',
                                                      'optimizer_adapter_hashes', 'next_update', 'next_sample_order',
                                                      'data_manifest_sha256', 'candidate_sha256'],
                                'true_proof': 'uninterrupted vs save/restart next updates on every rank with identical frozen inputs',
                                'rank0_rng_reused_for_other_ranks_allowed': False},
            'runtime_gate': 'CPU-only plan never releases trainer four-node hard gate; future real proofs and independent review required'}
    verify_frozen_inputs(plan)
    return plan


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--authorization', type=Path, default=AUTHORIZATION)
    p.add_argument('--candidate', type=Path, required=True)
    for name in ('label-manifest', 'cache-manifest', 'semantic-approval', 'model-runtime-proof'):
        p.add_argument('--' + name, type=Path)
    args = p.parse_args(argv)
    try:
        candidate, _ = read_json(args.candidate)
        plan = prepare(candidate, authorization=args.authorization, label_manifest=args.label_manifest,
                       cache_manifest=args.cache_manifest, semantic_approval=args.semantic_approval,
                       model_runtime_proof=args.model_runtime_proof)
        print(json.dumps(plan, sort_keys=True, indent=2, allow_nan=False))
        return 2  # Prepared but BLOCKED, never a runtime success receipt.
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(type(error).__name__ + ': ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
