"""CPU-only admission contract tests; no torch/model/GPU import required."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import prepare_teacher_training_admission as admission


def candidate():
    return {'seed': 20261021, 'max_steps': 1000, 'horizon': 1000, 'warmup': 20,
            'lr': 5e-5, 'weight_decay': 0.01, 'clip': 1.0, 'anchor_lambda': 0.0,
            'anchor_weighting': 'training_weight', 'offload': True, 'caption_version': 'teacher_v2',
            'lora_rank': 32, 'lora_alpha': 32, 'per_rank_batch': 1, 'grad_accum': 1}


class AdmissionTests(unittest.TestCase):
    def authorization(self, root, value=None):
        path = root / 'auth.json'
        path.write_text(json.dumps(admission.AUTH_EXPECTED if value is None else value))
        return path

    def test_single_candidate_blocked_contract_and_source_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            auth = self.authorization(Path(tmp))
            plan = admission.prepare(candidate(), authorization=auth)
            self.assertEqual(plan['status'], 'BLOCKED')
            self.assertFalse(plan['execution_allowed'] or plan['formal_training_allowed'])
            self.assertEqual(sum(p['cap_GPUh'] for p in plan['phase_bounds'].values()), 32)
            self.assertEqual(plan['phase_bounds']['multi_node_technical_validation']['global_wall_upper_seconds'], 675)
            self.assertEqual(plan['phase_bounds']['single_gpu_training_resume']['global_wall_upper_seconds'], 7200)
            self.assertIn('missing_semantic_approval', plan['blockers'])
            self.assertIn('missing_all_rank_rng_optimizer_next_order_resume_trajectory_proof', plan['blockers'])
            self.assertEqual(plan['resume_contract']['required_ranks'], list(range(32)))
            tech = plan['technical_validation']
            self.assertFalse(tech['four_updates_fit_phase_cap_guaranteed'])
            self.assertFalse(tech['single_vs_multi_trajectory_equality_claim'])
            self.assertFalse(tech['synthetic_proves_label_quality'])
            self.assertTrue(admission.verify_frozen_inputs(plan))
            auth.write_text(json.dumps(admission.AUTH_EXPECTED, indent=2))
            with self.assertRaisesRegex(ValueError, 'bytes/source changed'):
                admission.verify_frozen_inputs(plan)

    def test_authorization_flags_caps_and_bool_types_are_exact(self):
        changes = [('total_cap_GPUh', 33), ('phase_borrowing_allowed', True),
                   ('formal_candidate_training_budget_approved', True), ('AR_allowed', True),
                   ('student_or_distillation_allowed', True), ('failed_allocations_charged', False),
                   ('previous_experiment_budgets_reused', True), ('all_nodes_exclusive_to_current_experiment', 1)]
        with tempfile.TemporaryDirectory() as tmp:
            for key, value in changes:
                auth = copy.deepcopy(admission.AUTH_EXPECTED)
                auth[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    admission.validate_authorization(self.authorization(Path(tmp), auth))
            auth = copy.deepcopy(admission.AUTH_EXPECTED)
            auth['stage_caps_GPUh']['single_gpu_training_resume'] = 3
            with self.assertRaises(ValueError):
                admission.validate_authorization(self.authorization(Path(tmp), auth))

    def test_no_phase_borrowing_and_unsigned_resume_or_formal_phase(self):
        for phase, hours, cards, resume in [('single_gpu_training_resume', 2.01, 1, False),
                                           ('multi_node_technical_validation', 6.01, 32, False),
                                           ('label_rebuild_and_cache', 24.01, 32, False),
                                           ('single_gpu_training_resume', 1, 8, False),
                                           ('multi_node_technical_validation', 1, 8, False),
                                           ('multi_node_technical_validation', 1, 32, True),
                                           ('formal_training', 1, 32, False),
                                           ('single_gpu_training_resume', float('nan'), 1, False)]:
            with self.subTest(phase=phase, hours=hours, cards=cards, resume=resume), self.assertRaises(ValueError):
                admission.phase_bounds(phase, hours, cards, resume=resume)

    def test_old_manifest_v1_denied_and_present_evidence_never_self_approves(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            auth = self.authorization(root)
            proof = root / 'proof.json'
            proof.write_text(json.dumps({'schema': 'teacher_label_manifest_v1', 'approved': True}))
            with self.assertRaisesRegex(ValueError, 'v1'):
                admission.prepare(candidate(), authorization=auth, label_manifest=proof)
            proof.write_text(json.dumps({'schema': 'teacher_label_manifest_v2', 'approved': True}))
            plan = admission.prepare(candidate(), authorization=auth, label_manifest=proof, cache_manifest=proof,
                                     semantic_approval=proof, model_runtime_proof=proof)
            self.assertFalse(plan['execution_allowed'])
            self.assertIn('independent_validation_required_semantic_approval', plan['blockers'])
            proof.write_text(json.dumps({'schema': 'teacher_label_manifest_v2', 'approved': False}))
            with self.assertRaises(ValueError):
                admission.verify_frozen_inputs(plan)

    def test_frozen_single_candidate_no_grid_and_tamper_rejected(self):
        changes = [('arms', ['a', 'c']), ('horizon', 500), ('max_steps', True), ('lr', float('nan')),
                   ('anchor_lambda', -1), ('lora_rank', 16), ('grad_accum', 32)]
        for key, value in changes:
            c = candidate()
            c[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                admission.validate_candidate(c)
        with tempfile.TemporaryDirectory() as tmp:
            plan = admission.prepare(candidate(), authorization=self.authorization(Path(tmp)))
            plan['candidate']['lr'] *= 2
            with self.assertRaisesRegex(ValueError, 'Candidate changed'):
                admission.verify_frozen_inputs(plan)

    def test_model_sized_proof_is_refused_before_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'not_a_small_proof'
            with path.open('wb') as f:
                f.truncate(1024 * 1024 + 1)
            with self.assertRaisesRegex(ValueError, 'no model hashing'):
                admission.small_record(path)

    def test_allocated_sum_and_32_global_wall_bound_use_node_local_offsets(self):
        nodes = [{'node_rank': r, 'cards': 8, 'scope': 'pre_spawn_through_final_release',
                  'released': True, 'clock_id': 'host%d-clock' % r,
                  'started_monotonic': r * 10000000 + 100, 'released_monotonic': r * 10000000 + 100 + d}
                 for r, d in enumerate([500, 550, 600, 650])]
        bound = admission.allocated_cost_bound(nodes)
        self.assertAlmostEqual(bound['allocated_GPUh'], 8 * sum([500, 550, 600, 650]) / 3600)
        self.assertAlmostEqual(bound['allocated_GPUh_upper'], 32 * 650 / 3600)
        self.assertGreaterEqual(bound['allocated_GPUh_upper'], bound['allocated_GPUh'])
        for edit in ('duplicate', 'missing', 'unreleased', 'spawn_excluded', 'wrong_cards', 'backward_clock'):
            bad = copy.deepcopy(nodes)
            if edit == 'duplicate':
                bad[3]['node_rank'] = 2
            elif edit == 'missing':
                bad.pop()
            elif edit == 'unreleased':
                bad[0]['released'] = False
            elif edit == 'spawn_excluded':
                bad[0]['scope'] = 'trainer_only'
            elif edit == 'wrong_cards':
                bad[0]['cards'] = 1
            else:
                bad[0]['released_monotonic'] = 0
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                admission.allocated_cost_bound(bad)

    def test_cli_help_and_blocked_prepare_do_not_import_torch(self):
        code = ("import prepare_teacher_training_admission as a,sys\n"
                "try: rc=a.main(sys.argv[1:])\nexcept SystemExit as e: rc=e.code\n"
                "assert 'torch' not in sys.modules\nsys.exit(rc)")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(SCRIPTS))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = root / 'candidate.json'
            c.write_text(json.dumps(candidate()))
            auth = self.authorization(root)
            for argv, expected in [(['--help'], 0), (['--candidate', str(c), '--authorization', str(auth)], 2)]:
                p = subprocess.run([sys.executable, '-c', code, *argv], env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(p.returncode, expected, p.stderr)
                if expected == 2:
                    self.assertEqual(json.loads(p.stdout)['status'], 'BLOCKED')


if __name__ == '__main__':
    unittest.main()
