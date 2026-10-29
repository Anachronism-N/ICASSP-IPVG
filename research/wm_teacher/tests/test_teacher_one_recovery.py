"""CPU budget-only tests; no runtime/GPU authorization from a recovery plan."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import prepare_teacher_one_recovery as recovery


def fixture():
    return dict(schema=recovery.SCHEMA, execution_id='recovery_a2', user_authorized_recovery_attempts=1,
                retry_after_failure=False, GPU_authorized=False, status='BLOCKED',
                phase_seconds=copy.deepcopy(recovery.PHASES), shard_caps_seconds=copy.deepcopy(recovery.CAPS),
                videos_per_method=188, seeds=[0,1], cases_sha256=recovery.generation.dev.CASES_SHA256,
                call=dict(recovery.generation.smoke.SPEC), old_video_reuse_for_strict_comparison=False,
                preserved_complete_diagnostic_videos={'base':10,'alg':10,'dymos':0},
                prior_generation_charge_seconds=dict(recovery.PRIOR_SECONDS),
                remaining_generation_seconds={m:recovery.PHASES[m]['generation']-s for m,s in recovery.PRIOR_SECONDS.items()},
                quality_pass=False,sota_claim=False)


class BudgetTests(unittest.TestCase):
    def test_all_new_caps_plus_prior_failures_stay_in_original_stages(self):
        v=fixture();self.assertEqual(recovery.validate(v),v)
        self.assertAlmostEqual(sum(v['shard_caps_seconds']['base'])/3600,18.77777777777778)
        for method in recovery.generation.METHODS:
            self.assertLessEqual(sum(v['shard_caps_seconds'][method])+v['prior_generation_charge_seconds'][method],v['phase_seconds'][method]['generation'])

    def test_zeroed_or_lowered_old_cost_is_rejected(self):
        for method in recovery.generation.METHODS:
            v=fixture();v['prior_generation_charge_seconds'][method]-=1
            v['remaining_generation_seconds'][method]+=1
            with self.assertRaises(ValueError):recovery.validate(v)

    def test_budget_growth_caps_drift_and_extra_retry_refused(self):
        for mutate in (lambda v:v['phase_seconds']['base'].update(generation=90000),
                       lambda v:v['shard_caps_seconds']['base'].__setitem__(0,4500),
                       lambda v:v.update(user_authorized_recovery_attempts=2),
                       lambda v:v.update(retry_after_failure=True),
                       lambda v:v.update(GPU_authorized=True)):
            v=fixture();mutate(v)
            with self.assertRaises(ValueError):recovery.validate(v)

    def test_earlier_videos_not_silently_mixed_or_deleted(self):
        v=fixture();v['old_video_reuse_for_strict_comparison']=True
        with self.assertRaises(ValueError):recovery.validate(v)
        v=fixture();v['preserved_complete_diagnostic_videos']['base']=0
        with self.assertRaises(ValueError):recovery.validate(v)

    def test_input_seed_step_and_quality_gate_unchanged(self):
        for mutate in (lambda v:v.update(seeds=[0,1,2]), lambda v:v.update(videos_per_method=178),
                       lambda v:v['call'].update(num_inference_steps=25),lambda v:v.update(quality_pass=True)):
            v=fixture();mutate(v)
            with self.assertRaises(ValueError):recovery.validate(v)


if __name__=='__main__':unittest.main()
