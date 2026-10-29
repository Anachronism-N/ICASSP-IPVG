"""CPU budget-only tests; no runtime/GPU authorization from a recovery plan."""
import copy
import json
import tempfile
from unittest import mock
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


class ExecutionPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.budget=fixture()
        self.budget_patch=mock.patch.object(recovery,'validate_budget_record',return_value=self.budget)
        self.budget_patch.start();self.addCleanup(self.budget_patch.stop)
        source=self.root/'synthetic-source.py';source.write_text('pass\n')
        patch=mock.patch.object(recovery,'sources',return_value=[source])
        patch.start();self.addCleanup(patch.stop)

    def plan(self,method):
        cases=[dict(id=f'case{i:03d}') for i in range(94)]
        ranks=[0,3] if method in ('base','dymos') else [1,2]
        caps=recovery.CAPS[method]
        budget=dict(path=str(self.root/'budget.json'),bytes=1,sha256='b'*64)
        prior=dict(actual_seconds=recovery.PRIOR_SECONDS['base'] if method=='base' else None,
                   upper_seconds=recovery.PRIOR_SECONDS[method],recovery_budget=budget)
        if method!='base':prior['audit']=dict(sha256=recovery.generation.recovery.PRIOR[method][1])
        return dict(schema=recovery.PLAN_SCHEMA,private=True,execution_id='recovery_a2',root=str(self.root),method=method,
          GPU_used=False,GPU_authorized=False,status='BLOCKED',quality_pass=False,sota_claim=False,
          phase_seconds=recovery.PHASES[method],total_seconds=90000 if method=='base' else recovery.generation.dev.TOTAL_SECONDS,
          prior_generation=prior,recovery_budget=budget,shard_cap_seconds=caps,nshards=16,seeds=[0,1],video_count=188,
          call=dict(recovery.generation.smoke.SPEC),fps=24,checkpoint='NONE',parameters={} if method=='base' else recovery.generation.dev.PARAMS[method],
          active_steps=0 if method=='base' else 10,method_context_installed=method!='base',node_ranks=ranks,
          allocations=[dict(shard=i*8+g,node_rank=r,gpu_index=g,allocated_gpus=1,cap_seconds=caps[i*8+g]) for i,r in enumerate(ranks) for g in range(8)],
          output_root=str(recovery.output_root(self.root,method)),cases_file=dict(sha256=recovery.generation.dev.CASES_SHA256),cases=cases,
          common_contract_frozen=True,jobs=[dict(j,output=str(recovery.output_root(self.root,method)/f"seed{j['seed']}/videos/{method}/{j['case_id']}.mp4"))
            for j in recovery.generation.dev.jobs_for(cases,self.root,method)],
          source_files=[dict(path=str(p)) for p in recovery.sources(self.root,method)],historical_base_reuse_allowed=False,
          old_diagnostic_media_reused=False,smoke_charged_seconds=1500)

    def test_new_contract_and_physical_allocation_preserve_every_failed_cost(self):
        import run_teacher_base as base
        import prepare_teacher_comparison as compare
        for method in recovery.generation.METHODS:
            p=self.plan(method);self.assertEqual(recovery.validate_plan(p),p)
            self.assertEqual(recovery.generation.execution_id(p),'recovery_a2')
            self.assertEqual(recovery.generation.fixed_allocations(p),p['allocations'])
            self.assertEqual(recovery.generation.prior_charge(p),recovery.PRIOR_SECONDS[method])
            self.assertEqual(compare.plan_prior(p),recovery.PRIOR_SECONDS[method])
            self.assertEqual(compare.plan_location(p),recovery.output_root(self.root,method)/'plan.blocked.json')
            if method=='base':self.assertEqual(base.validate_contract(p),p)
            else:self.assertEqual(recovery.generation.recovery.validate_plan(p),p)

    def test_new_plan_rejects_old_root_caps_reset_budget_and_mixed_media(self):
        for method in recovery.generation.METHODS:
            p=self.plan(method)
            for mutate in (lambda q:q.update(execution_id='recovery_a1'),lambda q:q.update(output_root=str(self.root/'old')),
                           lambda q:q['shard_cap_seconds'].__setitem__(0,5001),lambda q:q['prior_generation'].update(upper_seconds=0),
                           lambda q:q.update(old_diagnostic_media_reused=True),lambda q:q['allocations'][0].update(gpu_index=7)):
                q=copy.deepcopy(p);mutate(q)
                with self.assertRaises(ValueError):recovery.validate_plan(q)

    def test_new_cost_summary_keeps_prior_when_every_parent_is_unknown(self):
        p=self.plan('base');row=recovery.generation.record(self._write('p.json',p),nonempty=True)
        e=dict(plans={'base':row},allocations={'base':p['allocations']})
        v=recovery.generation.collect_costs(e,'base',{})
        self.assertEqual(v['generation_charged_upper_seconds'],recovery.PRIOR_SECONDS['base']+sum(recovery.CAPS['base']))
        self.assertIsNone(v['new_actual_seconds'])
        self.assertEqual(len(v['unknown_shards']),16)
        self.assertFalse(v['budget_exceeded'])

    def _write(self,name,v):
        p=self.root/name;p.write_bytes(recovery.encoded(v));return p

    def test_live_sidecar_requires_one_explicit_budget_even_if_cli_gpu_flag_true(self):
        from run_teacher_generation import validate_authorization
        budget=dict(path='actualbudget',sha256='b'*64)
        value=dict(execution_id='recovery_a2',authorization={'path':'auth'},comparison={'sha256':'c'*64},allocations={},recovery_budget=budget)
        auth=dict(schema='teacher_generation_authorization_v1',GPU_authorized=True,execution_id='recovery_a2',methods=list(recovery.generation.METHODS),
                  comparison=value['comparison'],allocations={},canary_shard=0,retry_allowed=False,borrow_allowed=False)
        with mock.patch.object(recovery.generation,'read_bound',return_value=auth),self.assertRaises(ValueError):validate_authorization(value)
        auth['recovery_budget']=budget
        with mock.patch.object(recovery.generation,'read_bound',return_value=auth):self.assertEqual(validate_authorization(value),auth)


if __name__=='__main__':unittest.main()
