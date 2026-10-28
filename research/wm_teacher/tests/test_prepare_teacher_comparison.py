"""Synthetic CPU common-contract tests, never generation/score PASS."""
import copy
import json
import sys
from pathlib import Path
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
from test_teacher_base import Fixture as BaseFixture, base, dev, smoke
import prepare_teacher_comparison as comparison
import prepare_teacher_recovery as recovery


class Fixture(BaseFixture):
    def setUp(self):
        super().setUp()
        for patch in (mock.patch.object(comparison, "base", base), mock.patch.object(comparison, "dev", dev),
                      mock.patch.object(comparison, "smoke", smoke)):
            patch.start()
            self.addCleanup(patch.stop)
        patch = mock.patch.object(recovery, "dev", dev)
        patch.start()
        self.addCleanup(patch.stop)
        patch = mock.patch.object(recovery, "PRIOR", dict(recovery.PRIOR))
        patch.start()
        self.addCleanup(patch.stop)
        for name in ("scripts/prepare_teacher_recovery.py", "tests/test_teacher_recovery.py"):
            (self.root / name).write_text("# Synthetic CPU recovery source\n")
        self.paths, self.shas = {}, {}
        self.paths["base"], self.shas["base"] = self.prepare_base()
        base_plan = json.loads(self.paths["base"].read_bytes())
        for method in ("alg", "dymos"):
            plan = copy.deepcopy(base_plan)
            history = {}
            for role in ("audit", "old_plan", "old_ledger"):
                path = self.base / f"synthetic-{method}-{role}.json"
                path.write_bytes(smoke.encoded(dict(synthetic_CPU_only=True, role=role, method=method)))
                history[role] = smoke.record(path)
            name, _, prior_upper = recovery.PRIOR[method]
            recovery.PRIOR[method] = (name, history["audit"]["sha256"], prior_upper)
            folder = recovery.plan_folder(self.root, method)
            plan.update(schema=recovery.SCHEMA, method=method, parameters=dev.PARAMS[method], active_steps=10,
                        phase_seconds=dev.PHASE_SECONDS, total_seconds=dev.TOTAL_SECONDS,
                        output_root=str(folder), source_files=[smoke.record(p) for p in recovery.sources(self.root)],
                        source=smoke.record(self.root / "scripts/prepare_teacher_recovery.py"),
                        execution_id=recovery.EXECUTION_ID, common_contract_frozen=True,
                        GPU_authorized=False, old_reservations_retired=False, historical_base_reuse_allowed=False,
                        status="BLOCKED", method_context_installed=True, allocations=recovery.allocations(method),
                        prior_generation=dict(actual_seconds=None, upper_seconds=prior_upper, **history),
                        smoke_charged_seconds=1560., shard_cap_seconds=[recovery.CAPS[method]]*16)
            plan["jobs"] = [dict(j, output=str(folder / f"seed{j['seed']}/videos/{method}/{j['case_id']}.mp4"))
                            for j in dev.jobs_for(plan["cases"], self.root, method)]
            folder.mkdir(mode=0o700)
            self.paths[method] = folder / "plan.blocked.json"
            smoke.exclusive(self.paths[method], plan)
            self.shas[method] = smoke.sha(self.paths[method].read_bytes())
        self.output_dir = self.base / "comparison"
        self.output_dir.mkdir(mode=0o700)
        self.output = self.output_dir / "comparison.BLOCKED.json"

    def prepare_comparison(self):
        return comparison.prepare(self.paths, self.shas, self.bundle, self.bundle_sha, self.output)

    def mutate_plan(self, method, change):
        plan = json.loads(self.paths[method].read_bytes())
        change(plan)
        self.paths[method].write_bytes(smoke.encoded(plan))
        self.shas[method] = smoke.sha(self.paths[method].read_bytes())


class ContractTests(Fixture):
    def test_same_content_plus_explicit_score_bundle_passes_cpu_contract_only(self):
        path, digest = self.prepare_comparison()
        value = comparison.verify(path, digest)
        self.assertEqual(value["status"], "BLOCKED")
        self.assertFalse(value["runnable"])
        self.assertFalse(value["GPU_used"])
        self.assertFalse(value["strict_baseline_reuse_allowed"])
        self.assertEqual(value["common_execution_sha256"], smoke.sha(smoke.encoded(value["common_execution"])))
        self.assertEqual(value["common_execution"]["score_bundle_sha256"], self.bundle_sha)
        self.assertEqual(value["phase_seconds"]["base"], dict(generation=72000, score=18000))
        self.assertEqual(value["interventions"]["base"], dict(parameters={}, active_steps=0, method_context_installed=False))

    def test_old_plan_without_score_bundle_rejected_without_retrofit(self):
        self.mutate_plan("alg", lambda p: p.pop("score_bundle"))
        original = self.paths["alg"].read_bytes()
        with self.assertRaisesRegex(ValueError, "explicitly bind"):
            self.prepare_comparison()
        self.assertEqual(self.paths["alg"].read_bytes(), original)
        self.assertFalse(self.output.exists())

    def test_same_commit_is_not_enough_for_runtime_or_model_drift(self):
        self.mutate_plan("dymos", lambda p: p.update(commit="same-commit"))
        self.mutate_plan("alg", lambda p: p["runtime"]["packages"].update(mock="different CPU runtime"))
        with self.assertRaisesRegex(ValueError, "runtime"):
            self.prepare_comparison()

    def test_wrong_model_digest_and_cfg_rejected(self):
        self.mutate_plan("alg", lambda p: p.update(model_content_sha256="f"*64))
        with self.assertRaisesRegex(ValueError, "model digest"):
            self.prepare_comparison()

    def test_case_prompt_seed_and_original_image_bindings_are_exact(self):
        for method in ("alg", "dymos"):
            self.mutate_plan(method, lambda p: p["cases"][0].update(prompt="different caption"))
        with self.assertRaisesRegex(ValueError, "frozen case"):
            self.prepare_comparison()

    def test_job_reordered_or_seed_substitution_rejected(self):
        self.mutate_plan("alg", lambda p: p["jobs"][0].update(seed=1))
        with self.assertRaisesRegex(ValueError, "job.*drift"):
            self.prepare_comparison()

    def test_missing_common_source_rejected(self):
        self.mutate_plan("alg", lambda p: p["source_files"].pop())
        with self.assertRaisesRegex(ValueError, "source set drift"):
            self.prepare_comparison()

    def test_common_source_current_bytes_drift_rejected(self):
        (self.root / "scripts/gen_jobs.py").write_text('NEG = "original negative"\n# drift\n')
        with self.assertRaisesRegex(ValueError, "drift"):
            self.prepare_comparison()

    def test_other_score_bundle_hash_rejected(self):
        self.mutate_plan("dymos", lambda p: p["score_bundle"].update(sha256="f"*64))
        with self.assertRaisesRegex(ValueError, "score bundle"):
            self.prepare_comparison()

    def test_historical_allowed_false_must_remain_false(self):
        self.mutate_plan("base", lambda p: p["historical_base_reuse"]["seed0"].update(allowed=True))
        with self.assertRaisesRegex(ValueError, "allowed=False"):
            self.prepare_comparison()

    def test_source_duplicate_and_unapproved_plan_sha_rejected(self):
        self.shas["alg"] = "f"*64
        with self.assertRaisesRegex(ValueError, "plan SHA"):
            self.prepare_comparison()

    def test_comparison_never_overwrites_and_manifest_drift_rejected(self):
        path, digest = self.prepare_comparison()
        original = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "never overwrite"):
            self.prepare_comparison()
        self.assertEqual(path.read_bytes(), original)
        value = json.loads(path.read_bytes())
        value["runnable"] = True
        path.write_bytes(smoke.encoded(value))
        with self.assertRaisesRegex(ValueError, "cannot become"):
            comparison.verify(path, smoke.sha(path.read_bytes()))

    def test_generation_completion_interface_is_closed_for_every_method(self):
        for method in comparison.METHODS:
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, "BLOCKED.*completion"):
                comparison.verify_generation_completion(Path("nonexistent synthetic receipt"), "a"*64,
                    common_execution_sha256="b"*64, score_bundle_sha256="c"*64, method=method)

    def test_method_generation_caps_over_24gpuh_rejected(self):
        self.mutate_plan("alg", lambda p: p.update(shard_cap_seconds=[5401]*16))
        with self.assertRaisesRegex(ValueError, "content contract drift"):
            self.prepare_comparison()

    def test_method_context_cannot_be_bypassed(self):
        self.mutate_plan("dymos", lambda p: p.update(method_context_installed=False))
        with self.assertRaisesRegex(ValueError, "content contract drift"):
            self.prepare_comparison()

    def test_old_devset_schema_is_historical_only_not_strict_candidate(self):
        self.mutate_plan("alg", lambda p: p.update(schema=dev.SCHEMA))
        with self.assertRaisesRegex(ValueError, "old devset schema/root"):
            self.prepare_comparison()

    def test_old_candidate_root_rejected_even_with_new_schema(self):
        self.mutate_plan("alg", lambda p: p.update(output_root=str(dev.output_root(self.root, "alg"))))
        with self.assertRaisesRegex(ValueError, "new recovery root"):
            self.prepare_comparison()

    def test_old_prior_cost_or_retirement_cannot_be_zeroed(self):
        original = self.paths["dymos"].read_bytes()
        for change in (lambda p: p["prior_generation"].update(upper_seconds=0),
                       lambda p: p["prior_generation"].update(actual_seconds=0),
                       lambda p: p.update(old_reservations_retired=True),
                       lambda p: p.update(common_contract_frozen=False)):
            self.paths["dymos"].write_bytes(original)
            self.mutate_plan("dymos", change)
            with self.assertRaises(ValueError):
                self.prepare_comparison()

    def test_recovery_prior_plus_caps_are_explicit_and_preserved(self):
        path, digest = self.prepare_comparison()
        value = comparison.verify(path, digest)
        self.assertEqual(value["generation_charged_upper_seconds"]["base"], 72000)
        for method in ("alg", "dymos"):
            expected = recovery.PRIOR[method][2] + 16*recovery.CAPS[method]
            self.assertEqual(value["generation_charged_upper_seconds"][method], expected)
            self.assertLessEqual(expected, 86400)
            self.assertIsNone(value["prior_generation"][method]["actual_seconds"])

    def test_approved_score_content_api_failure_is_not_ignored(self):
        self.scoring.verify_bundle.side_effect = ValueError("recursive score dependency/model coverage drift")
        with self.assertRaisesRegex(ValueError, "coverage drift"):
            self.prepare_comparison()


if __name__ == "__main__":
    unittest.main()
