"""Synthetic CPU common-contract tests, never generation/score PASS."""
import copy
import json
import types
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
        common_sources = {r["path"] for r in value["common_execution"]["common_sources"]}
        self.assertTrue({"scripts/verify_teacher_history.py", "tests/test_verify_teacher_history.py"} <= common_sources)
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

    def test_generation_completion_booleans_without_execution_proofs_are_rejected(self):
        for method in comparison.METHODS:
            path = self.output_dir / f"fake-{method}-completion.json"
            value = dict(schema="teacher_generation_completion_v1", method=method, private=True,
                generation_complete=True, GPU_used=True, quality_pass=False, sota_claim=False,
                common_execution_sha256="b"*64, score_bundle_sha256="c"*64)
            smoke.exclusive(path, value)
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, "PASS booleans"):
                comparison.verify_generation_completion(path, smoke.sha(path.read_bytes()),
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

    def case_fixture(self, method, index=0):
        plan = json.loads(self.paths[method].read_bytes())
        job = plan["jobs"][index]
        case = plan["cases"][index % 94]
        image = smoke.record(case["image"])
        gt = smoke.record(case["gt"])
        inputs = dict(case_id=case["id"], prompt_sha256=case["prompt_sha256"], image=image, gt=gt)
        params = dict(w=dict(shape=[2,2], dtype="torch.bfloat16", sha256="a"*64))
        video_path = Path(job["output"])
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"synthetic media; no GPU generation")
        if method == "base":
            stats = dict(model_forwards=100, scheduler_steps=50, progress_steps=50, finite_checks=1)
        else:
            stats = dict(observed=dict(model_forwards=110 if method == "alg" else 100, scheduler_steps=50,
                                       progress_steps=50, finite_checks=1),
                         method=dict(completed_steps=50, active_steps=10, bypassed=False,
                                     model_forwards=110 if method == "alg" else 100))
            if method == "alg":
                stats["method"]["first_frame_sha256"] = "a"*64
            else:
                stats["method"].update(biased_attention_calls=300, original_attention_calls=2700,
                    reference_tokens=880, sequence_tokens=27280)
        helper = dict(schema="teacher_base_case_receipt_v1" if method == "base" else "teacher_devset_case_receipt_v1",
            method=method, plan_sha256=self.shas[method], job=job, success=True, failure=False, error=None,
            parameters_unchanged=True, stats=stats, video=self.fake_video(video_path, self.tools), elapsed_seconds=350.)
        if method == "base":
            helper.update(checkpoint="NONE", active_steps=0, method_context_installed=False, input_binding=inputs,
                parameters_before=params, parameters_after=params, score_bundle_sha256=self.bundle_sha)
        path = video_path.with_suffix(".receipt.json")
        smoke.exclusive(path, helper)
        return plan, job, path, params, inputs

    def verify_case(self, method, case):
        plan, job, path, params, inputs = case
        return comparison.verify_case_receipt(path, smoke.sha(path.read_bytes()), plan=plan,
            plan_sha256=self.shas[method], job=job, parameters_before=params, parameters_after=params, input_binding=inputs)

    def test_case_verifier_recomputes_original_media_and_inputs_all_three_methods(self):
        for method in comparison.METHODS:
            with self.subTest(method=method):
                case = self.case_fixture(method)
                value = self.verify_case(method, case)
                self.assertEqual(value["job"], case[1])
                self.assertEqual(value["video"]["decoded_frames"], 121)
                self.assertEqual(value["parameters"], case[3])

    def test_case_same_size_restoremtime_media_drift_and_bad_decode_rejected(self):
        case = self.case_fixture("base")
        video = Path(case[1]["output"])
        stat = video.stat()
        video.write_bytes(b"x" * stat.st_size)
        __import__("os").utime(video, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "media byte SHA"):
            self.verify_case("base", case)
        case = self.case_fixture("alg")
        wrong = self.fake_video(Path(case[1]["output"]), self.tools)
        wrong["decoded_frames"] = 120
        with mock.patch.object(smoke, "validate_video", return_value=wrong), self.assertRaisesRegex(ValueError, "decode/hash"):
            self.verify_case("alg", case)

    def test_case_input_and_nonfinite_observation_cannot_be_faked(self):
        case = self.case_fixture("dymos")
        case[4]["prompt_sha256"] = "f"*64
        with self.assertRaisesRegex(ValueError, "input binding drift"):
            self.verify_case("dymos", case)
        case = self.case_fixture("base")
        helper = json.loads(case[2].read_bytes())
        helper["elapsed_seconds"] = float("nan")
        case[2].write_text(json.dumps(helper))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            self.verify_case("base", case)

    def completion_fixture(self, method):
        plan = json.loads(self.paths[method].read_bytes())
        case_rows = []
        for index in range(188):
            _, job, path, params, inputs = self.case_fixture(method, index)
            helper = json.loads(path.read_bytes())
            row = dict(job=job, case_receipt=smoke.record(path), parameters_before=params,
                parameters_after=params, input_binding=inputs, stats=helper["stats"], video=helper["video"])
            if method != "base":
                claim_path = Path(job["output"]).with_suffix(".claim.json")
                smoke.exclusive(claim_path, dict(plan_sha256=self.shas[method], job=job,
                    pid=1234 + index % 16, hostname="synthetic CPU owner"))
                row["claim"] = smoke.record(claim_path)
            case_rows.append(row)
        execution_path = self.output_dir / f"explicit-mocked-independent-{method}-execution.json"
        smoke.exclusive(execution_path, dict(synthetic_CPU_only=True, independently_verified_fixture=True))
        execution = dict(plans={method: smoke.record(self.paths[method])}, common_execution_sha256="b"*64,
            score_bundle=smoke.record(self.bundle), allocations={method: plan["allocations"]})
        parents = []
        for shard in range(16):
            folder = Path(plan["output_root"]) / f"generation_shard{shard}"
            folder.mkdir(mode=0o700)
            launch = folder / "launch.json"
            supervised = folder / "supervised.json"
            for proof in (launch, supervised):
                smoke.exclusive(proof, dict(explicit_mocked_independent_process_proof=True))
            worker = folder / "worker.json"
            smoke.exclusive(worker, dict(method=method, shard=shard, success=True, error=None,
                common_execution_sha256="b"*64, score_bundle_sha256=self.bundle_sha,
                cases=case_rows[shard::16], load_seconds=10., owner=dict(pid=1234+shard)))
            parent = folder / "parent.json"
            smoke.exclusive(parent, dict(plan["allocations"][shard], success=True, exit_code=0, error=None,
                interrupted=None, timed_out=False, worker=smoke.record(worker), launch=smoke.record(launch),
                supervised=smoke.record(supervised), elapsed_seconds=4400., hostname="synthetic CPU owner"))
            parents.append(smoke.record(parent))
        prior = 0 if method == "base" else plan["prior_generation"]["upper_seconds"]
        completion = self.output_dir / f"{method}-completion.json"
        value = dict(schema="teacher_generation_completion_v1", private=True, method=method,
            execution=smoke.record(execution_path), plan=smoke.record(self.paths[method]), plan_sha256=self.shas[method],
            common_execution_sha256="b"*64, score_bundle_sha256=self.bundle_sha, parent_receipts=parents, cases=case_rows,
            costs=dict(generation_charged_upper_seconds=prior+16*4400.), generation_complete=True, GPU_used=True,
            quality_pass=False, sota_claim=False)
        smoke.exclusive(completion, value)
        generic = types.ModuleType("run_teacher_generation")
        generic.CLEANUP_SECONDS, generic.WORKER_MARGIN_SECONDS = 120, 10
        generic.shard_folder = lambda plan, shard: Path(plan["output_root"]) / f"generation_shard{shard}"
        generic.verify_parent_release = mock.Mock(return_value=True)
        generic.verify_execution = mock.Mock(return_value=execution)
        generic.verify_generation_completion = mock.Mock(side_effect=lambda path, *a, **kw: json.loads(Path(path).read_bytes()))
        return completion, value, generic

    def verify_full(self, path):
        value = json.loads(path.read_bytes())
        return comparison.verify_generation_completion(path, smoke.sha(path.read_bytes()),
            common_execution_sha256="b"*64, score_bundle_sha256=self.bundle_sha, method=value["method"])

    def test_all188_media_checked_with_explicit_mocked_independent_supervisor_all_methods(self):
        for method in comparison.METHODS:
            path, value, generic = self.completion_fixture(method)
            with self.subTest(method=method), mock.patch.dict(sys.modules, {"run_teacher_generation": generic}):
                actual = self.verify_full(path)
            self.assertEqual(actual, value)
            self.assertEqual(len(actual["cases"]), 188)
            generic.verify_generation_completion.assert_called_once()

    def test_native_canary_admission_requires_parent_chain_and_actual12_full_media(self):
        path, value, generic = self.completion_fixture("base")
        parent = json.loads(Path(value["parent_receipts"][0]["path"]).read_bytes())
        parent.update(method="base", plan_sha256=self.shas["base"])
        parent_path = Path(value["parent_receipts"][0]["path"])
        parent_path.write_bytes(smoke.encoded(parent))
        plan = json.loads(self.paths["base"].read_bytes())
        worker_path = Path(parent["worker"]["path"])
        worker = json.loads(worker_path.read_bytes())
        worker.update(method="base", shard=0)
        worker_path.write_bytes(smoke.encoded(worker))
        parent["worker"] = smoke.record(worker_path)
        parent_path.write_bytes(smoke.encoded(parent))
        with mock.patch.dict(sys.modules, {"run_teacher_generation": generic}):
            self.assertTrue(base.verify_native_canary_admission(parent, plan))
            generic.verify_parent_release.assert_called_once()
            generic.verify_parent_release.side_effect = ValueError("Missing raw actual GPU release proof")
            with self.assertRaisesRegex(ValueError, "actual GPU release"):
                base.verify_native_canary_admission(parent, plan)

    def test_completion_missing_or_duplicate_parent_and_job_rejected(self):
        path, value, generic = self.completion_fixture("base")
        original = copy.deepcopy(value)
        for change in (lambda v: v["parent_receipts"].pop(),
                       lambda v: v["parent_receipts"].__setitem__(1, v["parent_receipts"][0]),
                       lambda v: v["cases"].__setitem__(1, v["cases"][0])):
            altered = copy.deepcopy(original)
            change(altered)
            path.write_bytes(smoke.encoded(altered))
            with mock.patch.dict(sys.modules, {"run_teacher_generation": generic}), self.assertRaises(ValueError):
                self.verify_full(path)

    def test_completion_extra_partial_file_and_changed_media_rejected(self):
        path, value, generic = self.completion_fixture("base")
        video_root = Path(value["cases"][0]["job"]["output"]).parent
        extra = video_root / "extra.partial.mp4"
        extra.write_bytes(b"synthetic extra")
        with mock.patch.dict(sys.modules, {"run_teacher_generation": generic}), self.assertRaisesRegex(ValueError, "Extra, partial"):
            self.verify_full(path)
        extra.unlink()
        video = Path(value["cases"][0]["job"]["output"])
        stat = video.stat()
        video.write_bytes(b"x"*stat.st_size)
        __import__("os").utime(video, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with mock.patch.dict(sys.modules, {"run_teacher_generation": generic}), self.assertRaisesRegex(ValueError, "media byte SHA"):
            self.verify_full(path)

    def test_candidate_claim_owner_forgery_rejected_even_if_hash_reapproved(self):
        path, value, generic = self.completion_fixture("alg")
        row = value["cases"][0]
        claim_path = Path(row["claim"]["path"])
        claim = json.loads(claim_path.read_bytes())
        claim["pid"] += 1
        claim_path.write_bytes(smoke.encoded(claim))
        row["claim"] = smoke.record(claim_path)
        parent_path = Path(value["parent_receipts"][0]["path"])
        parent = json.loads(parent_path.read_bytes())
        worker_path = Path(parent["worker"]["path"])
        worker = json.loads(worker_path.read_bytes())
        worker["cases"][0] = row
        worker_path.write_bytes(smoke.encoded(worker))
        parent["worker"] = smoke.record(worker_path)
        parent_path.write_bytes(smoke.encoded(parent))
        value["parent_receipts"][0] = smoke.record(parent_path)
        path.write_bytes(smoke.encoded(value))
        with mock.patch.dict(sys.modules, {"run_teacher_generation": generic}), self.assertRaisesRegex(ValueError, "persistent claim"):
            self.verify_full(path)

    def test_completion_overcap_parent_cannot_be_hidden_in_total_cost(self):
        path, value, generic = self.completion_fixture("base")
        parent_path = Path(value["parent_receipts"][0]["path"])
        parent = json.loads(parent_path.read_bytes())
        parent["elapsed_seconds"] = 4501.
        parent_path.write_bytes(smoke.encoded(parent))
        value["parent_receipts"][0] = smoke.record(parent_path)
        value["costs"]["generation_charged_upper_seconds"] += 101.
        path.write_bytes(smoke.encoded(value))
        with mock.patch.dict(sys.modules, {"run_teacher_generation": generic}), self.assertRaisesRegex(ValueError, "stagecap"):
            self.verify_full(path)

    def test_completion_parameter_map_drift_detected_after_all_hashes_reapproved(self):
        path, value, generic = self.completion_fixture("base")
        row = value["cases"][16]
        row["parameters_before"] = copy.deepcopy(row["parameters_before"])
        row["parameters_before"]["w"]["sha256"] = "c"*64
        row["parameters_after"] = copy.deepcopy(row["parameters_before"])
        helper_path = Path(row["case_receipt"]["path"])
        helper = json.loads(helper_path.read_bytes())
        helper.update(parameters_before=row["parameters_before"], parameters_after=row["parameters_after"])
        helper_path.write_bytes(smoke.encoded(helper))
        row["case_receipt"] = smoke.record(helper_path)
        parent_path = Path(value["parent_receipts"][0]["path"])
        parent = json.loads(parent_path.read_bytes())
        worker_path = Path(parent["worker"]["path"])
        worker = json.loads(worker_path.read_bytes())
        worker["cases"][1] = row
        worker_path.write_bytes(smoke.encoded(worker))
        parent["worker"] = smoke.record(worker_path)
        parent_path.write_bytes(smoke.encoded(parent))
        value["parent_receipts"][0] = smoke.record(parent_path)
        path.write_bytes(smoke.encoded(value))
        with mock.patch.dict(sys.modules, {"run_teacher_generation": generic}), self.assertRaisesRegex(ValueError, "global native baseline"):
            self.verify_full(path)

    def test_completion_parent_verifier_failure_cannot_be_overridden_by_all_booleans(self):
        path, value, generic = self.completion_fixture("base")
        generic.verify_generation_completion.side_effect = ValueError("Missing actual raw release/ancestry proof")
        with mock.patch.dict(sys.modules, {"run_teacher_generation": generic}), self.assertRaisesRegex(ValueError, "actual raw release"):
            self.verify_full(path)

    def frozen_view_fixture(self):
        import score_e1 as e1
        self.scoring.BUNDLE_SCHEMA = "teacher_devset_score_bundle_v1"
        self.scoring.e1 = e1
        score_source = self.root / "scripts/score_e1.py"
        self.scoring.required_sources = lambda: [str(score_source)]
        score_row = {k: v for k, v in smoke.record(score_source).items() if k != "mtime_ns"}
        value = dict(schema=self.scoring.BUNDLE_SCHEMA, private=True, metric_protocol="score_e1_all8_gt_v1",
            dimensions=e1.DIMS, gt_keys=list(e1.GT_KEYS), sources=[score_row],
            dependencies=[dict(path="/absent-score-host/dependency.so", bytes=20, sha256="c"*64)],
            models=[dict(path="/absent-score-host/metric-model.bin", bytes=30, sha256="d"*64)],
            runtime=dict(files=[dict(path="/absent-score-host/python", bytes=40, sha256="e"*64)]))
        self.bundle.write_bytes(smoke.encoded(value))
        self.bundle_sha = smoke.sha(self.bundle.read_bytes())
        binding = smoke.record(self.bundle)
        for method in comparison.METHODS:
            self.mutate_plan(method, lambda p: p.update(score_bundle=binding))
        return self.prepare_comparison()

    def test_remote_frozen_view_checks_shared_bytes_without_missing_peer_assets(self):
        path, digest = self.frozen_view_fixture()
        original = comparison.verify(path, digest)
        plans = [json.loads(p.read_bytes()) for p in self.paths.values()]
        local_asset_paths = {r["path"] for plan in plans for r in plan["models"] + plan["runtime"]["files"]}
        original_record = comparison.record
        def reject_peer_assets(file_path, **kwargs):
            if str(file_path) in local_asset_paths or str(file_path).startswith("/absent-score-host/"):
                raise AssertionError("Remote metadata view opened score-host/model asset: " + str(file_path))
            return original_record(file_path, **kwargs)
        with mock.patch.object(comparison, "record", side_effect=reject_peer_assets), \
             mock.patch.object(comparison, "check_record", side_effect=lambda r, **kw: reject_peer_assets(r["path"], **kw)), \
             mock.patch.object(smoke, "runtime_records", side_effect=AssertionError("Remote view observed root0 runtime")):
            verified = comparison.verify_metadata(path, digest)
        self.assertEqual({k: v for k, v in verified.items() if k != "frozen_contract_view"}, original)
        self.assertEqual(verified["frozen_contract_view"], dict(shared_content_verified=True,
            actual_model_runtime_score_assets_verified=False, requires_full_score_owner_proof=True, GPU_authorized=False))
        self.scoring.verify_bundle.reset_mock()
        with mock.patch.object(smoke, "runtime_records", side_effect=AssertionError("Full verification must observe runtime")), \
             self.assertRaisesRegex(AssertionError, "observe runtime"):
            comparison.verify(path, digest)
        self.scoring.verify_bundle.assert_called_once()

    def test_frozen_view_rejects_changed_shared_source_and_case_bytes(self):
        path, digest = self.frozen_view_fixture()
        source = self.root / "scripts/score_e1.py"
        original = source.read_bytes()
        source.write_bytes(b"x"*len(original))
        with self.assertRaisesRegex(ValueError, "source content drift"):
            comparison.verify_metadata(path, digest)

    def test_frozen_view_rejects_changed_shared_case_bytes(self):
        path, digest = self.frozen_view_fixture()
        self.cases_path.write_bytes(self.cases_path.read_bytes()+b"\n")
        with self.assertRaisesRegex(ValueError, "case manifest SHA drift"):
            comparison.verify_metadata(path, digest)

    def test_frozen_view_rejects_common_model_or_runtime_metadata_substitution(self):
        path, digest = self.frozen_view_fixture()
        self.mutate_plan("alg", lambda p: p["runtime"]["packages"].update(mock="other runtime"))
        value = json.loads(path.read_bytes())
        value["plans"]["alg"] = smoke.record(self.paths["alg"])
        path.write_bytes(smoke.encoded(value))
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            comparison.verify_metadata(path, smoke.sha(path.read_bytes()))

    def test_actual_parameter_sha_maps_reject_bool_forgery_and_shape_drift(self):
        params = dict(w=dict(shape=[2,2], dtype="torch.bfloat16", sha256="a"*64))
        self.assertEqual(comparison.validate_parameter_maps(params, copy.deepcopy(params)), params)
        for before, after in (({}, {}), (True, True), (params, {"w": True}),
                              ({"w": dict(shape=[True], dtype="torch.bfloat16", sha256="a"*64)},
                               {"w": dict(shape=[True], dtype="torch.bfloat16", sha256="a"*64)})):
            with self.assertRaises(ValueError):
                comparison.validate_parameter_maps(before, after)

    def test_generation_evidence_reader_rejects_duplicate_nonfinite_and_hash_drift(self):
        path = self.output_dir / "evidence.json"
        for raw, message in ((b'{"success":false,"success":true}', "Duplicate"),
                             (b'{"seconds":NaN}', "Nonfinite")):
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, message):
                comparison.read_evidence(path, smoke.sha(raw))
        path.write_bytes(b'{"success":true}')
        with self.assertRaisesRegex(ValueError, "SHA"):
            comparison.read_evidence(path, "f"*64)

    def test_approved_score_content_api_failure_is_not_ignored(self):
        self.scoring.verify_bundle.side_effect = ValueError("recursive score dependency/model coverage drift")
        with self.assertRaisesRegex(ValueError, "coverage drift"):
            self.prepare_comparison()


if __name__ == "__main__":
    unittest.main()
