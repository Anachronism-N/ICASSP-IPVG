"""Synthetic CPU fixtures only; no real base/GPU/throughput PASS."""
import builtins
import contextlib
import copy
import io
import importlib.util
import json
import sys
from pathlib import Path
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
from test_teacher_devset import Fixture as DevFixture, dev, smoke
import run_teacher_base as base


class Fixture(DevFixture):
    def setUp(self):
        super().setUp()
        for name in ("run_teacher_base.py", "prepare_teacher_comparison.py", "run_teacher_generation.py",
                     "verify_teacher_history.py"):
            (self.root / "scripts" / name).write_text("# Synthetic CPU source fixture\n")
        for name in ("test_teacher_base.py", "test_prepare_teacher_comparison.py", "test_teacher_generation.py",
                     "test_verify_teacher_history.py"):
            (self.root / "tests" / name).write_text("# Synthetic CPU test fixture\n")
        for patch in (mock.patch.object(base, "dev", dev), mock.patch.object(base, "smoke", smoke)):
            patch.start()
            self.addCleanup(patch.stop)
        self.bundle = self.base / "approved-score-bundle.json"
        self.bundle.write_bytes(smoke.encoded(dict(schema="teacher_devset_score_bundle_v1", private=True, synthetic=True)))
        self.bundle_sha = smoke.sha(self.bundle.read_bytes())
        self.scoring = types.ModuleType("score_teacher_devset")
        self.scoring.verify_bundle = mock.Mock(return_value=json.loads(self.bundle.read_bytes()))
        patch = mock.patch.dict(sys.modules, {"score_teacher_devset": self.scoring})
        patch.start()
        self.addCleanup(patch.stop)

    def prepare_base(self, **kwargs):
        kwargs.setdefault("node_ranks", [0, 1])
        return base.prepare(self.pf, self.node_sha, self.bundle, self.bundle_sha, root=self.root, **kwargs)

    def verify_base(self, path, digest):
        return base.verify_plan(path, digest, self.pf, self.node_sha)


class PrepareTests(Fixture):
    def test_cpu_prepare_and_verify_are_always_blocked_native_none(self):
        path, digest = self.prepare_base()
        plan = self.verify_base(path, digest)
        self.assertEqual(plan["method"], "base")
        self.assertEqual(plan["parameters"], {})
        self.assertEqual(plan["active_steps"], 0)
        self.assertFalse(plan["method_context_installed"])
        self.assertEqual(plan["phase_seconds"], dict(generation=72000, score=18000))
        self.assertEqual(plan["total_seconds"], 90000)
        self.assertEqual(len(plan["jobs"]), 188)
        self.assertEqual({j["seed"] for j in plan["jobs"]}, {0, 1})
        self.assertEqual(sum(plan["shard_cap_seconds"]), 72000)
        self.assertEqual(path.name, "plan.BLOCKED.json")
        self.assertEqual(plan["admission"]["status"], "BLOCKED")
        self.assertIsNone(plan["admission"]["estimated_generation_seconds"])
        self.assertFalse(plan["admission"]["runnable"])
        self.assertTrue(all(r["allowed"] is False for r in plan["historical_base_reuse"].values()))
        self.scoring.verify_bundle.assert_called_with(str(self.bundle), self.bundle_sha, verify_content=True)
        self.assertFalse((path.parent / "plan.json").exists())
        self.assertEqual(path.stat().st_mode & 0o077, 0)

    def test_cpu_prepare_imports_no_torch_or_methods(self):
        original = builtins.__import__
        def reject(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "gen_jobs", "diffsynth", "teacher_alg", "teacher_dymos"):
                raise AssertionError("CPU preparation attempted runtime import: " + name)
            return original(name, *args, **kwargs)
        with mock.patch("builtins.__import__", side_effect=reject):
            path, digest = self.prepare_base()
            self.verify_base(path, digest)

    def test_default_cli_is_cpu_prepare(self):
        with mock.patch.object(base, "prepare", return_value=(Path("plan.BLOCKED.json"), "a"*64)) as fn, mock.patch("builtins.print"):
            base.main(["--node-preflight", str(self.pf), "--approved-node-preflight-sha256", self.node_sha,
                       "--score-bundle", str(self.bundle), "--approved-score-bundle-sha256", self.bundle_sha, "--node-ranks", "0", "1"])
        fn.assert_called_once()

    def test_no_gpu_run_or_direct_worker_even_with_authorization(self):
        with self.assertRaisesRegex(ValueError, "gpu-authorized"):
            base.run()
        with self.assertRaisesRegex(ValueError, "BLOCKED"):
            base.main(["--run", "--gpu-authorized"])
        with mock.patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
            base.main(["--worker", "--gpu-authorized"])

    def test_authorized_native_run_delegates_only_to_hash_approved_managed_parent(self):
        generic = types.ModuleType("run_teacher_generation")
        generic.supervise = mock.Mock(return_value=(Path("private/parent.json"), "b"*64))
        with mock.patch.dict(sys.modules, {"run_teacher_generation": generic}):
            result = base.run(Path("private/execution.json"), "a"*64, 0, 1, gpu_authorized=True,
                              state_dir=Path("local/gpu_hold"), local_root=Path("local"))
            self.assertEqual(result, (Path("private/parent.json"), "b"*64))
            generic.supervise.assert_called_once_with(Path("private/execution.json"), "a"*64, "base", 0, 1,
                gpu_authorized=True, state_dir=Path("local/gpu_hold"), local_root=Path("local"))
            generic.supervise.reset_mock()
            for shard, rank in ((True, 0), (16, 0), (0, True), (0, 4)):
                with self.assertRaises(ValueError):
                    base.run(Path("private/execution.json"), "a"*64, shard, rank, gpu_authorized=True)
            generic.supervise.assert_not_called()

    def test_native_node_mapping_requires_explicit_two_distinct_owners(self):
        for ranks in (None, [0,0], [0,4], [True,1], [0], [0,1,2]):
            with self.assertRaises(ValueError):
                base.allocations(ranks)
        rows = base.allocations([2,3])
        self.assertEqual(len(rows), 16)
        self.assertEqual([r["shard"] for r in rows], list(range(16)))
        self.assertEqual(sum(r["cap_seconds"] for r in rows), 72000)

    def test_prepare_rejects_gpu_arguments(self):
        with mock.patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
            base.main(["--gpu", "0"])

    def test_no_overwrite(self):
        path, _ = self.prepare_base()
        original = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.prepare_base()
        self.assertEqual(path.read_bytes(), original)

    def test_caps_invalid_types_or_total_overrun_rejected(self):
        for n, cap in ((16, 4501), (True, 4500), (0, 4500), (189, 100), (16, 35)):
            with self.subTest(n=n, cap=cap), self.assertRaises(ValueError):
                self.prepare_base(nshards=n, gen_cap_seconds=cap)
        self.assertFalse(base.output_root(self.root).exists())

    def test_approved_bundle_sha_and_content_failure_rejected_before_output(self):
        with self.assertRaisesRegex(ValueError, "score bundle SHA"):
            base.prepare(self.pf, self.node_sha, self.bundle, "f"*64, root=self.root, node_ranks=[0,1])
        self.scoring.verify_bundle.side_effect = ValueError("score content drift")
        with self.assertRaisesRegex(ValueError, "content drift"):
            self.prepare_base()
        self.assertFalse(base.output_root(self.root).exists())

    def test_actual_scoring_bundle_api_integrates_and_rejects_changed_model_bytes(self):
        spec = importlib.util.spec_from_file_location("actual_score_bundle_cpu_test", ROOT / "scripts/score_teacher_devset.py")
        scoring = importlib.util.module_from_spec(spec)
        original = builtins.__import__
        def reject(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "dreamsim", "vbench", "diffsynth"):
                raise AssertionError("Real score bundle API imported GPU runtime: " + name)
            return original(name, *args, **kwargs)
        with mock.patch("builtins.__import__", side_effect=reject):
            spec.loader.exec_module(scoring)
        scoring.ROOT = self.root
        def row(path):
            return {k: v for k, v in smoke.record(path).items() if k != "mtime_ns"}
        source_files = []
        for name in scoring.required_sources():
            path = Path(name)
            if not path.exists():
                path.write_bytes(b"Synthetic CPU scorer source fixture")
            source_files.append(row(path))
        def assets(kind, roles):
            roots, rows = {}, []
            for role in sorted(roles):
                folder = self.base / kind / role
                folder.mkdir(parents=True)
                path = folder / "fake-asset"
                path.write_bytes(b"CPU tiny content fixture")
                roots[role] = str(folder)
                rows.append(row(path))
            return roots, rows
        dep_roots, deps = assets("deps", scoring.DEPENDENCIES)
        model_roots, models = assets("metrics", {"dreamsim", "vbench"})
        tools = {}
        for tool in ("python", "ffmpeg", "ffprobe"):
            path = self.base / ("fake-" + tool)
            path.write_bytes(b"CPU executable content fixture; never run")
            tools[tool] = str(path)
        bundle = dict(schema=scoring.BUNDLE_SCHEMA, private=True, metric_protocol="score_e1_all8_gt_v1",
            dimensions=scoring.e1.DIMS, gt_keys=list(scoring.e1.GT_KEYS), sources=source_files,
            dependency_roots=dep_roots, dependencies=deps, model_roots=model_roots, models=models,
            runtime=dict(files=[row(p) for p in tools.values()], tools=tools,
                         versions={k: "synthetic CPU fixture" for k in scoring.RUNTIME_PACKAGES}))
        self.bundle.write_bytes(smoke.encoded(bundle))
        digest = smoke.sha(self.bundle.read_bytes())
        with mock.patch.dict(sys.modules, {"score_teacher_devset": scoring}), mock.patch("builtins.__import__", side_effect=reject):
            result = base.score_bundle_binding(self.bundle, digest)
            self.assertEqual(result["sha256"], digest)
            model = Path(models[0]["path"])
            old = model.read_bytes()
            model.write_bytes(b"x"*len(old))
            with self.assertRaisesRegex(RuntimeError, "SHA"):
                base.score_bundle_binding(self.bundle, digest)

    def test_native_parameter_bypass_and_false_admission_rejected(self):
        path, _ = self.prepare_base()
        plan = json.loads(path.read_bytes())
        for mutation in (dict(parameters=dict(factor=1)), dict(parameters=dict(gamma=0)),
                         dict(method_context_installed=True), dict(active_steps=10),
                         dict(admission=dict(status="ready", runnable=True)), dict(total_seconds=90001)):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                base.validate_contract(dict(plan, **mutation))

    def test_case_source_bundle_runtime_drift_rejected(self):
        path, digest = self.prepare_base()
        source = self.root / "scripts/run_teacher_base.py"
        source.write_text("changed source\n")
        with self.assertRaisesRegex(ValueError, "drift"):
            self.verify_base(path, digest)

    def test_plan_tampering_and_promoted_filename_rejected(self):
        path, digest = self.prepare_base()
        with self.assertRaisesRegex(ValueError, "SHA"):
            self.verify_base(path, "f"*64)
        promoted = path.with_name("plan.json")
        promoted.write_bytes(path.read_bytes())
        promoted.chmod(0o600)
        with self.assertRaisesRegex(ValueError, "root"):
            self.verify_base(promoted, digest)


class FakeTensor:
    def is_floating_point(self):
        return True


class NativeTests(Fixture):
    def native_runtime(self, plan, *, forwards=2, steps=50, frames=121, fail=False):
        truth = types.SimpleNamespace(all=lambda: types.SimpleNamespace(item=lambda: True))
        torch = types.SimpleNamespace(Tensor=FakeTensor, isfinite=lambda v: truth)
        tensor = FakeTensor()
        scheduler = types.SimpleNamespace(step=lambda *args: tensor)
        class NativePipeline:
            def __init__(self):
                self.dit = object()
                self.scheduler = scheduler
                self.model_fn = lambda **kwargs: tensor
            def __call__(self, **kwargs):
                self.call = kwargs
                for i in kwargs["progress_bar_cmd"](list(range(steps))):
                    if fail and i == 3:
                        raise RuntimeError("CPU synthetic native failure")
                    for _ in range(forwards):
                        prediction = self.model_fn(latents=tensor, first_frame_latents=tensor, context=tensor)
                    self.scheduler.step(prediction, i, tensor)
                return [None]*frames
        pipe = NativePipeline()
        generator = types.SimpleNamespace(load_image=mock.Mock(return_value="CPU image"),
            save_video=lambda video, path, **kwargs: Path(path).write_bytes(b"synthetic CPU native media; no generation"))
        return pipe, torch, generator

    def test_native_direct_pipeline_only_observes_no_method_context_and_binds_receipt(self):
        path, digest = self.prepare_base()
        plan = json.loads(path.read_bytes())
        job = plan["jobs"][94]
        pipe, torch, generator = self.native_runtime(plan)
        original_model, original_step = pipe.model_fn, pipe.scheduler.step
        before = dict(w=dict(shape=[2,2], dtype="fake CPU", sha256="a"*64))
        with mock.patch.object(smoke, "parameter_hashes", return_value=before):
            result = base._generate_native_case(pipe, torch, generator, plan, digest, job, lambda: None)
        self.assertTrue(result["success"])
        self.assertEqual(result["stats"], dict(model_forwards=100, scheduler_steps=50, progress_steps=50, finite_checks=550))
        self.assertEqual(result["parameters_before"], result["parameters_after"])
        self.assertFalse(result["allocation_released"])
        self.assertFalse(result["method_context_installed"])
        self.assertEqual(result["score_bundle_sha256"], self.bundle_sha)
        self.assertEqual(result["plan_sha256"], digest)
        self.assertEqual(result["video"]["decoded_frames"], 121)
        self.assertEqual(result["video"]["file"]["sha256"], smoke.sha(Path(job["output"]).read_bytes()))
        self.assertIs(pipe.model_fn, original_model)
        self.assertIs(pipe.scheduler.step, original_step)
        self.assertEqual(pipe.call["seed"], 1)
        self.assertEqual(pipe.call["num_inference_steps"], 50)

    def test_incomplete_native_stats_or_frames_fail_and_keep_failure_receipt(self):
        for name, kwargs in (("forwards", dict(forwards=3)), ("steps", dict(steps=49)),
                             ("frames", dict(frames=120)), ("failure", dict(fail=True))):
            with self.subTest(name=name):
                plan = json.loads(self.prepare_base()[0].read_bytes()) if name == "forwards" else plan
                job = plan["jobs"][{"forwards":0,"steps":1,"frames":2,"failure":3}[name]]
                pipe, torch, generator = self.native_runtime(plan, **kwargs)
                original = pipe.model_fn
                with mock.patch.object(smoke, "parameter_hashes", return_value={"w": "a"*64}), self.assertRaises((ValueError, RuntimeError)):
                    base._generate_native_case(pipe, torch, generator, plan, smoke.sha(smoke.encoded(plan)), job, lambda: None)
                receipt = json.loads(Path(job["output"]).with_suffix(".receipt.json").read_bytes())
                self.assertFalse(receipt["success"])
                self.assertTrue(receipt["failure"])
                self.assertFalse(Path(job["output"]).exists())
                self.assertIs(pipe.model_fn, original)

    def test_native_weight_mutation_rejected(self):
        path, digest = self.prepare_base()
        plan = json.loads(path.read_bytes())
        job = plan["jobs"][0]
        pipe, torch, generator = self.native_runtime(plan)
        with mock.patch.object(smoke, "parameter_hashes", side_effect=[{"w": "a"*64}, {"w": "b"*64}]), \
                self.assertRaisesRegex(ValueError, "parameters changed"):
            base._generate_native_case(pipe, torch, generator, plan, digest, job, lambda: None)
        receipt = json.loads(Path(job["output"]).with_suffix(".receipt.json").read_bytes())
        self.assertFalse(receipt["parameters_unchanged"])
        self.assertFalse(Path(job["output"]).exists())

    def test_native_finite_counter_is_observed_not_a_synthetic550_threshold(self):
        base.validate_native_stats(dict(model_forwards=100, scheduler_steps=50, progress_steps=50, finite_checks=1))
        for bad in (0, False, float("nan")):
            with self.assertRaises(ValueError):
                base.validate_native_stats(dict(model_forwards=100, scheduler_steps=50, progress_steps=50, finite_checks=bad))

    def canary_fixture(self):
        path, digest = self.prepare_base()
        plan = json.loads(path.read_bytes())
        params = dict(w=dict(shape=[2,2], dtype="torch.bfloat16", sha256="a"*64))
        receipts = [dict(schema="teacher_base_case_receipt_v1", method="base", checkpoint="NONE",
            method_context_installed=False, active_steps=0, job=j, success=True, failure=False, error=None,
            plan_sha256=digest, score_bundle_sha256=self.bundle_sha, parameters_before=params, parameters_after=params,
            stats=dict(model_forwards=100, scheduler_steps=50, progress_steps=50, finite_checks=1), elapsed_seconds=350.)
            for j in plan["jobs"] if j["index"] % 16 == 0]
        return plan, receipts

    def test_native_formal_canary_is12of188_charged_in20gpuh(self):
        plan, receipts = self.canary_fixture()
        result = base.native_canary_estimate(plan, receipts, load_seconds=10., allocated_elapsed_seconds=4245.)
        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["canary_jobs"], 12)
        self.assertLessEqual(result["estimated_generation_seconds"], 72000)
        self.assertTrue(all(s <= 4500 for s in result["estimated_shard_seconds"]))
        self.assertFalse(result["quality_pass"])
        self.assertFalse(result["estimate_is_guarantee"])

    def test_native_canary_cannot_borrow_dymos_or_zero_timing(self):
        plan, receipts = self.canary_fixture()
        original = copy.deepcopy(receipts)
        for change in (lambda rs: rs[0].update(method="dymos"),
                       lambda rs: rs[0].update(elapsed_seconds=0),
                       lambda rs: rs[0].update(elapsed_seconds=float("inf")),
                       lambda rs: rs.pop(), lambda rs: rs.reverse(),
                       lambda rs: rs[0].update(parameters_after={"w": "forged boolean proof"})):
            changed = copy.deepcopy(original)
            change(changed)
            with self.assertRaises(ValueError):
                base.native_canary_estimate(plan, changed, load_seconds=10., allocated_elapsed_seconds=4245.)

    def test_canary_slow_seed_blocks_remaining_shards_without_spec_reduction(self):
        plan, receipts = self.canary_fixture()
        receipts[0]["elapsed_seconds"] = 400.
        result = base.native_canary_estimate(plan, receipts, load_seconds=10., allocated_elapsed_seconds=4295.)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertFalse(result["native_throughput_PASS"])
        self.assertTrue(any(s > 4500 for s in result["estimated_shard_seconds"]))
        self.assertEqual(plan["call"]["num_inference_steps"], 50)

    def test_canary_reserves_driver_cleanup_and_worker_margin_for_tail_jobs(self):
        plan, receipts = self.canary_fixture()
        for receipt, seconds in zip(receipts, [360.]*11+[200.]):
            receipt["elapsed_seconds"] = seconds
        result = base.native_canary_estimate(plan, receipts, load_seconds=100., allocated_elapsed_seconds=4270.)
        self.assertEqual(result["cleanup_seconds"], 120)
        self.assertEqual(result["worker_margin_seconds"], 10)
        self.assertGreaterEqual(result["allocation_residual_seconds"], 130)
        self.assertEqual(result["estimated_body_seconds"][0], 4420)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertFalse(result["native_throughput_PASS"])
        self.assertEqual(plan["phase_seconds"]["generation"], 72000)
        result = base.native_canary_estimate(plan, receipts, load_seconds=50., allocated_elapsed_seconds=4220.)
        self.assertEqual(result["estimated_body_seconds"][0], 4370)
        self.assertEqual(result["status"], "BLOCKED")

    def test_canary_allocation_over_cap_and_overlapping_timer_rejected(self):
        plan, receipts = self.canary_fixture()
        for elapsed in (4501., 4000., float("nan")):
            with self.assertRaises(ValueError):
                base.native_canary_estimate(plan, receipts, load_seconds=10., allocated_elapsed_seconds=elapsed)

    def test_native_existing_output_preserved(self):
        path, digest = self.prepare_base()
        plan = json.loads(path.read_bytes())
        job = plan["jobs"][0]
        output = Path(job["output"])
        output.write_bytes(b"existing user media")
        pipe, torch, generator = self.native_runtime(plan)
        with self.assertRaisesRegex(ValueError, "Existing"):
            base._generate_native_case(pipe, torch, generator, plan, digest, job, lambda: None)
        self.assertEqual(output.read_bytes(), b"existing user media")


if __name__ == "__main__":
    unittest.main()
