"""CPU-only mocked driver tests. Fake inference is NOT real smoke/CUDA PASS."""
import builtins
import contextlib
import copy
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
from test_teacher_inference_smoke import Fixture as SmokeFixture, smoke

SPEC = importlib.util.spec_from_file_location("teacher_devset_driver", ROOT / "scripts/run_teacher_devset.py")
dev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dev)
REAL_SUBPROCESS_RUN = subprocess.run


def method_stats(method):
    value = dict(completed_steps=50, active_steps=10, bypassed=False,
                 model_forwards=110 if method == "alg" else 100)
    if method == "alg":
        value["first_frame_sha256"] = "a" * 64
    else:
        value.update(biased_attention_calls=300, original_attention_calls=2700,
                     reference_tokens=880, sequence_tokens=27280)
    return value


def observed(method):
    return dict(model_forwards=110 if method == "alg" else 100,
                scheduler_steps=50, progress_steps=50, finite_checks=350)


class Fixture(SmokeFixture):
    def setUp(self):
        super().setUp()
        patch = mock.patch.object(dev, "smoke", smoke)
        patch.start()
        self.addCleanup(patch.stop)
        images = self.cases_path.parent / "images"
        images.mkdir()
        image = images / "first.png"
        image.write_bytes(b"CPU fake first image")
        gt = self.base / "gt.mp4"
        gt.write_bytes(b"CPU fake reference")
        cases = [json.loads(line) for line in self.cases_path.read_bytes().splitlines()]
        for case in cases:
            case.update(image=str(image), image_sha256=smoke.sha(image.read_bytes()), gt=str(gt),
                        video_sha256=smoke.sha(gt.read_bytes()), n_frames=121, width=1280, height=704,
                        fps=24, dataset="fixture", parent_keys=["fixture:parent"], parent_resolved=True)
        self.cases_path.write_bytes(b"".join(smoke.encoded(c) for c in cases))
        self.cases = cases
        self.manifest_sha = smoke.sha(self.cases_path.read_bytes())
        patch = mock.patch.object(dev, "CASES_SHA256", self.manifest_sha)
        patch.start()
        self.addCleanup(patch.stop)
        for name in ("run_teacher_devset.py", "score_e1.py", "build_e1_jobs.py", "summarize_eval.py"):
            (self.root / "scripts" / name).write_text("# Explicit CPU fixture source\n")
        (self.root / "tests").mkdir()
        (self.root / "tests/test_teacher_devset.py").write_text("# Explicit CPU fixture test\n")
        (self.root / "scripts/measure_teacher_throughput.py").write_text("# CPU batch fixture\n")
        (self.root / "tests/test_teacher_throughput.py").write_text("# CPU batch fixture test\n")
        patch = mock.patch.object(dev.performance, "smoke", smoke)
        patch.start()
        self.addCleanup(patch.stop)
        self.original_helper = (self.root / "scripts/run_teacher_inference_smoke.py").read_bytes()
        self.history_manifest = smoke.encoded(dict(files=[dict(
            path="scripts/run_teacher_inference_smoke.py",
            source_sha256=smoke.sha(self.original_helper), snapshot_sha256=smoke.sha(self.original_helper))]))
        def git_fixture(command, **kwargs):
            if command[0] == "git":
                self.assertEqual(command[1:4], ["-C", str(dev.performance.HISTORY_REPO), "show"])
                self.assertTrue(command[4].startswith(dev.performance.HISTORY_COMMIT + ":"))
                data = self.history_manifest if command[4].endswith("PUBLICATION_MANIFEST.json") else self.original_helper
                return types.SimpleNamespace(stdout=data)
            return REAL_SUBPROCESS_RUN(command, **kwargs)
        patch = mock.patch.object(dev.performance.subprocess, "run", side_effect=git_fixture)
        patch.start()
        self.addCleanup(patch.stop)
        seed1 = self.root / dev.BASE_ROOTS["seed1"]
        seed1.mkdir()
        (seed1 / "cases.jsonl").write_bytes(self.cases_path.read_bytes())
        self.node_sha = smoke.sha(self.pf.read_bytes())
        self.decode = mock.patch.object(smoke, "validate_video", side_effect=self.fake_video)
        self.decode.start()
        self.addCleanup(self.decode.stop)

    def fake_video(self, path, tools):
        return dict(file=smoke.record(path, nonempty=True), decoded_frames=121, width=1280,
                    height=704, fps=24, full_decode_xerror=True)

    def finish_smokes(self, method="alg", elapsed=400, first_elapsed=20, load=100, with_timing=True):
        self.throughput_sha = None
        for suffix, date, cost in (("01", "20261025", first_elapsed), ("02", "20261026", elapsed)):
            folder = self.root / f"eval/teacher_{method}_smoke_{date}_{suffix}"
            path, digest = smoke.prepare(method, folder, self.pf, self.node_sha, root=self.root)
            request = json.loads(path.read_bytes())
            video = folder / "video.mp4"
            video.write_bytes(b"Explicit mocked CPU full-video fixture")
            stats = dict(method=method_stats(method), observed=observed(method))
            evidence = dict(success=suffix == "02", mechanism_PASS=suffix == "02",
                            parameters_unchanged=True, parameters_before={"weight": "unchanged"},
                            parameters_after={"weight": "unchanged"}, stats=stats,
                            video=self.fake_video(video, self.tools))
            launch = dict(request_sha256=digest, budget=smoke.BUDGET)
            receipt = dict(schema="teacher_smoke_receipt_v1", method=method, request_sha256=digest,
                           allocated_gpus=1, elapsed_seconds=cost, elapsed_GPUh=cost / 3600,
                           start=1000, end=1000 + cost, success=suffix == "02", mechanism_PASS=suffix == "02",
                           owned_tree_released=True, budget_exceeded=False, exit_code=0,
                           timed_out=False, interrupted_signal=None, error=None, evidence=evidence, budget=smoke.BUDGET)
            smoke.exclusive(folder / "launch.json", launch)
            smoke.exclusive(folder / "worker.json", evidence)
            smoke.exclusive(folder / "finished.json", receipt)
        self.batch_path = None
        if with_timing and first_elapsed + elapsed + 2700 < 3600:
            self.make_batch(method, load, elapsed - load - 10)

    def make_batch(self, method, load, inference):
        perf = dev.performance
        folder = perf.output_root(self.root, method)
        original = self.root / f"eval/teacher_{method}_smoke_20261026_02"
        request_path, digest = perf.prepare(method, folder,
            smoke.sha((original / "finished.json").read_bytes()), self.root)
        request = json.loads(request_path.read_bytes())
        cursor = 1000.
        def timer(seconds):
            nonlocal cursor
            result = dict(start=cursor, end=cursor + seconds, seconds=seconds)
            cursor += seconds
            return result
        load_timer = timer(float(load))
        videos = []
        for seed in (0, 1):
            media = folder / f"seed{seed}.mp4"
            media.write_bytes(f"CPU MOCK seed{seed} full video".encode())
            videos.append(dict(seed=seed, stats=dict(method=method_stats(method), observed=observed(method)),
                video=self.fake_video(media, self.tools),
                timers=dict(inference=timer(float(inference)), save_decode=timer(10.))))
        worker = dict(success=True, failure=False, mechanism_PASS=True, request_sha256=digest,
                      parameters_unchanged=True, parameters_before={"w": "same"}, parameters_after={"w": "same"},
                      clock="time.monotonic", compile_accounting="included in inference", load_count=1,
                      timers=dict(load=load_timer), videos=videos)
        launch = dict(request_sha256=digest, budget=perf.BUDGET, gpu=0, gpu_uuid="GPU-CPU-fixture", supervisor_pid=12345)
        smoke.exclusive(folder / "launch.json", launch)
        smoke.exclusive(folder / "worker.json", worker)
        elapsed = cursor - 1000.
        receipt = dict(schema=perf.SCHEMA, method=method, measured=True, allocated_gpus=1, budget=perf.BUDGET,
                       success=True, failure=False, mechanism_PASS=True, owned_tree_released=True,
                       budget_exceeded=False, exit_code=0, timed_out=False, interrupted_signal=None, error=None,
                       request_sha256=digest, gpu=0, gpu_uuid="GPU-CPU-fixture", supervisor_pid=12345,
                       evidence=worker, elapsed_seconds=elapsed, elapsed_GPUh=elapsed / 3600,
                       allocation_monotonic_start=1000., allocation_monotonic_end=cursor, start=2000., end=2000. + elapsed)
        for key in ("parameters", "call", "checkpoint", "model_content_sha256", "latest_smoke02", "prior_smoke_attempts"):
            receipt[key] = request[key]
        receipt["measurement_sources"] = [smoke.record(request_path), smoke.record(folder / "launch.json"),
            smoke.record(folder / "worker.json")] + [v["video"]["file"] for v in videos]
        self.batch_path = folder / "finished.json"
        smoke.exclusive(self.batch_path, receipt)
        self.throughput_sha = smoke.sha(self.batch_path.read_bytes())

    def prepare_dev(self, method="alg", nshards=2, cap=43200, timing=True):
        if not hasattr(self, "throughput_sha"):
            self.finish_smokes(method)
        return dev.prepare(method, self.pf, self.node_sha, nshards, cap, root=self.root,
                           approved_throughput_sha=self.throughput_sha if timing else None,
                           throughput_receipt=self.batch_path if timing else None)

    def verify(self, path, digest):
        return dev.verify_plan(path, digest, self.pf, self.node_sha)

    def mutate_plan(self, path, change):
        value = json.loads(path.read_bytes())
        change(value)
        path.write_bytes(smoke.encoded(value))
        return smoke.sha(path.read_bytes())

    def ledger(self, plan, digest):
        return dict(schema="teacher_devset_allocations_v1", method=plan["method"], plan_sha256=digest,
                    allocations=[dict(phase="generation", status="reserved", allocated_gpus=1,
                        shard=s, nshards=plan["nshards"], cap_seconds=cap, gpu_index=s,
                        hostname=os.uname().nodename, supervisor_pid=12345,
                        allocation_start_monotonic=time.monotonic(), start=time.time())
                        for s, cap in enumerate(plan["shard_cap_seconds"])])


class PrepareTests(Fixture):
    def test_frozen_basename_accepts_dots_plus_but_rejects_traversal(self):
        def manifest_with(name):
            cases = [dict(c) for c in self.cases]
            cases[1]["id"] = name
            raw = b"".join(smoke.encoded(c) for c in cases)
            self.cases_path.write_bytes(raw)
            return mock.patch.object(dev, "CASES_SHA256", smoke.sha(raw))
        with manifest_with("vfhq__Clip+sample.v1_w000"):
            self.assertEqual(len(dev.frozen_cases(self.root)[1]), 94)
        for name in ("../escape", "a/b", "a\\b", ".", "..", "a" + chr(0)):
            with manifest_with(name):
                with self.assertRaisesRegex(ValueError, "Unsafe case id"):
                    dev.frozen_cases(self.root)

    def test_default_prepare_no_torch_no_generator_no_metrics_import(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "gen_jobs", "teacher_alg", "teacher_dymos", "score_e1"):
                raise AssertionError("Forbidden GPU/score import in CPU prepare: " + name)
            return original(name, *args, **kwargs)
        with mock.patch("builtins.__import__", side_effect=guarded):
            path, _ = dev.prepare("alg", self.pf, self.node_sha, 2, 43200, root=self.root)
        value = json.loads(path.read_bytes())
        self.assertEqual(path.name, "plan.pending.json")
        self.assertEqual(value["admission"]["status"], "pending")
        self.assertFalse(value["GPU_used"])
        self.assertFalse(value["admission"]["mechanism_PASS"])
        self.assertEqual(len(value["jobs"]), 188)
        self.assertEqual([j["seed"] for j in value["jobs"]], [0] * 94 + [1] * 94)
        for seed in (0, 1):
            self.assertEqual((path.parent / f"seed{seed}/cases.jsonl").read_bytes(), self.cases_path.read_bytes())
        self.assertFalse(value["base_reuse"]["seed0"]["allowed"])
        self.assertFalse(value["base_reuse"]["seed1"]["allowed"])
        self.assertEqual(value["base_reuse"]["seed1"]["root"], str(self.root / dev.BASE_ROOTS["seed1"]))

    def test_pending_cannot_be_authorized(self):
        path, digest = dev.prepare("alg", self.pf, self.node_sha, root=self.root)
        with self.assertRaisesRegex(ValueError, "pending"):
            self.verify(path, digest)
        with self.assertRaisesRegex(ValueError, "gpu-authorized"):
            dev.run(path, digest, self.pf, self.node_sha, self.base / "missing.json", 0, 0, 1, 60, self.state)

    def test_ready_uses_segmented_batch_not_repeated_loading(self):
        self.finish_smokes(elapsed=600, load=400)
        path, digest = self.prepare_dev()
        plan, _ = self.verify(path, digest)
        self.assertEqual(path.name, "plan.json")
        self.assertEqual(plan["admission"]["estimated_generation_seconds"], 2 * (400 + 35) + 188 * 200)
        self.assertGreater(plan["admission"]["conservative_total_elapsed_upper_bound_seconds"], 86400)
        self.assertFalse(plan["admission"]["estimate_is_guarantee"])
        self.assertEqual(plan["phase_seconds"], dev.PHASE_SECONDS)

    def test_pass_without_segmented_measurement_is_blocked(self):
        self.finish_smokes(with_timing=False)
        path, _ = self.prepare_dev(timing=False)
        plan = json.loads(path.read_bytes())
        self.assertEqual(path.name, "plan.blocked.json")
        self.assertTrue(plan["admission"]["mechanism_PASS"])
        self.assertIsNone(plan["admission"]["estimated_generation_seconds"])

    def test_both_attempts_charged_and_generation_unaffordable_blocked(self):
        self.finish_smokes(elapsed=3500, first_elapsed=200, load=100)
        path, _ = self.prepare_dev()
        self.assertEqual(json.loads(path.read_bytes())["admission"]["status"], "blocked")
        self.assertEqual(json.loads(path.read_bytes())["admission"]["charged_or_reserved_seconds"], 3700)

    def test_true_pass_but_bad_batch_throughput_blocked(self):
        self.finish_smokes(elapsed=800, load=100)
        path, _ = self.prepare_dev()
        self.assertEqual(path.name, "plan.blocked.json")
        self.assertGreater(json.loads(path.read_bytes())["admission"]["estimated_generation_seconds"], 86400)

    def test_wrong_smoke_method_fails(self):
        self.finish_smokes()
        path = self.root / "eval/teacher_alg_smoke_20261026_02/request.json"
        self.change_request(path, lambda v: v.update(method="dymos"))
        with self.assertRaisesRegex(ValueError, "Wrong smoke"):
            self.prepare_dev()

    def test_false_pass_or_incomplete_mechanism_refused(self):
        self.finish_smokes()
        folder = self.root / "eval/teacher_alg_smoke_20261026_02"
        receipt = json.loads((folder / "finished.json").read_bytes())
        receipt["mechanism_PASS"] = False
        (folder / "finished.json").write_bytes(smoke.encoded(receipt))
        path, _ = self.prepare_dev()
        self.assertEqual(path.name, "plan.blocked.json")

    def test_smoke_media_hash_and_50_step_evidence_are_strict(self):
        self.finish_smokes()
        folder = self.root / "eval/teacher_alg_smoke_20261026_02"
        video = folder / "video.mp4"
        original_video = video.read_bytes()
        video.write_bytes(b"drifted smoke video")
        with self.assertRaisesRegex(ValueError, "media hash"):
            self.prepare_dev()
        video.write_bytes(original_video)
        # Restore metadata evidence after deliberately changing the fixture video.
        receipt = json.loads((folder / "finished.json").read_bytes())
        evidence = receipt["evidence"]
        evidence["video"] = self.fake_video(video, self.tools)
        evidence["stats"]["method"]["completed_steps"] = 49
        (folder / "finished.json").write_bytes(smoke.encoded(receipt))
        (folder / "worker.json").write_bytes(smoke.encoded(evidence))
        with self.assertRaisesRegex(ValueError, "50-step"):
            self.prepare_dev()

    def test_wrong_approved_plan_and_throughput_hashes_rejected(self):
        self.finish_smokes()
        self.throughput_sha = "0" * 64
        with self.assertRaisesRegex(ValueError, "performance receipt SHA"):
            self.prepare_dev()
        folder = self.root / "eval/teacher_alg_smoke_20261026_02"
        self.throughput_sha = smoke.sha(self.batch_path.read_bytes())
        path, _ = self.prepare_dev()
        with self.assertRaisesRegex(ValueError, "plan SHA"):
            self.verify(path, "0" * 64)

    def test_only_batch_receipt_not_legacy_single_sidecar_can_admit(self):
        self.finish_smokes(with_timing=False)
        folder = self.root / "eval/teacher_alg_smoke_20261026_02"
        sidecar = folder / "throughput.json"
        sidecar.write_bytes(smoke.encoded(dict(schema="teacher_smoke_throughput_v1", measured=True)))
        path, _ = self.prepare_dev()
        self.assertEqual(path.name, "plan.blocked.json")
        self.assertIsNone(json.loads(path.read_bytes())["admission"]["throughput"])

    def test_batch_prior_and_new_allocations_all_charged(self):
        self.finish_smokes(elapsed=400, first_elapsed=20, load=100)
        path, _ = self.prepare_dev()
        value = json.loads(path.read_bytes())
        self.assertEqual(value["admission"]["charged_or_reserved_seconds"], 420 + 100 + 2 * 300)
        self.assertEqual(value["admission"]["estimated_generation_seconds"], 2 * (100 + 35) + 188 * 300)
        self.assertFalse(value["base_reuse"]["seed0"]["allowed"])

    def test_batch_falsepass_bad_timers_seed_and_media_hash_rejected(self):
        self.finish_smokes()
        original = self.batch_path.read_bytes()
        changes = [lambda v: v.update(success=False, failure=True),
                   lambda v: v["evidence"]["videos"][1].update(seed=0),
                   lambda v: v["evidence"]["timers"]["load"].update(seconds=-1),
                   lambda v: v["evidence"]["videos"][0]["video"]["file"].update(sha256="0" * 64)]
        for change in changes:
            self.batch_path.write_bytes(original)
            value = json.loads(original)
            change(value)
            self.batch_path.write_bytes(smoke.encoded(value))
            self.throughput_sha = smoke.sha(self.batch_path.read_bytes())
            # Keep the two authenticated worker copies consistent to test deep validation.
            (self.batch_path.parent / "worker.json").write_bytes(smoke.encoded(value["evidence"]))
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.prepare_dev()
        self.batch_path.write_bytes(original)

    def test_unaccounted_single_video03_measurement_refused(self):
        self.finish_smokes()
        folder = self.root / "eval/teacher_alg_smoke_20261026_03"
        folder.mkdir()
        (folder / "launch.json").write_bytes(b"a separate allocation must not be free")
        with self.assertRaisesRegex(ValueError, "Unaccounted single-video03"):
            self.prepare_dev()

    def test_old_smoke_evidence_is_not_rewritten_as_current_source(self):
        self.finish_smokes(with_timing=False)
        folder = self.root / "eval/teacher_alg_smoke_20261026_02"
        old_request = (folder / "request.json").read_bytes()
        old_receipt = (folder / "finished.json").read_bytes()
        (self.root / "scripts/run_teacher_inference_smoke.py").write_text("# new timing instrumentation\n")
        with mock.patch.object(smoke, "verify_request", side_effect=AssertionError("Do not reattest historical sources")):
            path, _ = self.prepare_dev()
        self.assertEqual(path.name, "plan.blocked.json")
        self.assertEqual((folder / "request.json").read_bytes(), old_request)
        self.assertEqual((folder / "finished.json").read_bytes(), old_receipt)

    def test_current_batch_source_drift_refused(self):
        self.finish_smokes()
        (self.root / "scripts/measure_teacher_throughput.py").write_text("# changed current measurement source\n")
        with self.assertRaisesRegex(ValueError, "drift"):
            self.prepare_dev()

    def test_batch_fixture_audited_helper_preserves_original_attestation(self):
        self.finish_smokes(with_timing=False)
        folder = self.root / "eval/teacher_alg_smoke_20261026_02"
        original_request = (folder / "request.json").read_bytes()
        helper = self.root / "scripts/run_teacher_inference_smoke.py"
        helper.write_bytes(b"# Explicit CPU mock of audited observation-only helper\n")
        with mock.patch.object(dev.performance, "OBSERVATION_HELPER_SHA", smoke.sha(helper.read_bytes())):
            self.make_batch("alg", 100, 290)
            path, digest = self.prepare_dev()
            plan, _ = self.verify(path, digest)
        self.assertEqual(plan["admission"]["status"], "ready")
        self.assertEqual((folder / "request.json").read_bytes(), original_request)
        batch = json.loads((self.batch_path.parent / "request.json").read_bytes())
        proof = batch["historical_helper_attestation"]
        self.assertEqual(proof["source_sha256"], smoke.sha(self.original_helper))
        self.assertEqual(proof["commit"], dev.performance.HISTORY_COMMIT)

    def test_batch_fixture_bad_immutable_proof_or_unaudited_helper_rejected(self):
        self.finish_smokes(with_timing=False)
        manifest = self.history_manifest
        self.history_manifest = smoke.encoded(dict(files=[dict(path="scripts/run_teacher_inference_smoke.py",
            source_sha256="0" * 64, snapshot_sha256="0" * 64)]))
        with self.assertRaisesRegex(ValueError, "immutable published source"):
            self.make_batch("alg", 100, 290)
        self.history_manifest = manifest
        (self.root / "scripts/run_teacher_inference_smoke.py").write_bytes(b"# unaudited helper alteration\n")
        with self.assertRaisesRegex(ValueError, "Unapproved current smoke helper drift"):
            self.make_batch("alg", 100, 290)

    def test_missing_batch_finished_is_blocked_without_gpu_import(self):
        self.finish_smokes(with_timing=False)
        self.batch_path = self.base / "missing_performance/finished.json"
        self.throughput_sha = "a" * 64
        path, _ = self.prepare_dev()
        self.assertEqual(path.name, "plan.blocked.json")

    def test_full_case_manifest_drift_refused(self):
        self.cases_path.write_bytes(self.cases_path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "manifest SHA"):
            dev.prepare("alg", self.pf, self.node_sha, root=self.root)

    def test_shard_caps_cannot_each_be_24_hours(self):
        with self.assertRaisesRegex(ValueError, "sum <=24"):
            dev.prepare("alg", self.pf, self.node_sha, 2, 86400, root=self.root)

    def test_runtime_identical_bytes_other_install_mtime_are_allowed(self):
        path, digest = self.prepare_dev()
        runtime = smoke.runtime_records()
        moved = copy.deepcopy(runtime)
        for row in moved["files"]:
            row["mtime_ns"] += 1000000000
        with mock.patch.object(smoke, "runtime_records", return_value=moved):
            self.verify(path, digest)
        altered = copy.deepcopy(moved)
        altered["files"][0]["sha256"] = "0" * 64
        with mock.patch.object(smoke, "runtime_records", return_value=altered):
            with self.assertRaisesRegex(ValueError, "Runtime source/version"):
                self.verify(path, digest)

    def test_seed_parameter_source_job_and_base_drift_refused(self):
        path, _ = self.prepare_dev()
        original = path.read_bytes()
        changes = [lambda v: v.update(seeds=[0, 2]), lambda v: v["parameters"].update(factor=.5),
                   lambda v: v["jobs"][94].update(seed=0), lambda v: v.update(shard_cap_seconds=[86400, 86400]),
                   lambda v: v["base_reuse"]["seed1"].update(root=v["base_reuse"]["seed0"]["root"]),
                   lambda v: v.update(method="dymos")]
        for change in changes:
            path.write_bytes(original)
            digest = self.mutate_plan(path, change)
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError)):
                self.verify(path, digest)
        path.write_bytes(original)
        (self.root / "scripts/teacher_alg.py").write_text("# Drift\n")
        with self.assertRaisesRegex(ValueError, "drift"):
            self.verify(path, smoke.sha(original))

    def test_unowned_old_output_root_never_overwritten(self):
        folder = dev.output_root(self.root, "alg")
        folder.mkdir()
        old = folder / "old.mp4"
        old.write_bytes(b"do not delete")
        with self.assertRaisesRegex(ValueError, "unowned"):
            self.prepare_dev()
        self.assertEqual(old.read_bytes(), b"do not delete")


class LedgerTests(Fixture):
    def test_all_shards_reserved_and_phase_caps_independent(self):
        path, digest = self.prepare_dev()
        plan = json.loads(path.read_bytes())
        ledger = self.ledger(plan, digest)
        item, totals = dev.validate_ledger(ledger, plan, digest, 0, 2, 0, 43200)
        self.assertEqual(item["status"], "reserved")
        self.assertEqual(totals["generation"], 86400)
        for change in (lambda v: v["allocations"].pop(), lambda v: v.update(method="dymos"),
                       lambda v: v["allocations"][0].update(status="running"),
                       lambda v: v["allocations"][0].update(cap_seconds=86400),
                       lambda v: v["allocations"].append(dict(phase="score", status="reserved", allocated_gpus=2, cap_seconds=30000)),
                       lambda v: v["allocations"].append(dict(phase="failure", status="reserved", allocated_gpus=1, cap_seconds=14401))):
            bad = copy.deepcopy(ledger)
            change(bad)
            with self.subTest(change=change), self.assertRaises(ValueError):
                dev.validate_ledger(bad, plan, digest, 0, 2, 0, 43200)

    def test_failure_not_free_and_finished_over_cap_charged(self):
        path, digest = self.prepare_dev()
        plan = json.loads(path.read_bytes())
        ledger = self.ledger(plan, digest)
        ledger["allocations"][1].update(status="finished", success=False, start=100,
                                          end=43301, elapsed_seconds=43201)
        with self.assertRaisesRegex(ValueError, "budget exceeded"):
            dev.validate_ledger(ledger, plan, digest, 0, 2, 0, 43200)

    def test_real_blocking_ledger_lock_remains_deadline_interruptible(self):
        lock_path = self.base / "blocked_ledger.lock"
        ctx = multiprocessing.get_context("fork")
        reports = ctx.Queue()
        def wait_for_ledger():
            try:
                # Close inherited parent description; this process uses its own flock.
                inherited.close()
                with lock_path.open("a") as lock:
                    with dev.deadline(time.monotonic() - 24.8, 60):
                        dev.fcntl.flock(lock, dev.fcntl.LOCK_EX)
                reports.put("unexpected acquired")
            except TimeoutError:
                reports.put("deadline interrupted")
        with lock_path.open("a") as inherited:
            dev.fcntl.flock(inherited, dev.fcntl.LOCK_EX)
            process = ctx.Process(target=wait_for_ledger)
            process.start()
            try:
                self.assertEqual(reports.get(timeout=5), "deadline interrupted")
                process.join(timeout=3)
                self.assertEqual(process.exitcode, 0)
            finally:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=3)
                reports.close()
                reports.join_thread()

    def test_deadline_restores_signal_and_no_timer(self):
        before = signal.getsignal(signal.SIGALRM)
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with dev.deadline(time.monotonic(), 60):
                raise RuntimeError("fixture")
        self.assertEqual(signal.getsignal(signal.SIGALRM), before)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "exhausted"):
            with dev.deadline(time.monotonic() - 61, 60):
                pass


class GenerationTests(Fixture):
    def fake_pipeline(self, method="alg", failure=None):
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(max_memory_allocated=lambda _: 123,
                                                               synchronize=lambda _: None), compile=lambda fn, **_: fn)
        states = []
        class Pipe:
            dit = object()
            marker = "baseline"
            def __call__(pipe, **kwargs):
                states.append((pipe.marker, kwargs["seed"], kwargs["prompt"]))
                for _ in kwargs["progress_bar_cmd"](list(range(50))):
                    pass
                if failure:
                    raise RuntimeError(failure)
                return [object()] * 121
        pipe = Pipe()
        @contextlib.contextmanager
        def context(pipe, call_kwargs, checkpoint, **params):
            self.assertEqual(pipe.marker, "baseline")
            self.assertEqual(checkpoint, "NONE")
            self.assertEqual(params, dev.PARAMS[method])
            original = pipe.marker
            pipe.marker = method
            try:
                yield method_stats(method)
            finally:
                pipe.marker = original
        @contextlib.contextmanager
        def observe(pipe, torch, stats, check):
            try:
                yield
            finally:
                stats.update({k: v for k, v in observed(method).items() if k != "progress_steps"})
        def save(video, path, fps, quality):
            self.assertEqual(len(video), 121)
            self.assertEqual((fps, quality), (24, 5))
            Path(path).write_bytes(b"CPU fake encoded video")
        generator = types.SimpleNamespace(load_image=lambda p: p, save_video=save)
        return pipe, torch, generator, context, observe, states

    def generate(self, method="alg", failure=None):
        path, digest = self.prepare_dev(method)
        plan = json.loads(path.read_bytes())
        pipe, torch, generator, context, observer, states = self.fake_pipeline(method, failure)
        return plan, digest, pipe, torch, generator, context, observer, states

    def test_same_baseline_restored_each_seed_and_strict_reuse(self):
        plan, digest, pipe, torch, generator, context, observer, states = self.generate()
        with mock.patch.object(smoke, "observe", observer), mock.patch.object(smoke, "parameter_hashes", return_value={"w": "same"}):
            for index in (0, 94):
                job = plan["jobs"][index]
                result = dev.generate_case(pipe, torch, generator, context, plan, digest, job, lambda: None)
                self.assertTrue(result["success"])
                self.assertEqual(pipe.marker, "baseline")
                self.assertEqual(dev.strict_reuse(job, plan, digest), result)
        self.assertEqual([v[1] for v in states], [0, 1])
        job = plan["jobs"][0]
        Path(job["output"]).write_bytes(b"same path, drifted media")
        with self.assertRaisesRegex(ValueError, "hash/decode"):
            dev.strict_reuse(job, plan, digest)

    def test_failure_restores_context_preserves_failed_receipt_and_claim(self):
        plan, digest, pipe, torch, generator, context, observer, _ = self.generate(failure="inference failure")
        job = plan["jobs"][0]
        with mock.patch.object(smoke, "observe", observer), mock.patch.object(smoke, "parameter_hashes", return_value={"w": "same"}):
            with self.assertRaisesRegex(RuntimeError, "inference failure"):
                dev.generate_case(pipe, torch, generator, context, plan, digest, job, lambda: None)
        self.assertEqual(pipe.marker, "baseline")
        output = Path(job["output"])
        self.assertFalse(json.loads(output.with_suffix(".receipt.json").read_bytes())["success"])
        self.assertTrue(output.with_suffix(".claim.json").is_file())
        with self.assertRaisesRegex(ValueError, "receipt"):
            dev.strict_reuse(job, plan, digest)

    def test_run_fake_pipeline_once_per_shard_and_caller_finalizes_cost(self):
        plan, digest, pipe, fake_torch, generator, context, observer, states = self.generate()
        path = Path(plan["output_root"]) / "plan.json"
        ledger_path = self.base / "independent_method_ledger.json"
        ledger = self.ledger(plan, digest)
        ledger_path.write_bytes(smoke.encoded(ledger))
        fake_torch.cuda.device_count = lambda: 1
        fake_torch.cuda.set_device = lambda _: None
        fake_torch.cuda.reset_peak_memory_stats = lambda _: None
        generator.NEG = plan["negative_prompt"]
        method_module = types.SimpleNamespace(teacher_alg=context)
        before_path = list(sys.path)
        try:
            with mock.patch.dict(sys.modules, {"torch": fake_torch, "gen_jobs": generator, "teacher_alg": method_module}), \
                    mock.patch.object(dev, "managed_scope", return_value="GPU-CPU-fixture"), \
                    mock.patch.object(smoke, "observe", observer), \
                    mock.patch.object(smoke, "parameter_hashes", return_value={"w": "same"}), \
                    mock.patch.object(smoke, "build_pipe", return_value=pipe) as build:
                result = dev.run(path, digest, self.pf, self.node_sha, ledger_path,
                                 0, 0, 2, 43200, self.state, gpu_authorized=True)
                self.assertEqual(build.call_count, 1)
                self.assertEqual(build.call_args.args[0], "NONE")
            self.assertTrue(result["success"], result["error"])
            self.assertFalse(result["allocation_released"])
            self.assertEqual(len(result["jobs"]), 94)
            self.assertEqual(pipe.marker, "baseline")
            self.assertEqual({seed for _, seed, _ in states}, {0, 1})
            updated = json.loads(ledger_path.read_bytes())
            self.assertEqual(updated["allocations"][0]["status"], "running")
            self.assertTrue(updated["allocations"][0]["worker_complete"])
            self.assertEqual(updated["allocations"][1], ledger["allocations"][1])
            with self.assertRaisesRegex(ValueError, "reservation"):
                dev.run(path, digest, self.pf, self.node_sha, ledger_path,
                        0, 0, 2, 43200, self.state, gpu_authorized=True)
        finally:
            sys.path[:] = before_path

    def test_real_two_workers_shared_ledger_transactions_wait_without_lost_updates(self):
        plan, digest, pipe, fake_torch, generator, context, observer, _ = self.generate()
        plan["jobs"] = []  # CPU transaction-only fixture; never claims real media.
        path = Path(plan["output_root"]) / "plan.json"
        ledger_path = self.base / "concurrent_method_ledger.json"
        ledger = self.ledger(plan, digest)
        ledger_path.write_bytes(smoke.encoded(ledger))
        fake_torch.cuda.device_count = lambda: 1
        fake_torch.cuda.set_device = lambda _: None
        fake_torch.cuda.reset_peak_memory_stats = lambda _: None
        generator.NEG = plan["negative_prompt"]
        ctx = multiprocessing.get_context("fork")
        barrier, reports = ctx.Barrier(2), ctx.Queue()
        original_checkpoint = dev.checkpoint
        def slow_transaction(target, value):
            if Path(target) == ledger_path:
                time.sleep(0.1)  # Force real overlapping read/modify/write transactions.
            original_checkpoint(target, value)
        def worker(shard):
            try:
                barrier.wait(timeout=5)
                result = dev.run(path, digest, self.pf, self.node_sha, ledger_path,
                                 shard, shard, 2, 43200, self.state, gpu_authorized=True)
                reports.put((shard, result["success"], result["error"]))
            except BaseException as exc:
                reports.put((shard, False, f"{type(exc).__name__}: {exc}"))
                raise
        processes = []
        try:
            with mock.patch.dict(sys.modules, {"torch": fake_torch, "gen_jobs": generator,
                                               "teacher_alg": types.SimpleNamespace(teacher_alg=context)}), \
                    mock.patch.object(dev, "verify_plan", return_value=(plan, [])), \
                    mock.patch.object(dev, "managed_scope", return_value="GPU-CPU-fixture"), \
                    mock.patch.object(smoke, "build_pipe", return_value=pipe), \
                    mock.patch.object(dev, "checkpoint", side_effect=slow_transaction):
                processes = [ctx.Process(target=worker, args=(shard,)) for shard in (0, 1)]
                for process in processes:
                    process.start()
                results = [reports.get(timeout=10) for _ in processes]
                for process in processes:
                    process.join(timeout=5)
                    self.assertEqual(process.exitcode, 0)
            self.assertTrue(all(success for _, success, _ in results), results)
            updated = json.loads(ledger_path.read_bytes())
            self.assertEqual(len(updated["allocations"]), 2)
            for shard, process in enumerate(processes):
                item = updated["allocations"][shard]
                self.assertEqual(item["worker_pid"], process.pid)
                self.assertEqual(item["status"], "running")
                self.assertTrue(item["worker_complete"])
                self.assertTrue(item["worker_success"])
                self.assertNotIn("end", item)  # Worker completion is still not free finalization.
                self.assertEqual(item["cap_seconds"], 43200)
                self.assertGreater(item["worker_elapsed_seconds"], 0)
                receipt = Path(plan["output_root"]) / f"shard{shard}.finished.json"
                self.assertEqual(item["receipt_sha256"], smoke.sha(receipt.read_bytes()))
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=3)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=3)
            reports.close()
            reports.join_thread()

    def test_exclusive_case_claim_not_stolen(self):
        plan, digest, pipe, torch, generator, context, observer, _ = self.generate()
        job = plan["jobs"][0]
        claim = Path(job["output"]).with_suffix(".claim.json")
        claim.write_bytes(b"old worker claim")
        with self.assertRaises(FileExistsError):
            dev.generate_case(pipe, torch, generator, context, plan, digest, job, lambda: None)
        self.assertEqual(claim.read_bytes(), b"old worker claim")
        self.assertEqual(pipe.marker, "baseline")

    def test_old_unbound_video_is_not_reused_or_deleted(self):
        plan, digest, pipe, torch, generator, context, observer, _ = self.generate()
        job = plan["jobs"][0]
        output = Path(job["output"])
        output.write_bytes(b"unbound old video")
        with self.assertRaisesRegex(ValueError, "Old output"):
            dev.strict_reuse(job, plan, digest)
        self.assertEqual(output.read_bytes(), b"unbound old video")

    def test_dymos_restores_compiler_and_never_mix_method_receipts(self):
        plan, digest, pipe, torch, generator, context, observer, _ = self.generate("dymos")
        compile_before = torch.compile
        job = plan["jobs"][0]
        with mock.patch.object(smoke, "observe", observer), mock.patch.object(smoke, "parameter_hashes", return_value={"w": "same"}):
            result = dev.generate_case(pipe, torch, generator, context, plan, digest, job, lambda: None)
        self.assertIs(torch.compile, compile_before)
        self.assertEqual(pipe.marker, "baseline")
        self.assertEqual(result["compile"]["compiled_functions"], 0)
        bad = copy.deepcopy(plan)
        bad["method"] = "alg"
        with self.assertRaisesRegex(ValueError, "Unbound"):
            dev.strict_reuse(job, bad, digest)

    def test_compile_wrapper_times_and_restores_on_failure(self):
        _, torch, _, _, _, _ = self.fake_pipeline("dymos")
        original = torch.compile
        stats = {}
        with self.assertRaisesRegex(RuntimeError, "compiled failure"):
            with dev.compile_timing(torch, stats):
                def failing():
                    raise RuntimeError("compiled failure")
                torch.compile(failing)()
        self.assertIs(torch.compile, original)
        self.assertEqual(stats["compiled_functions"], 1)
        self.assertGreaterEqual(stats["first_compiled_call_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
