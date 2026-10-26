"""CPU-only fake-pipeline timing regressions; fixtures are never GPU evidence."""
import builtins
import contextlib
import importlib.util
import io
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("throughput_fixture", ROOT / "tests/test_teacher_inference_smoke.py")
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)
import measure_teacher_throughput as perf
smoke = fixture.smoke
REAL_VALIDATE_VIDEO = smoke.validate_video
REAL_SUBPROCESS_RUN = subprocess.run
REAL_OUTPUT_ROOT = perf.output_root


class Fixture(fixture.Fixture):
    def setUp(self):
        super().setUp()
        smoke_patch = mock.patch.object(perf, "smoke", smoke)
        smoke_patch.start()
        self.addCleanup(smoke_patch.stop)
        self.fixed_folder = mock.patch.object(perf, "output_root")
        self.output_mock = self.fixed_folder.start()
        self.addCleanup(self.fixed_folder.stop)
        helper_path = self.root / "scripts/run_teacher_inference_smoke.py"
        self.original_helper = helper_path.read_bytes()
        self.published_helper = self.original_helper
        self.history_manifest = smoke.encoded(dict(files=[dict(path="scripts/run_teacher_inference_smoke.py",
            source_sha256=smoke.sha(self.original_helper), snapshot_sha256=smoke.sha(self.original_helper))]))
        def git_fixture(command, **kwargs):
            if command[0] == "git":
                data = self.history_manifest if command[-1].endswith("PUBLICATION_MANIFEST.json") else self.published_helper
                return types.SimpleNamespace(stdout=data)
            return REAL_SUBPROCESS_RUN(command, **kwargs)
        git_patch = mock.patch.object(perf.subprocess, "run", side_effect=git_fixture)
        git_patch.start()
        self.addCleanup(git_patch.stop)
        (self.root / "tests").mkdir()
        (self.root / "scripts/measure_teacher_throughput.py").write_text("# CPU measurement fixture\n")
        (self.root / "tests/test_teacher_throughput.py").write_text("# CPU regression fixture\n")
        patch = mock.patch.object(smoke, "validate_video", side_effect=self.fake_decode)
        patch.start()
        self.addCleanup(patch.stop)
        self.latest = {}
        for method in smoke.PARAMS:
            for date, suffix in (("20261025", "01"), ("20261026", "02")):
                folder = self.root / f"eval/teacher_{method}_smoke_{date}_{suffix}"
                path, digest = smoke.prepare(method, folder, self.pf, smoke.sha(self.pf.read_bytes()), root=self.root)
                launch = dict(request_sha256=digest, budget=smoke.BUDGET)
                observed = dict(model_forwards=110 if method == "alg" else 100, scheduler_steps=50, progress_steps=50, finite_checks=200)
                stats = dict(completed_steps=50, active_steps=10, model_forwards=observed["model_forwards"], bypassed=False)
                if method == "alg":
                    stats["first_frame_sha256"] = "a" * 64
                else:
                    stats.update(biased_attention_calls=300, original_attention_calls=2700, reference_tokens=880, sequence_tokens=27280)
                (folder / "video.mp4").write_bytes(b"CPU fixture original full-format video")
                evidence = dict(success=True, mechanism_PASS=True, parameters_unchanged=True, parameters_before={"weight": "CPU"},
                                parameters_after={"weight": "CPU"}, stats=dict(method=stats, observed=observed),
                                video=self.fake_decode(folder / "video.mp4", self.tools))
                receipt = dict(schema="teacher_smoke_receipt_v1", method=method, request_sha256=digest, allocated_gpus=1,
                               elapsed_seconds=100.0, elapsed_GPUh=100.0 / 3600, start=1000., end=1100.,
                               success=suffix == "02", mechanism_PASS=suffix == "02", owned_tree_released=True,
                               budget_exceeded=False, exit_code=0 if suffix == "02" else 1, timed_out=False,
                               interrupted_signal=None, error=None, evidence=evidence, budget=smoke.BUDGET)
                smoke.exclusive(folder / "launch.json", launch)
                smoke.exclusive(folder / "finished.json", receipt)
                smoke.exclusive(folder / "worker.json", evidence)
                if suffix == "02":
                    self.latest[method] = smoke.sha((folder / "finished.json").read_bytes())

    def fake_decode(self, path, tools):
        return dict(file=smoke.record(path, nonempty=True), decoded_frames=121, width=1280, height=704, fps=24, full_decode_xerror=True)

    def prepare(self, method="alg", name="performance"):
        self.output_mock.return_value = self.base / name
        return perf.prepare(method, self.base / name, self.latest[method], self.root)

    def rewrite(self, path, change):
        value = json.loads(path.read_bytes())
        change(value)
        path.write_bytes(smoke.encoded(value))
        return smoke.sha(path.read_bytes())

    def fake_runtime(self, request, fail_seed=None, mutate=False, frames=121):
        import torch
        torch.set_num_threads(1)
        state, calls, contexts = {}, [], []
        method_name = request["method"]
        dit = torch.nn.Linear(2, 2, bias=False)
        scheduler = types.SimpleNamespace(timesteps=torch.arange(50))
        scheduler.step = lambda prediction, timestep, sample: sample - prediction * 0.001
        class FakePipeline:
            def __init__(self):
                self.dit, self.scheduler = dit, scheduler
                self.model_fn = self.predict
            def predict(self, **kwargs):
                state["stats"]["model_forwards"] += 1
                return kwargs["latents"] * 0.01
            def __call__(self, **kwargs):
                calls.append(dict(kwargs))
                if kwargs["seed"] == fail_seed:
                    raise RuntimeError("CPU fake inference failure")
                latent = torch.ones(1, 2, 3, 2, 2)
                reference = latent[:, :, :1].clone()
                stats = state["stats"]
                for index, timestep in enumerate(kwargs["progress_bar_cmd"](self.scheduler.timesteps)):
                    forwards = 3 if method_name == "alg" and index < 10 else 2
                    for _ in range(forwards):
                        prediction = self.model_fn(latents=latent, first_frame_latents=reference, context=torch.ones(1, 2, 2))
                    latent = self.scheduler.step(prediction, timestep, latent)
                    stats["completed_steps"] += 1
                    stats["active_steps"] += int(index < 10)
                    if method_name == "dymos":
                        stats["biased_attention_calls"] += 30 * int(index < 10)
                        stats["original_attention_calls"] += 30 * (1 if index < 10 else 2)
                if mutate:
                    with torch.no_grad():
                        self.dit.weight.add_(1)
                return [None] * frames
        @contextlib.contextmanager
        def method(pipe, **kwargs):
            self.assertEqual(kwargs["checkpoint"], "NONE")
            self.assertEqual({k: kwargs[k] for k in request["parameters"]}, request["parameters"])
            stats = dict(completed_steps=0, active_steps=0, model_forwards=0, bypassed=False)
            if method_name == "alg":
                stats["first_frame_sha256"] = "b" * 64
            else:
                stats.update(biased_attention_calls=0, original_attention_calls=0, reference_tokens=880, sequence_tokens=27280)
            state["stats"] = stats
            contexts.append(stats)
            yield stats
        generator = types.ModuleType("gen_jobs")
        generator.NEG = request["negative_prompt"]
        generator.load_image = mock.Mock(return_value="CPU fake image")
        def save(video, path, fps, quality):
            self.assertEqual((len(video), fps, quality), (121, 24, 5))
            Path(path).write_bytes(("CPU fake full video " + Path(path).name).encode())
        generator.save_video = mock.Mock(side_effect=save)
        module = types.ModuleType("teacher_" + method_name)
        setattr(module, "teacher_" + method_name, method)
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.dict(sys.modules, {"gen_jobs": generator, "teacher_" + method_name: module}))
        build = stack.enter_context(mock.patch.object(smoke, "build_pipe", return_value=FakePipeline()))
        for name, value in (("device_count", 1), ("set_device", None), ("reset_peak_memory_stats", None),
                            ("max_memory_allocated", 1234), ("synchronize", None), ("empty_cache", None)):
            stack.enter_context(mock.patch.object(torch.cuda, name, return_value=value))
        return stack, build, calls, contexts, generator

    def successful_receipt(self, method="alg", name="performance"):
        path, digest = self.prepare(method, name)
        request = perf.verify_request(path, digest)
        stack, build, calls, contexts, generator = self.fake_runtime(request)
        def launch(command, **kwargs):
            start = float(command[command.index("--allocation-start") + 1])
            result = perf.execute_worker(request, start, digest)
            self.assertTrue(result["success"], result.get("error"))
            return types.SimpleNamespace(pid=99999999, returncode=0, poll=lambda: 0, wait=lambda **kw: 0)
        with stack, mock.patch.object(smoke, "supervisor_scope", return_value="GPU-CPU-fixture"), \
             mock.patch.object(perf.subprocess, "Popen", side_effect=launch), mock.patch.object(smoke, "stop_group"):
            result = perf.run(path, digest, 2, 12345, self.state, True)
        self.assertTrue(result["success"], result.get("error"))
        return path.parent / "finished.json", result, (build, calls, contexts, generator)


class PrepareTests(Fixture):
    def test_cpu_prepare_freezes_original_request_sources_models_and_no_cuda(self):
        original_import = builtins.__import__
        def reject(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "gen_jobs", "diffsynth", "teacher_alg", "teacher_dymos"):
                raise AssertionError("CPU prepare imported GPU runtime")
            return original_import(name, *args, **kwargs)
        with mock.patch("builtins.__import__", side_effect=reject):
            path, digest = self.prepare()
            value = perf.verify_request(path, digest)
        self.assertFalse(value["GPU_used"])
        self.assertEqual(value["seeds"], [0, 1])
        self.assertEqual(value["video_count"], 2)
        self.assertEqual(len(value["models"]), 9)
        self.assertEqual(value["prior_smoke_seconds"], 200.)
        self.assertEqual(value["latest_smoke02"]["sha256"], self.latest["alg"])
        self.assertEqual(value["call"], smoke.SPEC)
        self.assertEqual(value["parameters"], {"factor": .4})
        self.assertEqual(path.parent.stat().st_mode & 0o077, 0)
        self.assertFalse((path.parent / "finished.json").exists())

    def test_default_cli_is_cpu_and_gpu_args_rejected(self):
        with mock.patch.object(perf, "prepare", return_value=(Path("private/request.json"), "a" * 64)) as prepare, mock.patch("builtins.print"):
            perf.main(["--method", "alg", "--request-dir", "private", "--approved-smoke02-sha256", "b" * 64])
        prepare.assert_called_once()
        with mock.patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
            perf.main(["--method", "alg", "--request-dir", "private", "--approved-smoke02-sha256", "b" * 64, "--gpu", "2"])

    def test_fixed_maximum_budget_and_timeout(self):
        self.assertEqual(perf.BUDGET["gpu_hours"], .75)
        self.assertLess(perf.BUDGET["outer_cap_seconds"], .75 * 3600)
        self.assertLess(perf.BUDGET["timeout_seconds"] + perf.BUDGET["kill_seconds"], 2700)
        path, digest = self.prepare()
        command = perf.run_command(perf.verify_request(path, digest), digest, 2, 12345, self.state, 100.)
        self.assertIn("2600s", command)
        self.assertIn("--kill-after=30s", command)
        self.assertEqual(command[command.index("--gpus") + 1], "2")
        self.assertIn("measure_teacher_throughput.py", " ".join(command))

    def test_wrong_original_receipt_sha_and_missing_original_evidence_block(self):
        with self.assertRaisesRegex(ValueError, "latest smoke02 SHA"):
            self.output_mock.return_value = self.base / "wrong"
            perf.prepare("alg", self.base / "wrong", "f" * 64, self.root)
        (self.root / "eval/teacher_alg_smoke_20261025_01/finished.json").rename(self.base / "missing.json")
        with self.assertRaisesRegex(ValueError, "Missing original"):
            self.prepare()

    def test_original_smoke_plus_reserved_performance_exceeds_one_rejected(self):
        path = self.root / "eval/teacher_alg_smoke_20261025_01/finished.json"
        self.rewrite(path, lambda v: v.update(elapsed_seconds=1000., elapsed_GPUh=1000. / 3600, end=v["start"] + 1000))
        with self.assertRaisesRegex(ValueError, "exceeds 1 GPUh"):
            self.prepare()

    def test_unapproved_reduction_or_budget_or_seed_rejected(self):
        for index, change in enumerate((lambda v: v["call"].update(num_inference_steps=10),
                                        lambda v: v["call"].update(num_frames=61),
                                        lambda v: v["budget"].update(gpu_hours=.8),
                                        lambda v: v.update(seeds=[0]),
                                        lambda v: v.update(parameters={"factor": .2}))):
            path, _ = self.prepare(name=f"changed{index}")
            digest = self.rewrite(path, change)
            with self.assertRaisesRegex(ValueError, "Unapproved"):
                perf.verify_request(path, digest)

    def test_request_sha_source_and_model_drift_rejected(self):
        path, digest = self.prepare()
        with self.assertRaisesRegex(ValueError, "request SHA"):
            perf.verify_request(path, "f" * 64)
        (self.root / "scripts/measure_teacher_throughput.py").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "drift"):
            perf.verify_request(path, digest)

    def test_model_binding_cannot_be_manually_rewritten(self):
        path, _ = self.prepare()
        digest = self.rewrite(path, lambda v: v["models"][0].update(sha256="f" * 64))
        with self.assertRaisesRegex(ValueError, "model/config/input/runtime mismatch"):
            perf.verify_request(path, digest)

    def test_historical_helper_pure_observation_patch_is_allowed_but_new_freeze_strict(self):
        helper = self.root / "scripts/run_teacher_inference_smoke.py"
        helper.write_text("# CPU fixture audited observation patch\nNEG = \"original negative\"\n")
        with mock.patch.object(perf, "OBSERVATION_HELPER_SHA", smoke.record(helper)["sha256"]):
            path, digest = self.prepare()
            request = perf.verify_request(path, digest)
            historical = next(r for r in request["original_source_files"] if r["path"] == str(helper))
            current = next(r for r in request["source_files"] if r["path"] == str(helper))
            self.assertNotEqual(historical["sha256"], current["sha256"])
            self.assertEqual(request["historical_helper_attestation"]["source_sha256"], historical["sha256"])
            helper.write_text("# unaudited algorithm change\n")
            with self.assertRaisesRegex(ValueError, "Unapproved current smoke helper drift"):
                perf.verify_request(path, digest)

    def test_redacted_snapshot_sha_is_not_original_source_sha(self):
        self.published_helper = b"# CPU redacted public snapshot with /srv path\n"
        self.history_manifest = smoke.encoded(dict(files=[dict(path="scripts/run_teacher_inference_smoke.py",
            source_sha256=smoke.sha(self.original_helper), snapshot_sha256=smoke.sha(self.published_helper))]))
        path, digest = self.prepare()
        request = perf.verify_request(path, digest)
        proof = request["historical_helper_attestation"]
        self.assertEqual(proof["source_sha256"], smoke.sha(self.original_helper))
        self.assertEqual(proof["snapshot_sha256"], smoke.sha(self.published_helper))
        self.assertNotEqual(proof["source_sha256"], proof["snapshot_sha256"])

    def test_immutable_published_helper_mismatch_rejected(self):
        self.history_manifest = smoke.encoded(dict(files=[dict(path="scripts/run_teacher_inference_smoke.py",
            source_sha256="f" * 64, snapshot_sha256="f" * 64)]))
        with self.assertRaisesRegex(ValueError, "immutable published source"):
            self.prepare()

    def test_current_algorithm_model_runtime_and_inputs_must_still_match_old_content(self):
        for name in ("algorithm", "model", "runtime", "image"):
            with self.subTest(name=name):
                if name == "algorithm":
                    changed = self.root / "scripts/teacher_alg.py"
                    raw, metadata = changed.read_bytes(), changed.stat()
                    changed.write_text("# CPU unaudited algorithm change\n")
                    context = contextlib.nullcontext()
                elif name == "model":
                    changed = Path(self.models[0]["path"])
                    raw, metadata = changed.read_bytes(), changed.stat()
                    changed.write_bytes(b"CPU changed weights")
                    context = contextlib.nullcontext()
                elif name == "image":
                    changed = Path(json.loads(self.cases_path.read_text().splitlines()[0])["image"])
                    raw, metadata = changed.read_bytes(), changed.stat()
                    changed.write_bytes(b"CPU changed image")
                    context = contextlib.nullcontext()
                else:
                    changed = None
                    context = mock.patch.object(smoke, "runtime_records", return_value=dict(self.runtime, packages={"changed": "runtime"}))
                with context, self.assertRaises(ValueError):
                    self.prepare(name="bad_" + name)
                if changed:
                    changed.write_bytes(raw)
                    os.utime(changed, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))

    def test_only_fixed_method_directory_authorized_and_extra03_allocation_rejected(self):
        fixed = REAL_OUTPUT_ROOT(self.root, "alg")
        self.output_mock.return_value = fixed
        with self.assertRaisesRegex(ValueError, "unique fixed method"):
            perf.prepare("alg", self.base / "unapproved", self.latest["alg"], self.root)
        extra = self.root / "eval/teacher_alg_smoke_20261026_03"
        extra.mkdir()
        (extra / "launch.json").write_bytes(smoke.encoded(dict(CPU=True)))
        with self.assertRaisesRegex(ValueError, "Unaccounted single-video03"):
            perf.prepare("alg", fixed, self.latest["alg"], self.root)

    def test_prepare_never_overwrites(self):
        path, _ = self.prepare()
        raw = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.prepare()
        self.assertEqual(path.read_bytes(), raw)


class WorkerTests(Fixture):
    def test_once_load_two_seeds_full_stats_format_and_real_recomputable_timers(self):
        for method in ("alg", "dymos"):
            with self.subTest(method=method):
                path, digest = self.prepare(method, method + "perf")
                request = perf.verify_request(path, digest)
                stack, build, calls, contexts, generator = self.fake_runtime(request)
                start = time.monotonic()
                with stack:
                    result = perf.execute_worker(request, start, digest)
                self.assertTrue(result["success"], result["error"])
                self.assertEqual(result["load_count"], 1)
                build.assert_called_once()
                self.assertEqual([v["seed"] for v in calls], [0, 1])
                self.assertEqual(generator.save_video.call_count, 2)
                self.assertIsNot(contexts[0], contexts[1])
                self.assertTrue(result["parameters_unchanged"])
                self.assertEqual(result["parameters_before"], result["parameters_after"])
                load, videos = perf.validate_timers(result, start, time.monotonic())
                self.assertGreater(load, 0)
                self.assertEqual(len(videos), 2)
                for video in result["videos"]:
                    smoke.validate_stats(method, video["stats"]["method"], video["stats"]["observed"])
                    self.assertEqual((video["video"]["decoded_frames"], video["video"]["width"], video["video"]["height"], video["video"]["fps"]), (121, 1280, 704, 24))
                self.assertNotEqual(result["videos"][0]["video"]["file"]["sha256"], result["videos"][1]["video"]["file"]["sha256"])
                self.assertEqual(json.loads((path.parent / "worker.json").read_bytes()), result)

    def test_fake_monotonic_clock_segments_are_exactly_recomputable(self):
        path, digest = self.prepare()
        request = perf.verify_request(path, digest)
        stack, build, calls, contexts, generator = self.fake_runtime(request)
        clock = itertools.count(1000., .01)
        with stack, mock.patch.object(perf.time, "monotonic", side_effect=lambda: next(clock)):
            start = perf.time.monotonic()
            result = perf.execute_worker(request, start, digest)
            end = perf.time.monotonic()
        self.assertTrue(result["success"], result["error"])
        load, videos = perf.validate_timers(result, start, end)
        self.assertGreater(load, 0.)
        self.assertEqual(len(videos), 2)
        self.assertLess(load + sum(videos), end - start)

    def test_two_cpu_encoded_mp4s_pass_real_full_format_decode(self):
        if not all(self.tools.values()):
            self.skipTest("ffmpeg/ffprobe not installed")
        path, digest = self.prepare()
        request = perf.verify_request(path, digest)
        stack, build, calls, contexts, generator = self.fake_runtime(request)
        def encode(video, output, fps, quality):
            color = "gray" if Path(output).name == "seed0.mp4" else "blue"
            subprocess.run([self.tools["ffmpeg"], "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                            f"color=c={color}:s=1280x704:r=24", "-frames:v", "121", "-c:v", "mpeg4",
                            "-threads", "1", "-pix_fmt", "yuv420p", output], check=True, capture_output=True, timeout=60)
        def decode(output, tools):
            if Path(output).name in ("seed0.mp4", "seed1.mp4"):
                return REAL_VALIDATE_VIDEO(output, tools)
            return self.fake_decode(output, tools)
        with stack, mock.patch.object(generator, "save_video", side_effect=encode), mock.patch.object(smoke, "validate_video", side_effect=decode):
            result = perf.execute_worker(request, time.monotonic(), digest)
        self.assertTrue(result["success"], result["error"])
        for video in result["videos"]:
            actual = REAL_VALIDATE_VIDEO(video["video"]["file"]["path"], self.tools)
            self.assertEqual(actual, video["video"])
        self.assertNotEqual(result["videos"][0]["video"]["file"]["sha256"], result["videos"][1]["video"]["file"]["sha256"])

    def test_second_video_failure_preserves_first_stats_and_bills(self):
        path, digest = self.prepare()
        request = perf.verify_request(path, digest)
        stack, build, calls, contexts, generator = self.fake_runtime(request, fail_seed=1)
        with stack:
            result = perf.execute_worker(request, time.monotonic(), digest)
        self.assertFalse(result["success"])
        self.assertEqual(result["videos"][0]["stats"]["observed"]["scheduler_steps"], 50)
        self.assertIsNotNone(result["videos"][0]["video"])
        self.assertGreater(result["elapsed_GPUh"], 0)
        self.assertTrue(result["parameters_unchanged"])
        self.assertTrue((path.parent / "worker.json").exists())
        build.assert_called_once()

    def test_mutation_or_reduced_frames_fail_without_fallback(self):
        for name, kwargs in (("mutation", dict(mutate=True)), ("short", dict(frames=61))):
            path, digest = self.prepare(name=name)
            request = perf.verify_request(path, digest)
            stack, build, calls, contexts, generator = self.fake_runtime(request, **kwargs)
            with stack:
                result = perf.execute_worker(request, time.monotonic(), digest)
            self.assertFalse(result["success"])
            build.assert_called_once()
            self.assertGreater(result["elapsed_GPUh"], 0)
            self.assertLessEqual(len(calls), 2)

    def test_direct_worker_cannot_bypass_live_authorized_parent(self):
        path, digest = self.prepare()
        request = perf.verify_request(path, digest)
        smoke.exclusive(path.parent / "launch.json", dict(gpu=2, supervisor_pid=12345, parent_pid=99999999, gpu_uuid="GPU-fixture", budget=perf.BUDGET))
        with mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES="GPU-fixture"), self.assertRaisesRegex(ValueError, "authorized measurement parent"):
            perf.worker_scope(request, 2, 12345, self.state, time.monotonic())


class ReceiptTests(Fixture):
    def test_actual_raw_receipt_roundtrip_and_tampered_receipt_sha(self):
        path, receipt, _ = self.successful_receipt()
        digest = smoke.sha(path.read_bytes())
        actual, request, sources, prior = perf.verify_measurement(path, digest, "alg", self.root)
        self.assertEqual(actual, receipt)
        self.assertEqual(prior, 200.)
        self.assertEqual(len(sources), 5)
        self.assertLess(prior + actual["elapsed_seconds"], 3600)
        with self.assertRaisesRegex(ValueError, "receipt SHA"):
            perf.verify_measurement(path, "f" * 64, "alg", self.root)

    def test_actual_performance_cap_cannot_be_exceeded(self):
        path, receipt, _ = self.successful_receipt()
        receipt.update(elapsed_seconds=2700., elapsed_GPUh=.75,
                       allocation_monotonic_end=receipt["allocation_monotonic_start"] + 2700.,
                       end=receipt["start"] + 2700.)
        path.write_bytes(smoke.encoded(receipt))
        with self.assertRaisesRegex(ValueError, "actual/cumulative"):
            perf.verify_measurement(path, smoke.sha(path.read_bytes()), "alg", self.root)

    def test_latest_smoke_and_method_links_cannot_be_replaced(self):
        path, receipt, _ = self.successful_receipt()
        receipt["latest_smoke02"]["sha256"] = "f" * 64
        path.write_bytes(smoke.encoded(receipt))
        with self.assertRaisesRegex(ValueError, "receipt/request binding"):
            perf.verify_measurement(path, smoke.sha(path.read_bytes()), "alg", self.root)
        with self.assertRaisesRegex(ValueError, "actual performance receipt"):
            perf.verify_measurement(path, smoke.sha(path.read_bytes()), "dymos", self.root)

    def test_manual_timing_summary_without_raw_worker_is_not_evidence(self):
        path, receipt, _ = self.successful_receipt()
        self.rewrite(path, lambda v: v.update(evidence={"load_seconds": 1, "inference_seconds": 1, "save_decode_seconds": 1}))
        with self.assertRaisesRegex(ValueError, "real launch/worker/weights"):
            perf.verify_measurement(path, smoke.sha(path.read_bytes()), "alg", self.root)

    def test_raw_timing_cannot_be_replaced_by_inconsistent_duration(self):
        path, receipt, _ = self.successful_receipt()
        worker_path = path.parent / "worker.json"
        worker = json.loads(worker_path.read_bytes())
        worker["timers"]["load"]["seconds"] += 10
        worker_path.write_bytes(smoke.encoded(worker))
        receipt["evidence"] = worker
        receipt["measurement_sources"][2] = smoke.record(worker_path)
        path.write_bytes(smoke.encoded(receipt))
        with self.assertRaisesRegex(ValueError, "nonrecomputable"):
            perf.verify_measurement(path, smoke.sha(path.read_bytes()), "alg", self.root)

    def test_wrong_video_sha_and_short_observed_steps_fail(self):
        for failure in ("sha", "steps"):
            with self.subTest(failure=failure):
                path, receipt, _ = self.successful_receipt(name=failure)
                if failure == "sha":
                    (path.parent / "seed1.mp4").write_bytes(b"changed CPU fixture")
                    with self.assertRaisesRegex(ValueError, "actual SHA"):
                        perf.verify_measurement(path, smoke.sha(path.read_bytes()), "alg", self.root)
                else:
                    worker_path = path.parent / "worker.json"
                    worker = json.loads(worker_path.read_bytes())
                    worker["videos"][1]["stats"]["observed"]["scheduler_steps"] = 49
                    worker_path.write_bytes(smoke.encoded(worker))
                    receipt["evidence"] = worker
                    receipt["measurement_sources"][2] = smoke.record(worker_path)
                    path.write_bytes(smoke.encoded(receipt))
                    with self.assertRaisesRegex(ValueError, "Incomplete 50-step"):
                        perf.verify_measurement(path, smoke.sha(path.read_bytes()), "alg", self.root)

    def test_launch_failure_is_billed_and_cleans_owned_pause(self):
        path, digest = self.prepare()
        with mock.patch.object(smoke, "supervisor_scope", return_value="GPU-fixture"), \
             mock.patch.object(perf.subprocess, "Popen", side_effect=OSError("CPU launch failure")):
            result = perf.run(path, digest, 2, 12345, self.state, True)
        self.assertFalse(result["success"])
        self.assertFalse(result["measured"])
        self.assertGreater(result["elapsed_GPUh"], 0)
        self.assertTrue(result["owned_tree_released"])
        self.assertFalse((self.state / "pause.2").exists())
        self.assertTrue((path.parent / "finished.json").exists())

    def test_timeout_retains_raw_progress_and_never_passes(self):
        path, digest = self.prepare()
        def launch(*args, **kwargs):
            smoke.checkpoint(path.parent / "progress.json", dict(success=False, mechanism_PASS=False, videos=[dict(seed=0, stats={"scheduler_steps": 3})]))
            return types.SimpleNamespace(pid=99999999, returncode=124, poll=lambda: 124, wait=lambda **kw: 124)
        with mock.patch.object(smoke, "supervisor_scope", return_value="GPU-fixture"), \
             mock.patch.object(perf.subprocess, "Popen", side_effect=launch), mock.patch.object(smoke, "stop_group"):
            result = perf.run(path, digest, 2, 12345, self.state, True)
        self.assertFalse(result["success"])
        self.assertTrue(result["timed_out"])
        self.assertEqual(result["evidence"]["videos"][0]["stats"]["scheduler_steps"], 3)
        self.assertGreater(result["elapsed_GPUh"], 0)

    def test_no_authorization_and_old_run_cannot_launch(self):
        path, digest = self.prepare()
        with mock.patch.object(smoke, "supervisor_scope", side_effect=AssertionError("scope reached")), self.assertRaisesRegex(ValueError, "gpu-authorized"):
            perf.run(path, digest, 2, 12345, self.state)
        smoke.exclusive(path.parent / "finished.json", {"old": True})
        with mock.patch.object(smoke, "supervisor_scope", return_value="GPU-fixture"), self.assertRaisesRegex(ValueError, "Old performance run"):
            perf.run(path, digest, 2, 12345, self.state, True)
        self.assertEqual(json.loads((path.parent / "finished.json").read_bytes()), {"old": True})


if __name__ == "__main__":
    unittest.main()
