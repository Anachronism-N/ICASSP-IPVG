import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location(
    "teacher_probe", Path(__file__).resolve().parents[1] / "scripts/run_teacher_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.out = self.root / "new_run"
        self.ckpt = self.root / "full1_a07.safetensors"
        self.ckpt.write_bytes(b"fixed-test-checkpoint")
        self.cache = self.root / "checkpoint_cache"
        self.cache.mkdir()
        self.cache_file = self.cache / (hashlib.sha256(str(self.ckpt).encode()).hexdigest() + ".safetensors")
        self.cache_file.write_bytes(self.ckpt.read_bytes())
        self.csv = self.root / "original.csv"
        self.csv.write_text("video,caption\noriginal.mp4,original caption\n")
        self.prompt = " ".join("visible-object-action" + str(i) for i in range(90))
        self.rows = []
        for case in range(6):
            image = self.root / (str(case) + ".png")
            image.write_bytes(("unique image " + str(case)).encode())
            review = {"csv_path": str(self.csv), "csv_sha256": probe.sha256(self.csv),
                      "prompt_sha256": hashlib.sha256(self.prompt.encode()).hexdigest(),
                      "reviewed_by": "test-agent", "reviewed_at": "2026-10-19T01:00:00+00:00",
                      "csv_row": 0, "video_column": "video", "prompt_column": "caption",
                      "original_video": "original.mp4",
                      "original_prompt_sha256": hashlib.sha256(b"original caption").hexdigest(),
                      "label_protocol": "agent_authored_conservative_dense_internal_dev_v1",
                      "reviewer_type": "agent", "use_scope": "training_visible_internal_diagnostic_only"}
            for model in ("base", "m_full1_a07"):
                for cfg in (3, 5):
                    identifier = str(case) + "_" + model + "_" + str(cfg)
                    self.rows.append({"id": identifier, "case_id": "case" + str(case),
                                      "canonical_source": "sequence" + str(case), "model": model,
                                      "ckpt": "NONE" if model == "base" else str(self.ckpt),
                                      "image": str(image), "prompt": self.prompt, "seed": 42 + case,
                                      "cfg": cfg, "shift": 5, "steps": 50,
                                      "out": "videos/" + identifier + ".mp4", "prompt_review": review})
        self.jobs = self.root / "jobs.jsonl"
        for patcher in (mock.patch.object(probe, "MERGE", self.ckpt),
                        mock.patch.object(probe, "MERGE_SHA256", probe.sha256(self.ckpt)),
                        mock.patch.object(probe, "CHECKPOINT_CACHE", self.cache)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.write_jobs()

    def write_jobs(self, rows=None):
        self.jobs.write_bytes(b"".join(probe.json_bytes(row) for row in (self.rows if rows is None else rows)))

    def normalized(self):
        return probe.validate_jobs(self.jobs, self.out)[0]

    def test_budget_defaults_and_rejection_before_files_or_gpu(self):
        self.assertEqual(probe.budget("0,1,2,3", 3600, 8, 24), (["0", "1", "2", "3"], 4, 4))
        for values in (("0,1,2,3", 3601, 8, 24), ("0,1", 3600, 1, 24),
                       ("0,0", 10, 8, 24), ("0", float("nan"), 8, 24),
                       ("0", 10, float("inf"), 24), ("0", 10, 8, 23)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                probe.budget(*values)
        with mock.patch.object(probe, "validate_jobs") as validate, mock.patch.object(probe, "assert_idle") as idle:
            with self.assertRaises(ValueError):
                probe.main(["--jobs", "missing.jsonl", "--out-dir", str(self.out), "--max-gpu-hours", "1"])
            validate.assert_not_called()
            idle.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_manifest_missing_arm_duplicate_changed_prompt_and_escape(self):
        bad_rows = []
        missing = copy.deepcopy(self.rows)
        missing[3]["model"], missing[3]["ckpt"] = "base", "NONE"
        bad_rows.append(missing)
        duplicate = copy.deepcopy(self.rows)
        duplicate[1]["id"] = duplicate[0]["id"]
        bad_rows.append(duplicate)
        changed = copy.deepcopy(self.rows)
        changed[1]["prompt"] += " extra"
        changed[1]["prompt_review"]["prompt_sha256"] = hashlib.sha256(changed[1]["prompt"].encode()).hexdigest()
        bad_rows.append(changed)
        for output in ("../escape.mp4", str(self.root / "escape.mp4"), "worker_0.json", "videos/shared.mp4"):
            escaped = copy.deepcopy(self.rows)
            escaped[0]["out"] = output
            if output == "videos/shared.mp4":
                escaped[1]["out"] = output
            bad_rows.append(escaped)
        for rows in bad_rows:
            with self.subTest(row=rows[0]), self.assertRaises(ValueError):
                self.write_jobs(rows)
                self.normalized()
        self.assertFalse(self.out.exists())

    def test_hash_review_source_and_sampler_fail_closed(self):
        for key, value in (("cfg", 4), ("cfg_zero_star", True), ("shift", 4), ("steps", 49),
                           ("num_frames", 120), ("quality", 6), ("seed", -1), ("model", "ft2"),
                           ("ckpt_sha256", "f" * 64), ("prompt", "too short")):
            rows = copy.deepcopy(self.rows)
            rows[3][key] = value
            self.write_jobs(rows)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.normalized()
        self.write_jobs()
        self.csv.write_text("changed CSV")
        with self.assertRaisesRegex(ValueError, "CSV SHA256"):
            self.normalized()
        self.csv.write_text("video,caption\noriginal.mp4,original caption\n")
        rows = copy.deepcopy(self.rows)
        for r in rows[4:8]:
            r["canonical_source"] = rows[0]["canonical_source"]
        self.write_jobs(rows)
        with self.assertRaisesRegex(ValueError, "independent"):
            self.normalized()

    def test_original_row_new_label_chains_and_agent_scope_are_distinct(self):
        for key, value in (("csv_row", 1), ("original_video", "wrong.mp4"),
                           ("original_prompt_sha256", "0" * 64), ("prompt_sha256", "0" * 64),
                           ("reviewer_type", "human"), ("use_scope", "unseen_generalization")):
            rows = copy.deepcopy(self.rows)
            for row in rows[:4]:
                row["prompt_review"][key] = value
            self.write_jobs(rows)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.normalized()
        self.write_jobs()
        jobs = self.normalized()
        self.assertNotEqual(jobs[0]["prompt_review"]["original_prompt_sha256"], jobs[0]["prompt_review"]["prompt_sha256"])

    def test_checkpoint_hash_once_and_no_heavy_import_in_parent(self):
        with mock.patch.object(probe, "sha256", wraps=probe.sha256) as digest:
            jobs = self.normalized()
        self.assertEqual(sum(call.args[0] == self.cache_file for call in digest.call_args_list), 1)
        self.assertEqual(sum(call.args[0] == self.ckpt for call in digest.call_args_list), 0)
        self.assertEqual(len(jobs), 24)
        self.assertTrue(all(j["num_frames"] == 121 and j["cfg_zero_star"] is False for j in jobs))
        self.assertNotIn("torch", probe.__dict__)
        self.assertEqual(set(j["cfg"] for j in jobs), {3, 5})

    def test_existing_directory_is_never_overwritten(self):
        self.out.mkdir()
        marker = self.out / "probe_summary.json"
        marker.write_text("existing user record")
        with mock.patch.object(probe, "assert_idle") as idle, self.assertRaises(ValueError):
            probe.main(["--jobs", str(self.jobs), "--out-dir", str(self.out)])
        self.assertEqual(marker.read_text(), "existing user record")
        idle.assert_not_called()

    def test_nan_atomic_write_does_not_replace_old_record(self):
        target = self.root / "record.json"
        target.write_text("old")
        with self.assertRaises(ValueError):
            probe.atomic_json(target, {"bad": float("nan")})
        self.assertEqual(target.read_text(), "old")

    def test_idle_gpu_exact_id_and_no_process_signalling(self):
        result = subprocess.CompletedProcess([], 0, "0, 0\n1, 99\n10, 0\n")
        original_read = Path.read_bytes
        proc_files = {"/proc/100/cmdline": b"python\0/project/gen_jobs.py\0",
                      "/proc/100/environ": b"CUDA_VISIBLE_DEVICES=10\0"}
        def read(path):
            return proc_files[str(path)] if str(path) in proc_files else original_read(path)
        with mock.patch.object(probe.subprocess, "run", return_value=result),                 mock.patch.object(probe.os, "listdir", return_value=["100"]),                 mock.patch.object(Path, "read_bytes", read):
            probe.assert_idle(["0", "1"])
            with self.assertRaisesRegex(ValueError, "PID 100"):
                probe.assert_idle(["10"])
        with mock.patch.object(probe.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "0, 100\n")):
            with self.assertRaisesRegex(ValueError, ">=100"):
                probe.assert_idle(["0"])

    def fake_worker_runtime(self, jobs, *, generation_error=False, decode_error=False):
        self.out.mkdir()
        assignment = self.out / "assignment_0.json"
        probe.atomic_json(assignment, {"gpu": "0", "jobs": jobs})
        cuda = types.SimpleNamespace(reset_peak_memory_stats=mock.Mock(), synchronize=mock.Mock(),
                                     max_memory_allocated=lambda: 12345, max_memory_reserved=lambda: 23456,
                                     empty_cache=mock.Mock())
        torch = types.ModuleType("torch")
        torch.cuda = cuda
        pipe = mock.Mock(return_value=["generated frames"])
        if generation_error:
            pipe.side_effect = RuntimeError("fake pipeline failure")
        gen = types.ModuleType("gen_jobs")
        gen.build_pipe = mock.Mock(return_value=pipe)
        gen.load_image = mock.Mock(return_value="loaded input")
        gen.NEG = "original negative prompt"
        data = types.ModuleType("diffsynth.utils.data")
        def save(video, path, fps, quality):
            self.assertEqual((video, fps, quality), (["generated frames"], 24, 5))
            Path(path).write_bytes(b"encoded fake MP4")
        data.save_video = save
        modules = {"torch": torch, "gen_jobs": gen, "diffsynth": types.ModuleType("diffsynth"),
                   "diffsynth.utils": types.ModuleType("diffsynth.utils"), "diffsynth.utils.data": data}
        with mock.patch.dict(sys.modules, modules),                 mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0", "TEACHER_PROBE_PARENT_PID": str(os.getppid())}),                 mock.patch.object(probe, "decode_video", return_value={"frames": 121, "width": 1280, "height": 704, "fps": 24, "bytes": 16},
                                  side_effect=ValueError("decode failed") if decode_error else None):
            rc = probe.worker(self.out, 0, probe.sha256(assignment))
        return rc, gen, pipe, json.loads((self.out / "worker_0.json").read_text())

    def test_worker_passes_original_quantitative_arguments_and_records_peak_time(self):
        jobs = [j for j in self.normalized() if j["model"] == "base"][:2]
        rc, gen, pipe, report = self.fake_worker_runtime(jobs)
        self.assertEqual(rc, 0)
        gen.build_pipe.assert_called_once_with("NONE")
        self.assertEqual(report["status"], "completed")
        self.assertGreaterEqual(report["worker_seconds"], sum(j["seconds"] for j in report["jobs"]))
        self.assertEqual(len(report["loads"]), 1)
        for call, job in zip(pipe.call_args_list, jobs):
            self.assertEqual(call.kwargs, {"prompt": job["prompt"], "negative_prompt": gen.NEG, "seed": job["seed"],
                                          "tiled": True, "height": 704, "width": 1280, "num_frames": 121,
                                          "input_image": "loaded input", "cfg_scale": job["cfg"], "sigma_shift": 5,
                                          "num_inference_steps": 50, "cfg_zero_star": False, "cfg_zero_steps": 0})
        for row in report["jobs"] + report["loads"]:
            self.assertGreaterEqual(row["seconds"], 0)
            self.assertEqual((row["peak_allocated_bytes"], row["peak_reserved_bytes"]), (12345, 23456))
        self.assertEqual(len(list((self.out / "videos").glob("*.mp4"))), 2)
        self.assertFalse(list((self.out / "videos").glob("*.tmp.mp4")))

    def test_worker_generation_failure_is_recorded_and_not_skipped(self):
        jobs = [self.normalized()[0]]
        rc, _, _, report = self.fake_worker_runtime(jobs, generation_error=True)
        self.assertEqual(rc, 1)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["jobs"][0]["status"], "failed")
        self.assertGreaterEqual(report["worker_seconds"], report["jobs"][0]["seconds"])
        self.assertFalse(Path(jobs[0]["out"]).exists())

    def test_failed_decode_preserves_independent_tmp_and_never_marks_mp4_complete(self):
        jobs = [self.normalized()[0]]
        rc, _, _, report = self.fake_worker_runtime(jobs, decode_error=True)
        self.assertEqual(rc, 1)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(Path(jobs[0]["out"]).exists())
        tmp = Path(report["jobs"][0]["temporary_output"])
        self.assertTrue(tmp.is_file())
        self.assertTrue(tmp.name.endswith(".tmp.mp4"))
        self.assertEqual(report["jobs"][0]["status"], "failed")

    def test_decode_reads_all_frames_and_rejects_wrong_count_dimension_fps(self):
        target = self.root / "video.mp4"
        target.write_bytes(b"not empty")
        imageio = types.ModuleType("imageio.v2")
        package = types.ModuleType("imageio")
        package.v2 = imageio
        class Reader:
            def __init__(self, count, shape=(704, 1280, 3), fps=24):
                self.count, self.shape, self.fps, self.closed = count, shape, fps, False
            def get_meta_data(self):
                return {"fps": self.fps}
            def __iter__(self):
                return iter([types.SimpleNamespace(shape=self.shape)] * self.count)
            def close(self):
                self.closed = True
        for count, shape, fps in ((121, (704, 1280, 3), 24), (120, (704, 1280, 3), 24),
                                  (121, (704, 1281, 3), 24), (121, (704, 1280, 3), 25)):
            reader = Reader(count, shape, fps)
            imageio.get_reader = lambda path: reader
            with mock.patch.dict(sys.modules, {"imageio": package, "imageio.v2": imageio}):
                if count == 121 and shape[1] == 1280 and fps == 24:
                    self.assertEqual(probe.decode_video(target)["frames"], 121)
                else:
                    with self.assertRaises(ValueError):
                        probe.decode_video(target)
            self.assertTrue(reader.closed)

    def test_real_mock_subprocess_failure_stops_only_owned_workers_and_bills_failure(self):
        self.out.mkdir()
        original_popen = subprocess.Popen
        started = []
        def spawn(command, **kwargs):
            index = int(command[command.index("--worker") + 1])
            code = "import sys; sys.exit(7)" if index == 1 else "import time; time.sleep(30)"
            proc = original_popen([sys.executable, "-c", code], cwd=str(self.root),
                                  env=kwargs["env"], stdin=kwargs["stdin"], stdout=kwargs["stdout"], stderr=kwargs["stderr"],
                                  start_new_session=kwargs["start_new_session"])
            started.append(proc)
            return proc
        with mock.patch.object(probe.subprocess, "Popen", side_effect=spawn):
            resources, error = probe.launch_workers(self.out, [{"gpu": "0", "jobs": []}, {"gpu": "1", "jobs": []}], 5)
        self.assertIn("failed", error)
        self.assertTrue(all(p.poll() is not None for p in started))
        self.assertEqual({r["pid"] for r in resources}, {p.pid for p in started})
        self.assertTrue(all(r["gpu_hours"] > 0 and r["seconds"] > 0 for r in resources))
        self.assertTrue(any(r["returncode"] == 7 for r in resources))

    def test_real_mock_timeout_terminates_kills_and_waits_exact_owned_pid(self):
        self.out.mkdir()
        original_popen = subprocess.Popen
        children = []
        def spawn(command, **kwargs):
            child = original_popen([sys.executable, "-c",
                                    "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"],
                                   cwd=str(self.root), env=kwargs["env"], stdout=kwargs["stdout"], stderr=kwargs["stderr"],
                                   start_new_session=kwargs["start_new_session"])
            children.append(child)
            return child
        t0 = time.perf_counter()
        with mock.patch.object(probe.subprocess, "Popen", side_effect=spawn):
            resources, error = probe.launch_workers(self.out, [{"gpu": "0", "jobs": []}], 0.4)
        self.assertIn("TimeoutError", error)
        self.assertLess(time.perf_counter() - t0, 2)
        self.assertEqual(children[0].returncode, -9)
        self.assertEqual(resources[0]["pid"], children[0].pid)
        self.assertGreater(resources[0]["gpu_hours"], 0)
        self.assertLessEqual(resources[0]["seconds"], 0.4)

    def test_framework_snapshot_hashes_resources_and_ignores_nonruntime_files(self):
        source = self.root / "DiffSynth-Studio" / "diffsynth"
        source.mkdir(parents=True)
        (source / "__init__.py").write_text("SNAPSHOT_MARKER = 'local frozen framework'\n")
        (source / "runtime.py").write_text("VALUE = 7\n")
        (source / "runtime.json").write_text('{"resource": "kept"}')
        (source / "unused.pyc").write_bytes(b"bytecode ignored")
        definitions = source / "models"
        definitions.mkdir()
        (definitions / "wan_video_dit.py").write_text("RUNTIME_MODEL_CODE = True\n")
        for name in ("data", "models", ".git"):
            excluded = self.root / "DiffSynth-Studio" / name
            excluded.mkdir()
            (excluded / "not_package.py").write_text("not runtime package")
        for name in ("__pycache__", ".git"):
            directory = source / name
            directory.mkdir()
            (directory / "ignored.py").write_text("ignored")
        self.out.mkdir()
        with mock.patch.object(probe, "ROOT", self.root):
            hashes = probe.freeze_framework(self.out)
        self.assertEqual(set(hashes), {"diffsynth/__init__.py", "diffsynth/runtime.py", "diffsynth/runtime.json",
                                     "diffsynth/models/wan_video_dit.py"})
        for relative, digest in hashes.items():
            self.assertEqual(probe.sha256(self.out / "framework" / relative), digest)
        (source / "runtime.py").write_text("changed shared source")
        self.assertEqual((self.out / "framework/diffsynth/runtime.py").read_text(), "VALUE = 7\n")

    def test_worker_resolves_frozen_framework_and_pipeline_copy(self):
        self.out.mkdir()
        code = self.out / "code"
        code.mkdir()
        (code / "run_teacher_probe.py").write_text("# frozen entrypoint\n")
        (code / "gen_jobs.py").write_text("MARKER = 'frozen gen_jobs'\n")
        package = self.out / "framework" / "diffsynth"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("MARKER = 'frozen framework'\n")
        original_popen = subprocess.Popen
        launched = []
        def spawn(command, **kwargs):
            self.assertEqual(command[1], str(code / "run_teacher_probe.py"))
            self.assertEqual(kwargs["cwd"], str(self.out / "framework"))
            self.assertEqual(kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0], str(self.out / "framework"))
            self.assertEqual(kwargs["env"]["TEACHER_PROBE_ROOT"], str(probe.ROOT))
            self.assertEqual(kwargs["env"]["WM_CHECKPOINT_CACHE_DIR"], str(self.cache))
            script = ("import sys; sys.path.insert(0, " + repr(str(code)) + "); "
                      "import diffsynth,gen_jobs; "
                      "assert diffsynth.MARKER == 'frozen framework'; "
                      "assert gen_jobs.MARKER == 'frozen gen_jobs'; "
                      "assert diffsynth.__file__.startswith(" + repr(str(package)) + "); "
                      "assert gen_jobs.__file__.startswith(" + repr(str(code)) + ")")
            proc = original_popen([sys.executable, "-c", script], **kwargs)
            launched.append(proc)
            return proc
        with mock.patch.object(probe.subprocess, "Popen", side_effect=spawn):
            resources, error = probe.launch_workers(self.out, [{"gpu": "0", "jobs": []}], 5)
        self.assertIsNone(error)
        self.assertEqual(resources[0]["returncode"], 0)
        self.assertIsNotNone(launched[0].poll())

    def test_parent_failure_freezes_manifest_csv_and_writes_failed_summary(self):
        scripts = self.root / "scripts"
        scripts.mkdir()
        for name in ("run_teacher_probe.py", "gen_jobs.py"):
            (scripts / name).write_text("# frozen test entrypoint\n")
        package = self.root / "DiffSynth-Studio" / "diffsynth"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("# frozen test runtime\n")
        with mock.patch.object(probe, "ROOT", self.root), mock.patch.object(probe, "assert_idle"),                 mock.patch.object(probe, "launch_workers", return_value=([{"index": 0, "gpu": "0", "pid": 77,
                                                                          "returncode": 2, "seconds": 2, "gpu_hours": 2 / 3600}], "worker failure")),                 mock.patch.object(probe, "decode_video") as decode:
            rc = probe.main(["--jobs", str(self.jobs), "--out-dir", str(self.out), "--gpus", "0"])
        self.assertEqual(rc, 1)
        summary = json.loads((self.out / "probe_summary.json").read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["gpu_hours"], 2 / 3600)
        self.assertEqual((self.out / "jobs.input.jsonl").read_bytes(), self.jobs.read_bytes())
        self.assertEqual(summary["manifest_sha256"], probe.sha256(self.out / "jobs.frozen.jsonl"))
        self.assertEqual((self.out / "original_csv" / (probe.sha256(self.csv) + ".csv")).read_bytes(), self.csv.read_bytes())
        decode.assert_not_called()
        manifest = json.loads((self.out / "probe_manifest.json").read_text())
        self.assertEqual(manifest["framework_root"], str(self.out / "framework"))
        self.assertIn("diffsynth/__init__.py", manifest["framework_source_sha256"])
        for name, digest in manifest["source_sha256"].items():
            self.assertEqual(probe.sha256(self.out / "code" / name), digest)
        for relative, digest in manifest["framework_source_sha256"].items():
            self.assertEqual(probe.sha256(self.out / "framework" / relative), digest)

    def test_completion_requires_all_24_decodes_and_zero_exits_and_charge_within_cap(self):
        jobs = self.normalized()
        self.out.mkdir()
        for job in jobs:
            target = Path(job["out"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"mock video")
        rows = [{"id": j["id"], "out": j["out"], "status": "completed", "seconds": 1,
                 "peak_allocated_bytes": 1, "peak_reserved_bytes": 2} for j in jobs]
        probe.atomic_json(self.out / "worker_0.json", {"status": "completed", "gpu": "0", "jobs": rows})
        resource = {"index": 0, "gpu": "0", "pid": 99, "returncode": 0, "seconds": 24, "gpu_hours": 24 / 3600}
        with mock.patch.object(probe, "decode_video", return_value={"frames": 121}) as decode:
            self.assertEqual(probe.finish(self.out, jobs, [resource], None, 1, 4, "hash"), 0)
            self.assertEqual(decode.call_count, 24)
        self.assertEqual(json.loads((self.out / "probe_summary.json").read_text())["status"], "completed")
        resource["gpu_hours"] = 4.1
        with mock.patch.object(probe, "decode_video") as decode:
            self.assertEqual(probe.finish(self.out, jobs, [resource], None, 4, 4, "hash"), 1)
            decode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
