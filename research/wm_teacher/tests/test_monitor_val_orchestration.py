import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load_script("score_merge_cpu")
monitor = load_script("monitor_merge")
val = load_script("val_merge_score")


def loss_fixture():
    clips = []
    for i in range(120):
        dataset = "alpha" if i < 30 else "beta"
        losses = [(i + 1) / 1000 + k / 10 for k in range(5)]
        clips.append({"video": f"dataset/wan22_5s/{dataset}_5s/{i:03}.mp4",
                      "dataset": dataset, "loss_t": losses, "loss": sum(losses) / 5})
    return {"ckpt": "none", "n": 120, "num_frames": 121,
            "t_fracs": list(val.T_FRACS), "t_ids": [100, 300, 500, 700, 900],
            "mean_loss": sum(c["loss"] for c in clips) / 120,
            "mean_loss_per_t": [sum(c["loss_t"][k] for c in clips) / 120 for k in range(5)],
            "per_dataset": {d: sum(c["loss"] for c in clips if c["dataset"] == d) / n
                            for d, n in (("alpha", 30), ("beta", 90))}, "clips": clips}


class TemporaryRoot(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write_json(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return path


class MonitorTests(TemporaryRoot):
    def make_jobs(self, count=4, model="m_full1_a07", start=0):
        jobs = []
        for i in range(start, start + count):
            path = self.root / "eval/wmb/videos" / model / f"{i}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not-real-video")
            os.utime(path, ns=(i + 1, i + 1))
            jobs.append({"model": model, "id": str(i), "out": str(path)})
        (self.root / "eval/jobs_merge_all.jsonl").write_text(
            "".join(json.dumps(j) + "\n" for j in jobs))
        return jobs

    def state(self):
        return {"tag": "test_generation", "nodes": [{"rank": 0, "ip": "launcher.example"},
                {"rank": 1, "ip": "worker.example", "user": "tester", "port": 2211}]}

    def decode_report(self, jobs):
        model = jobs[0]["model"]
        return self.write_json(self.root / "eval/diag" / f"decode_{model}_20261018.json",
            {"model": model, "count": 1011, "expected": 1011, "all_frames_decoded": True,
             "frames": 121, "fps": 24, "width": 1280, "height": 704,
             "outputs": [j["out"] for j in jobs]})

    def test_logs_partial_cr_lf_errors_and_truncate_are_cumulative(self):
        path = self.root / "gen_test_s0.log"
        cursors, errors = {}, []
        path.write_bytes(b"sampling 2/50 [abc]\r[model] FA")
        self.assertEqual(monitor.read_logs([path], cursors, errors), 1)
        self.assertEqual(errors, [])
        with path.open("ab") as handle:
            handle.write(b"IL clip: OutOfMemoryError")
        monitor.read_logs([path], cursors, errors)
        self.assertEqual(len(errors), 1)
        monitor.read_logs([path], cursors, errors)
        self.assertEqual(len(errors), 1)
        with path.open("ab") as handle:
            handle.write(b" more detail\r\nTraceback (most recent call last)\n[shard fai")
        monitor.read_logs([path], cursors, errors)
        self.assertEqual(len(errors), 2)
        with path.open("ab") as handle:
            handle.write(b"led] 2 generation errors\r")
        monitor.read_logs([path], cursors, errors)
        self.assertEqual(len(errors), 3)
        path.write_bytes(b"OutOfMemoryError\n")
        monitor.read_logs([path], cursors, errors)
        self.assertEqual(len(errors), 4)
        self.assertEqual(cursors[path.name]["offset"], path.stat().st_size)
        monitor.read_logs([path], cursors, errors)
        self.assertEqual(len(errors), 4)
        other = self.root / "rotated"
        other.write_bytes(b"[shard failed] rotated log\n")
        other.replace(path)
        monitor.read_logs([path], cursors, errors)
        self.assertEqual(len(errors), 5)
        self.assertTrue(all(row["log"] == path.name for row in errors))

    def test_truncated_log_regrown_past_old_offset_keeps_errors_and_resets_partial(self):
        path = self.root / "gen.log"
        path.write_bytes(b"[model] FAIL old\npartial")
        cursors, errors = {}, []
        monitor.read_logs([path], cursors, errors)
        old_offset = cursors[path.name]["offset"]
        path.write_bytes(b"[shard failed] new error\n" + b"z" * 300)
        self.assertGreater(path.stat().st_size, old_offset)
        monitor.read_logs([path], cursors, errors)
        self.assertEqual(len(errors), 2)
        self.assertTrue(errors[-1]["line"].startswith("[shard failed]"))
        monitor.read_logs([path], cursors, errors)
        self.assertEqual(len(errors), 2)
        self.assertNotIn("partial", cursors[path.name]["partial"])

    def test_logs_read_all_chunks_and_resume_serialized_cursor(self):
        path = self.root / "large.log"
        path.write_bytes(b"x" * (65536 - 5) + b"[shard failed] 1\n" + b"x" * 70000)
        cursors, errors = {}, []
        monitor.read_logs([path], cursors, errors)
        self.assertEqual(len(errors), 1)
        cursors = json.loads(json.dumps(cursors))
        with path.open("ab") as handle:
            handle.write(b"\n[model] FAIL clip\n")
        monitor.read_logs([path], cursors, errors)
        self.assertEqual(len(errors), 2)
        self.assertEqual(cursors[path.name]["offset"], path.stat().st_size)

    def test_gpu_uses_state_hosts_and_reports_nonzero_timeout_and_oserror(self):
        run = mock.Mock(return_value=subprocess.CompletedProcess([], 255, "", "unreachable"))
        node = self.state()["nodes"][1]
        report = monitor.node_gpu(node, run=run)
        args = run.call_args.args[0]
        self.assertIn("tester@worker.example", args)
        self.assertIn("2211", args)
        self.assertEqual(report["rc"], 255)
        self.assertEqual(report["error"], "unreachable")
        self.assertFalse(any("password" in word.lower() for word in args))
        for failure in (subprocess.TimeoutExpired("nvidia-smi", 20), FileNotFoundError("missing nvidia-smi")):
            with self.subTest(failure=type(failure).__name__):
                report = monitor.node_gpu(node, run=mock.Mock(side_effect=failure))
                self.assertTrue(report["error"])
        run.reset_mock()
        monitor.node_gpu(self.state()["nodes"][0], run=run)
        self.assertEqual(run.call_args.args[0][0], "nvidia-smi")
        with self.assertRaises(RuntimeError):
            monitor.node_gpu({"rank": 1, "ip": "-bad"}, run=run)

    def test_snapshot_newest_partial_decode_and_resource_errors_are_diagnostic(self):
        jobs = self.make_jobs()
        logs = self.root / "dataset/download_logs"
        logs.mkdir(parents=True)
        (logs / "gen_test_generation_s0.log").write_text("[shard failed] 1 error\n")
        (logs / "gen_other_s0.log").write_text("OutOfMemoryError\n")
        Path(jobs[0]["out"]).write_bytes(b"")
        Path(jobs[1]["out"]).unlink()
        known, errors, cursors = set(), [], {}
        decode = mock.Mock(side_effect=[RuntimeError("corrupt"), None])
        query = mock.Mock(side_effect=lambda node, rank: {"rank": node["rank"], "error": "unreachable"})
        with mock.patch.object(monitor.time, "strftime", side_effect=["started", "finished"]):
            report = monitor.snapshot(self.root, self.state(), known, cursors, errors,
                                      sample_limit=2, decode=decode, query=query)
        self.assertEqual([c.args[0] for c in decode.call_args_list],
                         [Path(jobs[3]["out"]), Path(jobs[2]["out"])])
        self.assertEqual(known, {jobs[2]["out"]})
        self.assertEqual(report["complete_nonempty"], 2)
        self.assertEqual(report["total"], 4)
        self.assertEqual(report["fully_decoded_models"], [])
        self.assertEqual(report["snapshot_started_at"], "started")
        self.assertEqual(report["snapshot_finished_at"], "finished")
        self.assertEqual(report["generation_logs"], 1)
        self.assertEqual(len(report["errors"]), 1)
        self.assertTrue(all(n["error"] for n in report["nodes"]))
        self.assertCountEqual([c.args[0] for c in query.call_args_list], self.state()["nodes"])
        self.assertEqual([n["rank"] for n in report["nodes"]], [n["rank"] for n in self.state()["nodes"]])
        self.assertIn("Nonempty count/sample decoding only", report["caveat"])
        self.assertNotIn("accepted", report)

    def test_valid_all1011_decode_skips_samples(self):
        jobs = self.make_jobs(1011)
        self.decode_report(jobs)
        decode = mock.Mock(side_effect=AssertionError("must skip complete decode"))
        report = monitor.snapshot(self.root, self.state(), set(), {}, [], decode=decode,
                                  query=lambda node, rank: {"rank": node["rank"]})
        decode.assert_not_called()
        self.assertEqual(report["fully_decoded_models"], [jobs[0]["model"]])
        self.assertEqual(report["sample_decodes"], [])

    def test_full_model_is_skipped_while_other_model_uses_latest_unseen_output(self):
        complete = self.make_jobs(1011, model="m_full1_a03")
        self.decode_report(complete)
        pending = self.make_jobs(4, model="m_full1_a07", start=2000)
        (self.root / "eval/jobs_merge_all.jsonl").write_text(
            "".join(json.dumps(j) + "\n" for j in complete + pending))
        known = {pending[-1]["out"]}
        decode = mock.Mock()
        report = monitor.snapshot(self.root, self.state(), known, {}, [], sample_limit=1,
                                  decode=decode, query=lambda node, rank: {})
        decode.assert_called_once_with(Path(pending[-2]["out"]))
        self.assertEqual(report["fully_decoded_models"], ["m_full1_a03"])
        self.assertEqual(report["checked_new_outputs"], 2)

    def test_duplicate1011_job_paths_are_not_complete_decode_evidence(self):
        jobs = self.make_jobs(1) * 1011
        (self.root / "eval/jobs_merge_all.jsonl").write_text(
            "".join(json.dumps(j) + "\n" for j in jobs))
        self.decode_report(jobs)
        report = monitor.snapshot(self.root, self.state(), set(), {}, [], sample_limit=0,
                                  decode=mock.Mock(), query=lambda node, rank: {})
        self.assertEqual(report["fully_decoded_models"], [])
        self.assertIn("Nonempty count/sample decoding only", report["caveat"])

    def test_counts_alone_partial_or_wrong_decode_report_never_marks_full(self):
        jobs = self.make_jobs(1011)
        report_path = self.decode_report(jobs)
        original = json.loads(report_path.read_text())
        variants = [None, dict(original, outputs=original["outputs"][::-1]),
                    dict(original, all_frames_decoded=False), dict(original, count=1010)]
        for report in variants:
            with self.subTest(report=None if report is None else report["count"]):
                report_path.write_text("{" if report is None else json.dumps(report))
                decode = mock.Mock()
                result = monitor.snapshot(self.root, self.state(), set(), {}, [], sample_limit=1,
                                          decode=decode, query=lambda node, rank: {})
                self.assertEqual(result["fully_decoded_models"], [])
                decode.assert_called_once_with(Path(jobs[-1]["out"]))
        short = jobs[:4]
        (self.root / "eval/jobs_merge_all.jsonl").write_text(
            "".join(json.dumps(j) + "\n" for j in short))
        self.decode_report(short)
        result = monitor.snapshot(self.root, self.state(), set(), {}, [], decode=mock.Mock(),
                                  query=lambda node, rank: {})
        self.assertEqual(result["complete_nonempty"], 4)
        self.assertEqual(result["fully_decoded_models"], [])

    def test_main_reads_state_and_resumes_matching_tag_without_external_operations(self):
        state = self.state()
        self.write_json(self.root / "eval/diag/custom_state.json", state)
        latest = self.root / "eval/diag/merge_monitor_latest.json"
        saved = {"tag": state["tag"], "sample_checked_outputs": ["known.mp4"],
                 "log_cursors": {"old.log": {"offset": 3}}, "errors": [{"line": "old failure"}]}
        for same_tag in (True, False):
            with self.subTest(same_tag=same_tag):
                self.write_json(latest, dict(saved, tag=state["tag"] if same_tag else "other"))
                report = {"at": "finished", "complete_nonempty": 0, "total": 0, "by_model": {}, "errors": []}
                with mock.patch.object(sys, "argv", ["monitor_merge.py", "--root", str(self.root),
                                       "--state", "eval/diag/custom_state.json", "--once"]), \
                     mock.patch.object(monitor, "snapshot", return_value=report) as snapshot, \
                     mock.patch.object(monitor, "write_snapshot") as write, \
                     mock.patch.object(monitor.time, "sleep", side_effect=AssertionError("once must not sleep")):
                    monitor.main()
                arguments = snapshot.call_args.args
                self.assertEqual(arguments[:2], (self.root, state))
                self.assertEqual(arguments[2], {"known.mp4"} if same_tag else set())
                self.assertEqual(arguments[3], saved["log_cursors"] if same_tag else {})
                self.assertEqual(arguments[4], saved["errors"] if same_tag else [])
                write.assert_called_once_with(self.root, report)

    def test_snapshot_publication_records_interval_and_accumulated_state(self):
        self.make_jobs(1)
        report = monitor.snapshot(self.root, self.state(), set(), {}, [], sample_limit=0,
                                  decode=mock.Mock(), query=lambda node, rank: {})
        monitor.write_snapshot(self.root, report)
        monitor.write_snapshot(self.root, report)
        path = self.root / "eval/diag/merge_monitor_latest.json"
        self.assertEqual(json.loads(path.read_text()), report)
        events = (path.parent / "merge_monitor_events.jsonl").read_text().splitlines()
        self.assertEqual(len(events), 2)
        self.assertEqual(json.loads(events[0])["snapshot_started_at"], report["snapshot_started_at"])


class ValLossTests(TemporaryRoot):
    def setUp(self):
        super().setUp()
        self.base = loss_fixture()
        self.base_path = self.write_json(self.root / val.LOSS_DIR / "loss_base.json", self.base)
        self.destination = self.base_path.parent / "loss_m_full1_a03.json"
        self.csv_path = self.root / "training_metadata/round2/round2_val.csv"
        self.csv_path.parent.mkdir(parents=True)
        with self.csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["video", "prompt"])
            writer.writeheader()
            for clip in self.base["clips"]:
                writer.writerow({"video": clip["video"], "prompt": "unchanged"})
        (self.root / "dataset/download_logs").mkdir(parents=True)
        (self.root / "DiffSynth-Studio").mkdir()

    def assert_rejected(self, result, pattern=None):
        self.write_json(self.destination, result)
        before = self.destination.read_bytes()
        with self.assertRaises(RuntimeError) if pattern is None else self.assertRaisesRegex(RuntimeError, pattern):
            val.validate_loss(self.destination, self.base)
        self.assertEqual(self.destination.read_bytes(), before)

    def test_original_base_is_read_only_and_preserves_csv_sampling_order(self):
        before = self.base_path.read_bytes()
        self.assertEqual(val.fixed_base(self.root), self.base)
        self.assertEqual(self.base_path.read_bytes(), before)
        with self.csv_path.open() as handle:
            rows = list(csv.DictReader(handle))
        with self.csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["video", "prompt"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
                writer.writerow({"video": "unsampled.mp4", "prompt": "ignored"})
        self.assertEqual(val.fixed_base(self.root), self.base)
        with self.csv_path.open("a") as handle:
            handle.write("extra.mp4,prompt\n")
        self.assertEqual(val.fixed_base(self.root), self.base)
        self.csv_path.write_text("video,prompt\nwrong.mp4,prompt\n")
        with self.assertRaisesRegex(RuntimeError, "CSV no longer matches"):
            val.fixed_base(self.root)
        self.assertEqual(self.base_path.read_bytes(), before)

    def test_missing_original_base_fields_report_and_never_guess(self):
        for key in ("mean_loss_per_t", "per_dataset", "t_ids", "clips"):
            with self.subTest(key=key):
                result = copy.deepcopy(self.base)
                result.pop(key)
                self.write_json(self.base_path, result)
                before = self.base_path.read_bytes()
                with self.assertRaisesRegex(RuntimeError, "Missing/invalid.*loss"):
                    val.fixed_base(self.root)
                self.assertEqual(self.base_path.read_bytes(), before)

    def test_exact120_unique_video_order_and121_frames(self):
        for change in (lambda r: r.update(n=119), lambda r: r.update(num_frames=120),
                       lambda r: r["clips"].pop(), lambda r: r["clips"].reverse(),
                       lambda r: r["clips"][0].update(video=r["clips"][1]["video"])):
            result = copy.deepcopy(self.base)
            change(result)
            self.assert_rejected(result)

    def test_fixed_timestep_indices_and_fracs_never_change(self):
        for change in (lambda r: r["t_fracs"].reverse(), lambda r: r["t_ids"].reverse(),
                       lambda r: r["t_ids"].pop(), lambda r: r["t_ids"].__setitem__(0, 100.0)):
            result = copy.deepcopy(self.base)
            change(result)
            self.assert_rejected(result)

    def test_clip_and_overall_mean_consistency_is_required(self):
        for change in (lambda r: r["clips"][0].update(loss=8.0),
                       lambda r: r["clips"][0]["loss_t"].pop(),
                       lambda r: r.update(mean_loss=r["mean_loss"] + 0.01)):
            result = copy.deepcopy(self.base)
            change(result)
            self.assert_rejected(result)

    def test_each_per_t_aggregate_must_match_actual_clips(self):
        for k in range(5):
            result = copy.deepcopy(self.base)
            result["mean_loss_per_t"][k] += 0.001
            with self.subTest(t=k):
                self.assert_rejected(result, "per-t")
        result = copy.deepcopy(self.base)
        result["mean_loss_per_t"].reverse()
        self.assert_rejected(result, "per-t")

    def test_dataset_exact_key_set_membership_and_weighted_means(self):
        for change in (lambda r: r["per_dataset"].update(alpha=r["per_dataset"]["alpha"] + 0.01),
                       lambda r: r["per_dataset"].pop("beta"),
                       lambda r: r["per_dataset"].update(unexpected=0.1),
                       lambda r: r["clips"][0].update(dataset="beta"),
                       lambda r: r["clips"][0].update(dataset="unexpected")):
            result = copy.deepcopy(self.base)
            change(result)
            self.assert_rejected(result, "dataset")
        self.assertNotAlmostEqual(self.base["mean_loss"],
                                  sum(self.base["per_dataset"].values()) / 2)
        self.write_json(self.destination, self.base)
        self.assertEqual(val.validate_loss(self.destination, self.base), self.base)

    def test_all_loss_levels_reject_nonfinite_nonnumeric_bool(self):
        for invalid in (float("nan"), float("inf"), -float("inf"), True, "0.1", None):
            for field in ("clip", "t", "mean", "per_t", "dataset"):
                with self.subTest(invalid=invalid, field=field):
                    result = copy.deepcopy(self.base)
                    if field == "clip":
                        result["clips"][0]["loss"] = invalid
                    elif field == "t":
                        result["clips"][0]["loss_t"][0] = invalid
                    elif field == "mean":
                        result["mean_loss"] = invalid
                    elif field == "per_t":
                        result["mean_loss_per_t"][0] = invalid
                    else:
                        result["per_dataset"]["alpha"] = invalid
                    self.assert_rejected(result)

    def test_python_script_process_detection_does_not_match_shell_or_bodies(self):
        yes = [["python", "gen_jobs.py", "jobs", "0", "32"],
               ["/opt/bin/python3.11", "-u", "/tmp/gen_jobs.py", "jobs"],
               [b"python", b"-B", b"gen_jobs.py", b""]]
        no = [[], ["sh", "-c", "python gen_jobs.py jobs"],
              ["python", "-c", "exec(open('gen_jobs.py').read())"],
              ["python", "-m", "gen_jobs.py"], ["python", "other.py", "gen_jobs.py"],
              ["python", "-u", "prefix_gen_jobs.py"],
              ["python", "-W", "gen_jobs.py", "other.py"],
              ["python", "-X", "gen_jobs.py", "other.py"],
              ["python", "-uc", "print(1)", "gen_jobs.py"],
              ["python-helper", "gen_jobs.py"],
              ["python", "--help", "gen_jobs.py"],
              ["python", "-", "gen_jobs.py"]]
        for args in yes:
            with self.subTest(args=args):
                self.assertTrue(val.is_generator(args))
        for args in no:
            with self.subTest(args=args):
                self.assertFalse(val.is_generator(args))
        proc = self.root / "proc"
        proc.mkdir()
        for i, args in enumerate(no[1:]):
            folder = proc / str(i)
            folder.mkdir()
            (folder / "cmdline").write_bytes(b"\0".join(a.encode() for a in args) + b"\0")
        (proc / "self").mkdir()
        (proc / "999").mkdir()
        self.assertFalse(val.generators_active(proc))
        (proc / "999/cmdline").write_bytes(b"python\0-u\0/tmp/gen_jobs.py\0jobs\0")
        self.assertTrue(val.generators_active(proc))

    def test_gpu_gate_requires_all_external_zero_one_two_released(self):
        for stdout, expected in (("0, 0\n1, 90\n2, 99\n3, 79900\n", True),
                                 ("0, 0\n1, 0\n2, 100\n", False),
                                 ("0, 0\n1, 0\n", False),
                                 ("4, 0\n5, 0\n6, 0\n", False)):
            with self.subTest(stdout=stdout):
                run = mock.Mock(return_value=subprocess.CompletedProcess([], 0, stdout, ""))
                self.assertEqual(val.gpu_idle(run=run), expected)
                self.assertTrue(run.call_args.kwargs["check"])
        active = mock.Mock(side_effect=[True, False, False])
        idle = mock.Mock(side_effect=[False, True])
        sleep = mock.Mock()
        val.wait_for_release(17, active=active, idle=idle, sleep=sleep)
        self.assertEqual(active.call_count, 3)
        self.assertEqual(idle.call_count, 2)
        self.assertEqual(sleep.call_args_list, [mock.call(17), mock.call(17)])

    def test_existing_valid_loss_skips_and_invalid_loss_is_preserved(self):
        self.write_json(self.destination, self.base)
        before = self.destination.read_bytes()
        run = mock.Mock(side_effect=AssertionError("must not score"))
        val.score_model(self.root, "m_full1_a03", 0, self.base, "mock-python", run=run)
        run.assert_not_called()
        self.assertEqual(self.destination.read_bytes(), before)
        result = copy.deepcopy(self.base)
        result["mean_loss_per_t"][0] += 0.1
        self.write_json(self.destination, result)
        before = self.destination.read_bytes()
        with self.assertRaises(RuntimeError):
            val.score_model(self.root, "m_full1_a03", 0, self.base, "mock-python", run=run)
        run.assert_not_called()
        self.assertEqual(self.destination.read_bytes(), before)

    def test_score_uses_external_gpu_and_unchanged_protocol_then_publishes(self):
        before = self.base_path.read_bytes()
        def runner(args, **kwargs):
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2")
            self.assertTrue(kwargs["check"])
            self.assertEqual(args[args.index("--n") + 1], "120")
            self.assertEqual(args[args.index("--num-frames") + 1], "121")
            self.assertEqual(args[args.index("--val-csv") + 1], str(self.csv_path))
            self.assertTrue(args[args.index("--ckpt") + 1].endswith("full1_a03.safetensors"))
            self.assertEqual(kwargs["cwd"], self.root / "DiffSynth-Studio")
            self.write_json(Path(args[args.index("--out") + 1]), self.base)
        run = mock.Mock(side_effect=runner)
        val.score_model(self.root, "m_full1_a03", 2, self.base, "mock-python", run=run)
        run.assert_called_once()
        self.assertEqual(json.loads(self.destination.read_text()), self.base)
        self.assertEqual(self.base_path.read_bytes(), before)

    def test_subprocess_failure_and_invalid_output_never_publish(self):
        for model in val.MODELS:
            destination = self.base_path.parent / ("loss_" + model + ".json")
            def runner(args, **kwargs):
                output = Path(args[args.index("--out") + 1])
                result = copy.deepcopy(self.base)
                if model == val.MODELS[1]:
                    result["per_dataset"]["beta"] += 0.1
                self.write_json(output, result)
                if model != val.MODELS[1]:
                    raise subprocess.CalledProcessError(7, args)
            with self.subTest(model=model):
                with self.assertRaises((RuntimeError, subprocess.CalledProcessError)):
                    val.score_model(self.root, model, val.MODELS.index(model), self.base,
                                    "mock-python", run=runner)
                self.assertFalse(destination.exists())
                work = list(destination.parent.glob("." + model + ".loss-*"))
                self.assertEqual(len(work), 1)
                self.assertTrue((work[0] / destination.name).is_file())

    def test_main_waits_before_three_external_gpu_launches(self):
        events = []
        def wait(interval):
            events.append("released")
        def score(root, model, gpu, base, python):
            self.assertEqual(events[0], "released")
            self.assertEqual(gpu, val.MODELS.index(model))
            events.append((model, gpu))
        with mock.patch.object(sys, "argv", ["val_merge_score.py", "--root", str(self.root)]), \
             mock.patch.object(val, "wait_for_release", side_effect=wait), \
             mock.patch.object(val, "score_model", side_effect=score) as scorer:
            val.main()
        self.assertEqual(scorer.call_count, 3)
        self.assertEqual({gpu for _, gpu in events[1:]}, {0, 1, 2})

    def test_help_has_no_io_gpu_scoring_or_loop_side_effects(self):
        for module in (monitor, val):
            with self.subTest(module=module.__name__), \
                 mock.patch.object(sys, "argv", [module.__name__, "--help"]), \
                 mock.patch.object(module, "read_json", side_effect=AssertionError("no IO")), \
                 mock.patch.object(module.subprocess, "run", side_effect=AssertionError("no process")), \
                 mock.patch.object(module.time, "sleep", side_effect=AssertionError("no loop")):
                with self.assertRaises(SystemExit) as caught:
                    module.main()
                self.assertEqual(caught.exception.code, 0)
                self.assertEqual(module.ROOT, ROOT)


if __name__ == "__main__":
    unittest.main()
