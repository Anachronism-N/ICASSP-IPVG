import csv
import importlib.util
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location(
    "score_merge_cpu", Path(__file__).resolve().parents[1] / "scripts/score_merge_cpu.py")
cpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpu)


MODEL = "m_full1_a03"


def lock_waiter(root, suite, entered, release):
    with cpu.suite_lock(Path(root), MODEL, suite):
        entered.set()
        release.wait(5)


class CpuEvalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bench = self.root / "bench"
        self.cases = {f"case-{i:03}" for i in range(108)}
        self.scenarios = {f"scene-{i:02}.mp4" for i in range(66)}
        self.jobs = []
        for suite, count in cpu.COUNTS.items():
            for i in range(count):
                if suite == "physiq":
                    scenario = f"scene-{i // 3:02}.mp4"
                    name = f"{i + 1:04}_perspective-{('left', 'center', 'right')[i % 3]}_{scenario}"
                    output = self.root / "eval/physiq/videos" / (MODEL + "-op-run_01") / name
                    case = f"{i + 1:04}"
                else:
                    case = f"case-{i:03}"
                    output = self.root / "eval" / suite / "videos" / MODEL / (case + ".mp4")
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"fake-video")
                self.jobs.append({"model": MODEL, "id": case, "out": str(output)})
        manifest = self.root / "eval/jobs_merge_all.jsonl"
        manifest.write_text("".join(json.dumps(job) + "\n" for job in self.jobs))
        (self.root / "eval/wm_bench/cases.jsonl").write_text(
            "".join(json.dumps({"id": case}) + "\n" for case in sorted(self.cases)))
        descriptions = self.bench / "physics-IQ-benchmark/descriptions/descriptions_original.csv"
        descriptions.parent.mkdir(parents=True)
        with descriptions.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["scenario", "generated_video_name"])
            writer.writeheader()
            for job in self.physics_jobs:
                name = Path(job["out"]).name
                parts = name.split("_", 2)
                writer.writerow({"scenario": f"{parts[0]}_{parts[1]}_take-1_{parts[2]}",
                                 "generated_video_name": name})
        self.decode_path = self.root / "eval/diag" / f"decode_{MODEL}_20261018.json"
        self.gtref_path = self.root / "eval/wm_bench" / f"gtref_{MODEL}.json"
        self.metrics_path, self.table_path = cpu.physics_result_paths(
            self.root / "eval/physiq/results", MODEL + "-op-run_01")

    @property
    def physics_jobs(self):
        return [job for job in self.jobs if "/physiq/" in job["out"]]

    def write_decode(self):
        self.decode_path.parent.mkdir(parents=True, exist_ok=True)
        self.decode_path.write_text(json.dumps({"model": MODEL, "count": 1011, "expected": 1011,
            "all_frames_decoded": True, "frames": 121, "fps": 24, "width": 1280, "height": 704,
            "outputs": [job["out"] for job in self.jobs]}))

    def gtref(self):
        return {case: dict.fromkeys(cpu.GT_KEYS, 0.1) for case in self.cases}

    def write_physics(self, metrics=None, table=None):
        metrics = metrics or self.metrics_path
        table = table or self.table_path
        metrics.parent.mkdir(parents=True, exist_ok=True)
        metrics.write_text(json.dumps({"final_score_orig": 0.5}))
        with table.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["scenario"])
            writer.writeheader()
            writer.writerows({"scenario": scenario} for scenario in sorted(self.scenarios))

    def test_complete_results_skip_all_decode_and_scoring(self):
        self.write_decode()
        self.gtref_path.write_text(json.dumps(self.gtref()))
        self.write_physics()
        before = {p: p.read_bytes() for p in (self.decode_path, self.gtref_path, self.metrics_path, self.table_path)}
        decode, run = mock.Mock(side_effect=AssertionError("must skip")), mock.Mock(side_effect=AssertionError("must skip"))
        cpu.evaluate_model(self.root, MODEL, decode=decode, run=run, bench=self.bench)
        decode.assert_not_called()
        run.assert_not_called()
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_incomplete_generation_waits_and_never_scores(self):
        Path(self.jobs[-1]["out"]).write_bytes(b"")
        sleep, run, decode = mock.Mock(side_effect=TimeoutError("still waiting")), mock.Mock(), mock.Mock()
        with self.assertRaises(TimeoutError):
            cpu.evaluate_model(self.root, MODEL, sleep=sleep, run=run, decode=decode, bench=self.bench)
        sleep.assert_called_once_with(60)
        run.assert_not_called()
        decode.assert_not_called()
        self.assertFalse(self.decode_path.exists())

    def test_decode_report_rejects_missing_or_reordered_outputs(self):
        self.write_decode()
        report = json.loads(self.decode_path.read_text())
        for outputs in (report["outputs"][:-1], report["outputs"][::-1]):
            with self.subTest(outputs=len(outputs)):
                invalid = dict(report, outputs=outputs)
                self.decode_path.write_text(json.dumps(invalid))
                before = self.decode_path.read_bytes()
                with self.assertRaisesRegex(RuntimeError, "decode report"):
                    cpu.decode_model(self.root, MODEL, self.jobs, decode=mock.Mock())
                self.assertEqual(before, self.decode_path.read_bytes())

    def test_gtref_rejects_missing_wrong_case_missing_value_and_nonfinite(self):
        invalid = self.gtref()
        invalid.pop(next(iter(self.cases)))
        variants = [invalid]
        invalid = self.gtref()
        invalid["unexpected"] = invalid.pop(next(iter(self.cases)))
        variants.append(invalid)
        for value in (None, float("nan"), float("inf"), "0.1", True):
            invalid = self.gtref()
            invalid[next(iter(self.cases))]["dreamsim"] = value
            variants.append(invalid)
        invalid = self.gtref()
        invalid[next(iter(self.cases))].pop("dur")
        variants.append(invalid)
        run = mock.Mock()
        for invalid in variants:
            with self.subTest(invalid=str(invalid)[:80]):
                self.gtref_path.write_text(json.dumps(invalid))
                before = self.gtref_path.read_bytes()
                with self.assertRaises(RuntimeError):
                    cpu.score_gtref(self.root, MODEL, self.cases, run=run)
                self.assertEqual(before, self.gtref_path.read_bytes())
        run.assert_not_called()

    def test_duplicate_json_keys_are_rejected(self):
        self.gtref_path.write_text('{"case": {}, "case": {}}')
        with self.assertRaisesRegex(RuntimeError, "Duplicate JSON key"):
            cpu.validate_gtref(self.gtref_path, self.cases)

    def test_partial_physics_and_nonfinite_score_are_preserved(self):
        run = mock.Mock()
        self.write_physics()
        for value in (None, float("nan"), float("inf"), "0.5", True):
            self.metrics_path.write_text(json.dumps({"final_score_orig": value}))
            before = self.metrics_path.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "Physics score"):
                cpu.score_physics(self.root, MODEL, self.physics_jobs, bench=self.bench, run=run)
            self.assertEqual(before, self.metrics_path.read_bytes())
        self.write_physics()
        self.table_path.write_text("scenario\n" + "\n".join(sorted(self.scenarios)[:-1]) + "\n")
        before = self.table_path.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "Physics66"):
            cpu.score_physics(self.root, MODEL, self.physics_jobs, bench=self.bench, run=run)
        self.assertEqual(before, self.table_path.read_bytes())
        self.table_path.unlink()
        with self.assertRaisesRegex(RuntimeError, "Physics CSV"):
            cpu.score_physics(self.root, MODEL, self.physics_jobs, bench=self.bench, run=run)
        run.assert_not_called()

    def test_wrong_or_duplicate_physics_scenarios_rejected(self):
        self.write_physics()
        for last in ("wrong.mp4", sorted(self.scenarios)[0]):
            self.table_path.write_text("scenario\n" + "\n".join(sorted(self.scenarios)[:-1] + [last]) + "\n")
            with self.assertRaisesRegex(RuntimeError, "Physics66"):
                cpu.validate_physics(self.metrics_path, self.table_path, self.scenarios)

    def test_physics_missing_or_extra_input_rejected_even_with_valid_scores(self):
        self.write_physics()
        missing = Path(self.physics_jobs[0]["out"])
        missing.unlink()
        with self.assertRaisesRegex(RuntimeError, "Physics198 input"):
            cpu.score_physics(self.root, MODEL, self.physics_jobs, bench=self.bench, run=mock.Mock())
        missing.write_bytes(b"fake-video")
        (missing.parent / "unexpected.mp4").write_bytes(b"fake-video")
        with self.assertRaisesRegex(RuntimeError, "Physics198 input"):
            cpu.score_physics(self.root, MODEL, self.physics_jobs, bench=self.bench, run=mock.Mock())

    def test_gtref_subprocess_failure_propagates_preserves_work_and_blocks_retry(self):
        failure = subprocess.CalledProcessError(7, ["mock-scorer"])
        run = mock.Mock(side_effect=failure)
        with self.assertRaisesRegex(RuntimeError, "preserved work") as caught:
            cpu.score_gtref(self.root, MODEL, self.cases, run=run)
        self.assertIs(caught.exception.__cause__, failure)
        self.assertTrue(run.call_args.kwargs["check"])
        self.assertFalse(self.gtref_path.exists())
        with self.assertRaisesRegex(RuntimeError, "Unreviewed partial work"):
            cpu.score_gtref(self.root, MODEL, self.cases, run=run)
        self.assertEqual(run.call_count, 1)

    def test_gtref_scores_into_isolated_work_and_publishes(self):
        def run(command, **kwargs):
            work = Path(command[-2])
            self.assertNotEqual(work, self.gtref_path.parent)
            self.assertTrue((work / "videos").is_symlink())
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
            (work / self.gtref_path.name).write_text(json.dumps(self.gtref()))
        cpu.score_gtref(self.root, MODEL, self.cases, run=run)
        cpu.validate_gtref(self.gtref_path, self.cases)
        self.assertFalse(list(self.gtref_path.parent.glob(f".{MODEL}.gtref.work.*")))

    def test_gtref_wrapper_executes_existing_main_with_overridden_output_root(self):
        scorer = self.root / "scripts/gt_ref_metrics.py"
        scorer.parent.mkdir()
        scorer.write_text(
            "import json,pathlib,sys\nROOT='incorrect'; EV='incorrect'\n"
            "def main():\n"
            " assert pathlib.Path(ROOT).joinpath('eval').is_dir()\n"
            " cases=[json.loads(line)['id'] for line in pathlib.Path(EV).joinpath('cases.jsonl').open()]\n"
            " pathlib.Path(EV).joinpath('gtref_'+sys.argv[1]+'.json').write_text(json.dumps(\n"
            "  {case: dict.fromkeys(('dreamsim','dreamsim_last','motion_ratio','dur'),0.1) for case in cases}))\n")
        with mock.patch.object(cpu, "VBench_PYTHON", sys.executable):
            cpu.score_gtref(self.root, MODEL, self.cases)
        cpu.validate_gtref(self.gtref_path, self.cases)

    def test_full_decode_checks_every_frame_and_rejects_format_or_count(self):
        stream = mock.Mock(average_rate=24, width=1280, height=704)
        frame = mock.Mock(width=1280, height=704)
        container = mock.Mock()
        container.streams.video = [stream]
        container.decode.return_value = [frame] * 121
        av = mock.MagicMock()
        av.open.return_value.__enter__.return_value = container
        with mock.patch.dict("sys.modules", {"av": av}):
            self.assertEqual(cpu.check_video(Path("fake.mp4")), "fake.mp4")
            container.decode.assert_called_once_with(video=0)
            container.decode.return_value = [frame] * 120
            with self.assertRaisesRegex(RuntimeError, "got 120"):
                cpu.check_video(Path("fake.mp4"))
            stream.average_rate = 30
            with self.assertRaisesRegex(RuntimeError, "Unexpected video format"):
                cpu.check_video(Path("fake.mp4"))
            stream.average_rate = 24
            bad_frame = mock.Mock(width=1280, height=720)
            container.decode.return_value = [frame] * 120 + [bad_frame]
            with self.assertRaisesRegex(RuntimeError, "Unexpected frame size"):
                cpu.check_video(Path("fake.mp4"))

    def test_physics_scores_isolated_output_and_appends_existing_log(self):
        log = self.root / "dataset/download_logs/score_physiq.log"
        log.parent.mkdir(parents=True)
        log.write_text("Phase E must remain\n")
        def run(command, **kwargs):
            work = Path(command[command.index("--output_folder") + 1])
            self.assertNotEqual(work, self.metrics_path.parent.parent.parent)
            self.assertIn("--original_physics_iq", command)
            self.assertEqual(command[-2:], ["--n_process", "16"])
            kwargs["stdout"].write("mock Physics completion\n")
            self.write_physics(*cpu.physics_result_paths(work, MODEL + "-op-run_01"))
        with mock.patch.object(cpu, "prepare_physics") as prepare:
            cpu.score_physics(self.root, MODEL, self.physics_jobs, bench=self.bench, run=run)
        prepare.assert_called_once()
        cpu.validate_physics(self.metrics_path, self.table_path, self.scenarios)
        self.assertEqual(log.read_text(), "Phase E must remain\nmock Physics completion\n")
        self.assertFalse(list((self.root / "eval/physiq/results").glob(f".{MODEL}.physiq.work.*")))

    def test_physics_subprocess_failure_propagates_without_publishing(self):
        failure = subprocess.CalledProcessError(9, ["mock-physics"])
        with mock.patch.object(cpu, "prepare_physics"):
            with self.assertRaisesRegex(RuntimeError, "preserved work") as caught:
                cpu.score_physics(self.root, MODEL, self.physics_jobs, bench=self.bench,
                                  run=mock.Mock(side_effect=failure))
        self.assertIs(caught.exception.__cause__, failure)
        self.assertFalse(self.metrics_path.exists())
        self.assertFalse(self.table_path.exists())

    def test_decode_atomic_publication_and_failure(self):
        decode = mock.Mock(side_effect=lambda path: str(path))
        cpu.decode_model(self.root, MODEL, self.jobs, decode=decode)
        self.assertEqual(len(decode.call_args_list), 1011)
        cpu.validate_decode(self.decode_path, MODEL, self.jobs)
        self.assertFalse(list(self.decode_path.parent.glob(f".{self.decode_path.name}.cpu-*")))
        self.decode_path.unlink()
        with self.assertRaisesRegex(RuntimeError, "bad video"):
            cpu.decode_model(self.root, MODEL, self.jobs, decode=mock.Mock(side_effect=RuntimeError("bad video")))
        self.assertFalse(self.decode_path.exists())

    def test_pad_uses_bundled_libx264_original_protocol_and_atomic_destination(self):
        source = Path(self.physics_jobs[0]["out"])
        destination = self.root / "padded"
        def run(command, **kwargs):
            self.assertEqual(command[0], "/bundled/ffmpeg")
            self.assertIn("libx264", command)
            self.assertEqual(command[command.index("-frames:v") + 1], "120")
            self.assertEqual(command[command.index("-crf") + 1], "12")
            self.assertEqual(command[command.index("-vf") + 1],
                             "pad=1280:720:0:8,fillborders=top=8:bottom=8:mode=smear")
            self.assertNotEqual(Path(command[-1]), destination / source.name)
            Path(command[-1]).write_bytes(b"mock-padded")
        ffmpeg = mock.Mock()
        ffmpeg.get_ffmpeg_exe.return_value = "/bundled/ffmpeg"
        decode = mock.Mock()
        with mock.patch.dict("sys.modules", {"imageio_ffmpeg": ffmpeg}):
            cpu.prepare_physics(source.parent, [source], destination, run=run, decode=decode)
            cpu.prepare_physics(source.parent, [source], destination, run=mock.Mock(side_effect=AssertionError()), decode=decode)
        self.assertEqual((destination / source.name).read_bytes(), b"mock-padded")
        self.assertTrue(all(call.kwargs == {"frames": 120, "height": 720} for call in decode.call_args_list))

    def test_flock_excludes_same_model_suite_but_not_different_suite(self):
        context = multiprocessing.get_context("fork")
        entered, release = context.Event(), context.Event()
        other_entered = context.Event()
        with cpu.suite_lock(self.root, MODEL, "gtref"):
            process = context.Process(target=lock_waiter, args=(str(self.root), "gtref", entered, release))
            other = context.Process(target=lock_waiter, args=(str(self.root), "physiq", other_entered, release))
            process.start()
            other.start()
            try:
                self.assertFalse(entered.wait(0.2))
                self.assertTrue(other_entered.wait(3))
            finally:
                release.set()
        self.assertTrue(entered.wait(3))
        process.join(5)
        other.join(5)
        self.assertEqual(process.exitcode, 0)
        self.assertEqual(other.exitcode, 0)

    def test_result_created_while_waiting_for_lock_is_rechecked(self):
        @cpu.contextmanager
        def lock(*args):
            self.gtref_path.write_text(json.dumps(self.gtref()))
            yield
        run = mock.Mock()
        with mock.patch.object(cpu, "suite_lock", lock):
            cpu.score_gtref(self.root, MODEL, self.cases, run=run)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
