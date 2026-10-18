import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
wmb = importlib.import_module("wmb_merge_score")
gate = importlib.import_module("merge_final_gate")
cpu = importlib.import_module("score_merge_cpu")
val = importlib.import_module("val_merge_score")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")


def write_jobs(path, jobs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(job) + "\n" for job in jobs))


def wmb_result(names):
    return {
        "preds": {name: {"instruction": ["Score: 2."],
                         "physical_laws": ["No"] * 5,
                         "common_sense": ["Yes", "No"]} for name in names},
        "accs": {"instruction": [2.0] * len(names),
                 "physical_laws": [True] * (5 * len(names)),
                 "common_sense": [False, True] * len(names)},
    }


def jobs_for(root, model, suites):
    return [{"model": model, "id": f"case-{i:03}",
             "out": str(root / "eval" / suite / "videos" / model / f"case-{i:03}.mp4")}
            for suite, count in suites.items() for i in range(count)]


class WmbTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = "m_full1_a05"
        self.bench = self.root / "bench"
        self.names = [f"case-{i:03}" for i in range(350)]
        self.jobs = jobs_for(self.root, self.model, cpu.COUNTS)
        write_jobs(self.root / "eval/jobs_merge_all.jsonl", self.jobs)
        write_jobs(self.root / "eval/wm_bench/cases.jsonl",
                   [{"id": f"case-{i:03}"} for i in range(108)])
        write_json(self.bench / "worldmodelbench.json",
                   [{"first_frame": f"frames/{name}.png", "text_instruction": "Move a ball"}
                    for name in self.names])
        for job in self.jobs:
            if "/wmb/" in job["out"]:
                path = Path(job["out"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"mock-video")
        (self.root / "dataset/download_logs").mkdir(parents=True)
        self.destination = self.root / "eval/wmb/results" / (self.model + ".json")

    def score(self, run):
        wmb.score_model(self.root, self.model, self.bench, "mock-python", "mock-judge", run=run)

    def test_existing_valid_original350_skips_judge_and_preserves_bytes(self):
        write_json(self.destination, wmb_result(self.names))
        before = self.destination.read_bytes()
        run = mock.Mock(side_effect=AssertionError("judge must not run"))
        self.score(run)
        run.assert_not_called()
        self.assertEqual(before, self.destination.read_bytes())

    def test_invalid_existing_results_are_preserved_and_never_rerun(self):
        variants = []
        invalid = wmb_result(self.names)
        invalid["preds"]["wrong"] = invalid["preds"].pop(self.names[0])
        variants.append(invalid)
        for key, count in wmb.KEYS.items():
            invalid = wmb_result(self.names)
            invalid["preds"][self.names[0]][key] = ["No"] * (count - 1)
            variants.append(invalid)
            invalid = wmb_result(self.names)
            invalid["accs"][key].pop()
            variants.append(invalid)
        for key, values in {"instruction": [-1, 4, True, "2", float("nan"), float("inf")],
                            "physical_laws": [-1, 2, "1", float("nan")],
                            "common_sense": [-1, 2, float("inf")]}.items():
            for value in values:
                invalid = wmb_result(self.names)
                invalid["accs"][key][0] = value
                variants.append(invalid)
        for prediction in ("Score: nan", "Score: inf", "Score: 4", "", 2):
            invalid = wmb_result(self.names)
            invalid["preds"][self.names[0]]["instruction"][0] = prediction
            variants.append(invalid)
        invalid = wmb_result(self.names)
        invalid["accs"]["instruction"][0] = 1
        variants.append(invalid)
        run = mock.Mock()
        for invalid in variants:
            with self.subTest(invalid=repr(invalid)[:120]):
                write_json(self.destination, invalid)
                before = self.destination.read_bytes()
                with self.assertRaises((RuntimeError, TypeError, KeyError, ValueError)):
                    self.score(run)
                self.assertEqual(before, self.destination.read_bytes())
        run.assert_not_called()

    def test_wrong_original_basename_set_blocks_even_valid_existing_result(self):
        cases = json.loads((self.bench / "worldmodelbench.json").read_text())
        cases[0]["first_frame"] = "frames/wrong.png"
        write_json(self.bench / "worldmodelbench.json", cases)
        write_json(self.destination, wmb_result(self.names))
        run = mock.Mock()
        with self.assertRaises(RuntimeError):
            self.score(run)
        run.assert_not_called()

    def test_incomplete_or_extra_inputs_never_invoke_judge(self):
        path = self.root / "eval/wmb/videos" / self.model / (self.names[0] + ".mp4")
        path.write_bytes(b"")
        run = mock.Mock()
        with self.assertRaises(RuntimeError):
            self.score(run)
        path.write_bytes(b"mock-video")
        (path.parent / "extra.mp4").write_bytes(b"mock-video")
        with self.assertRaises(RuntimeError):
            self.score(run)
        run.assert_not_called()
        self.assertFalse(self.destination.exists())

    def test_judge_failure_or_invalid_new_result_never_publishes(self):
        run = mock.Mock(side_effect=subprocess.CalledProcessError(7, ["mock-judge"]))
        with self.assertRaises(subprocess.CalledProcessError):
            self.score(run)
        self.assertFalse(self.destination.exists())
        self.assertTrue(run.call_args.kwargs["check"])
        def invalid_run(command, **kwargs):
            result = wmb_result(self.names)
            result["accs"]["instruction"][0] = 4
            write_json(Path(command[command.index("--save_name") + 1]).with_suffix(".json"), result)
        with self.assertRaises(RuntimeError):
            self.score(invalid_run)
        self.assertFalse(self.destination.exists())
        self.assertEqual(len(list(self.bench.glob("migration_*"))), 2)

    def test_new_valid_result_uses_original_cli_and_publishes_then_skips(self):
        def run(command, **kwargs):
            self.assertEqual(kwargs["cwd"], self.bench)
            self.assertTrue(kwargs["check"])
            self.assertNotIn("--cot", command)
            self.assertEqual(command[command.index("--judge") + 1], "mock-judge")
            self.assertEqual(command[command.index("--model_name") + 1], self.model)
            save = Path(command[command.index("--save_name") + 1])
            self.assertNotEqual(save.with_suffix(".json"), self.destination)
            write_json(save.with_suffix(".json"), wmb_result(self.names))
        judge = mock.Mock(side_effect=run)
        self.score(judge)
        wmb.validate_result(self.destination, set(self.names))
        self.score(judge)
        self.assertEqual(judge.call_count, 1)

    def test_smoke_is_one_real_symlink_not_full_or_published_result(self):
        work_parent = self.root / "work"
        work_parent.mkdir()
        def run(command, **kwargs):
            work = kwargs["cwd"]
            self.assertTrue(kwargs["check"])
            self.assertEqual(len(json.loads((work / "worldmodelbench.json").read_text())), 1)
            videos = Path(command[command.index("--video_dir") + 1])
            self.assertEqual(len(list(videos.iterdir())), 1)
            self.assertTrue(next(videos.iterdir()).is_symlink())
            save = Path(command[command.index("--save_name") + 1])
            write_json(save.with_suffix(".json"), wmb_result(self.names[:1]))
        wmb.smoke(self.root, self.bench, "mock-python", "mock-judge", work_parent, run=run)
        self.assertFalse(self.destination.exists())


class FinalGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.merges = [j for model in gate.MODELS for j in jobs_for(self.root, model, cpu.COUNTS)]
        suites = {
            "ft2": cpu.COUNTS,
            "full1_po": {"vbench_i2v": 355, "physiq": 198, "wmb": 350},
            "ft2_po": {"vbench_i2v": 355, "physiq": 198, "wmb": 350},
            "base_po": {"physiq": 198, "wmb": 350},
            "round4_po": {"physiq": 198, "wmb": 350},
        }
        self.ft2 = [j for model, counts in suites.items() for j in jobs_for(self.root, model, counts)]
        self.paths = [str(self.root / "eval/jobs_ft2_all.jsonl"),
                      str(self.root / "eval/jobs_merge_all.jsonl")]
        write_jobs(Path(self.paths[0]), self.ft2)
        write_jobs(Path(self.paths[1]), self.merges)
        write_jobs(self.root / "eval/wm_bench/cases.jsonl", [{"id": f"case-{i:03}"} for i in range(108)])
        clips = [{"video": f"dataset/wan22_5s/mock_5s/clip-{i:03}.mp4", "dataset": "mock", "loss_t": [0.2] * 5, "loss": 0.2}
                 for i in range(120)]
        benchmark = self.root / "official-wmb"
        write_json(benchmark / "worldmodelbench.json",
                   [{"first_frame": f"frames/case-{i:03}.png"} for i in range(350)])
        patcher = mock.patch.object(gate, "WMB_BENCH", benchmark)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.loss = {"n": 120, "num_frames": 121, "clips": clips,
                     "t_fracs": val.T_FRACS, "t_ids": [100, 300, 500, 700, 900],
                     "mean_loss": 0.2, "mean_loss_per_t": [0.2] * 5, "per_dataset": {"mock": 0.2}}
        csv_path = self.root / "training_metadata/round2/round2_val.csv"
        csv_path.parent.mkdir(parents=True)
        csv_path.write_text("video,dataset\n" + "".join(c["video"] + ",mock\n" for c in clips))
        write_json(self.root / gate.LOSS_DIR / "loss_base.json", self.loss)
        for model in gate.MODELS:
            write_json(self.root / gate.LOSS_DIR / ("loss_" + model + ".json"), self.loss)
            jobs = [j for j in self.merges if j["model"] == model]
            write_json(self.decode_path(model), {"model": model, "count": 1011, "expected": 1011,
                       "all_frames_decoded": True, "frames": 121, "fps": 24, "width": 1280, "height": 704,
                       "outputs": [j["out"] for j in jobs]})
        for model in suites.keys() | set(gate.MODELS):
            write_json(self.root / "eval/wmb/results" / (model + ".json"),
                       wmb_result([f"case-{i:03}" for i in range(350)]))
        self.marker = self.root / "eval/diag/merge_eval_complete_20261018.json"
        self.coverage = mock.Mock()
        self.run = mock.Mock()

    def decode_path(self, model):
        return self.root / "eval/diag" / ("decode_" + model + "_20261018.json")

    def finish(self):
        gate.finish(self.root, "mock-python", run=self.run, validate_jobs=self.coverage)

    def assert_blocked(self):
        with self.assertRaises((RuntimeError, OSError, KeyError, TypeError, ValueError)):
            self.finish()
        self.run.assert_not_called()
        self.assertFalse(self.marker.exists())

    def test_complete_gate_validates_both_manifests_before_and_after_reporting(self):
        self.finish()
        self.assertEqual(self.coverage.call_args_list, [mock.call(self.paths), mock.call(self.paths)])
        self.run.assert_called_once()
        self.assertEqual(self.run.call_args.args[0][-2:], self.paths)
        self.assertTrue(self.run.call_args.kwargs["check"])
        report = json.loads(self.marker.read_text())
        self.assertEqual(report["status"], "COMPLETE")
        self.assertEqual(report["video_expected"], 3033)
        self.assertEqual(report["fixed_val_n"], 120)
        self.assertIn("Teacher quality gate and user review are separate", report["caveat"])
        self.assertIn("no Student/FT3 training authorized", report["caveat"])

    def test_empty_subset_duplicate_and_wrong_ft2_suite_manifests_block_reports(self):
        variants = [[], self.ft2[:-1], self.ft2[:-1] + [self.ft2[0]]]
        changed = copy.deepcopy(self.ft2)
        changed[0]["out"] = str(self.root / "eval/wmb/videos/ft2/wrong.mp4")
        variants.append(changed)
        for jobs in variants:
            with self.subTest(size=len(jobs)):
                write_jobs(Path(self.paths[0]), jobs)
                self.assert_blocked()

    def test_merge_subset_wrong_model_and_duplicate_block_reports(self):
        for change in ("missing", "wrong-model", "duplicate"):
            with self.subTest(change=change):
                jobs = copy.deepcopy(self.merges)
                if change == "missing":
                    jobs.pop()
                elif change == "wrong-model":
                    jobs[-1]["model"] = "unknown"
                else:
                    jobs[-1]["out"] = jobs[0]["out"]
                write_jobs(Path(self.paths[1]), jobs)
                self.assert_blocked()

    def test_each_model_requires_full1011_decode_with_same_outputs(self):
        for model in gate.MODELS:
            path = self.decode_path(model)
            original = json.loads(path.read_text())
            for change in ("missing", "subset", "reordered", "not-decoded"):
                with self.subTest(model=model, change=change):
                    report = copy.deepcopy(original)
                    if change == "missing":
                        path.unlink()
                    else:
                        if change == "subset":
                            report["outputs"].pop()
                        elif change == "reordered":
                            report["outputs"].reverse()
                        else:
                            report["all_frames_decoded"] = False
                        write_json(path, report)
                    self.assert_blocked()
                    write_json(path, original)

    def test_each_model_requires_fixed120_same_video_order_five_t_ids_and_finite_loss(self):
        for model in gate.MODELS:
            path = self.root / gate.LOSS_DIR / ("loss_" + model + ".json")
            for change in ("missing", "subset", "order", "t_ids", "four-t", "nan", "wrong-mean"):
                with self.subTest(model=model, change=change):
                    loss = copy.deepcopy(self.loss)
                    if change == "missing":
                        path.unlink()
                    else:
                        if change == "subset":
                            loss["clips"].pop()
                        elif change == "order":
                            loss["clips"].reverse()
                        elif change == "t_ids":
                            loss["t_ids"][0] += 1
                        elif change == "four-t":
                            loss["clips"][0]["loss_t"].pop()
                        elif change == "nan":
                            loss["clips"][0]["loss_t"][0] = float("nan")
                        else:
                            loss["mean_loss"] = 0.3
                        write_json(path, loss)
                    self.assert_blocked()
                    write_json(path, self.loss)

    def test_synchronized_wrong_wmb_manifest_and_predictions_still_block(self):
        for path, jobs in zip(self.paths, (self.ft2, self.merges)):
            changed = copy.deepcopy(jobs)
            for job in changed:
                if "/wmb/" in job["out"] and Path(job["out"]).stem == "case-000":
                    job["out"] = str(Path(job["out"]).with_name("wrong.mp4"))
            write_jobs(Path(path), changed)
        for path in (self.root / "eval/wmb/results").glob("*.json"):
            result = json.loads(path.read_text())
            result["preds"]["wrong"] = result["preds"].pop("case-000")
            write_json(path, result)
        self.assert_blocked()

    def test_invalid_wmb_pred_or_range_blocks_final_even_if_summary_accepts(self):
        path = self.root / "eval/wmb/results/m_full1_a03.json"
        original = json.loads(path.read_text())
        for change in ("pred-count", "out-of-range"):
            with self.subTest(change=change):
                result = copy.deepcopy(original)
                if change == "pred-count":
                    result["preds"]["case-000"]["physical_laws"].pop()
                else:
                    result["accs"]["instruction"][0] = 4
                write_json(path, result)
                self.assert_blocked()

    def test_missing_score_coverage_never_reports_or_marks(self):
        self.coverage.side_effect = RuntimeError("Evaluation incomplete: missing score")
        self.assert_blocked()
        self.coverage.assert_called_once_with(self.paths)

    def test_report_subprocess_failure_has_no_marker(self):
        self.run.side_effect = subprocess.CalledProcessError(9, ["mock-report"])
        with self.assertRaises(subprocess.CalledProcessError):
            self.finish()
        self.assertFalse(self.marker.exists())
        self.coverage.assert_called_once_with(self.paths)

    def test_post_report_recheck_failure_has_no_marker(self):
        self.coverage.side_effect = [None, RuntimeError("score disappeared during report")]
        with self.assertRaises(RuntimeError):
            self.finish()
        self.run.assert_called_once()
        self.assertFalse(self.marker.exists())

    def test_actual_summary_coverage_rejects_missing_scores_without_reporting(self):
        write_jobs(self.root / "eval/vbench_i2v/cases.jsonl",
                   [{"id": f"case-{i:03}"} for i in range(355)])
        summary = importlib.import_module("summarize_eval")
        with mock.patch.object(summary, "ROOT", str(self.root)), \
             mock.patch.object(summary, "EV", str(self.root / "eval")), \
             mock.patch.object(gate, "summary_api", return_value=summary.validate_jobs):
            with self.assertRaisesRegex(RuntimeError, "missing score"):
                gate.finish(self.root, "mock-python", run=self.run)
        self.run.assert_not_called()
        self.assertFalse(self.marker.exists())

    def test_report_wrapper_executes_both_local_reporters_with_original_manifests(self):
        scripts = self.root / "scripts"
        scripts.mkdir()
        (scripts / "summarize_eval.py").write_text(
            "import json,sys\nfrom pathlib import Path\nROOT='wrong'; EV='wrong'\n"
            "def main():\n"
            " assert EV == ROOT + '/eval'\n"
            " assert sys.argv[1:] == ['--validate-jobs',EV+'/jobs_ft2_all.jsonl',"
            "'--validate-jobs',EV+'/jobs_merge_all.jsonl']\n"
            " Path(EV+'/mock_summary.json').write_text(json.dumps({'reported':True}))\n")
        (scripts / "build_demo.py").write_text("ROOT='wrong'; DEMO='wrong'; MEDIA='wrong'\n")
        (scripts / "build_teacher_review.py").write_text(
            "import sys\nfrom pathlib import Path\nimport build_demo\n"
            "ROOT='wrong'; DEMO='wrong'; MEDIA='wrong'; EV='wrong'; WM='wrong'; LOSS='wrong'\n"
            "def main():\n"
            " assert sys.argv == ['build_teacher_review.py']\n"
            " assert build_demo.ROOT == ROOT and build_demo.DEMO == DEMO\n"
            " assert DEMO == ROOT+'/demo' and MEDIA == DEMO+'/media'\n"
            " assert EV == ROOT+'/eval' and WM == EV+'/wm_bench'\n"
            " assert LOSS == ROOT+'/models/train/round2_5s_lora_r32/eval'\n"
            " assert Path(EV+'/mock_summary.json').is_file()\n"
            " Path(EV+'/mock_teacher_review').write_text('quality gate stays separate')\n")
        gate.finish(self.root, sys.executable, run=subprocess.run, validate_jobs=self.coverage)
        self.assertTrue((self.root / "eval/mock_teacher_review").is_file())
        self.assertTrue(self.marker.exists())
        self.assertEqual(self.coverage.call_args_list, [mock.call(self.paths), mock.call(self.paths)])

    def test_default_summary_validator_is_loaded_with_requested_root(self):
        script = self.root / "scripts/summarize_eval.py"
        script.parent.mkdir()
        script.write_text("ROOT='wrong'\nEV='wrong'\ndef validate_jobs(paths):\n"
                          " assert EV == ROOT + '/eval'\n"
                          " assert paths[0] == EV + '/jobs_ft2_all.jsonl'\n"
                          " raise RuntimeError('real coverage validator called')\n")
        with self.assertRaisesRegex(RuntimeError, "real coverage validator called"):
            gate.check_complete(self.root)
        self.assertFalse(self.marker.exists())


if __name__ == "__main__":
    unittest.main()
