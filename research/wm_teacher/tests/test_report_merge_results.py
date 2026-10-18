import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("report_merge_results", SCRIPTS / "report_merge_results.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / "new.md"
        self.bench = self.root / "bench/physics-IQ-benchmark"
        self.wmb_bench = self.root / "bench/WorldModelBench"
        self.cases = [f"case-{i:03}" for i in range(108)]
        self.public = [f"public-{i:03}" for i in range(355)]
        self.scenarios = [f"scene-{i:02}.mp4" for i in range(66)]
        self.stems = [f"wmb-{i:03}" for i in range(350)]
        self.write_lines("eval/wm_bench/cases.jsonl", [{"id": c} for c in self.cases])
        self.write_lines("eval/vbench_i2v/cases.jsonl", [{"id": c} for c in self.public])
        descriptions = self.bench / "descriptions/descriptions_original.csv"
        descriptions.parent.mkdir(parents=True)
        with descriptions.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["scenario", "generated_video_name"])
            writer.writeheader()
            for i, scene in enumerate(self.scenarios):
                for j, view in enumerate(("perspective-left", "perspective-center", "perspective-right")):
                    writer.writerow({"scenario": f"{i * 3 + j:04}_{view}_take-1_{scene}",
                                     "generated_video_name": f"{i * 3 + j:04}_{view}_{scene}"})
        self.write_json(self.wmb_bench / "worldmodelbench.json",
                        [{"first_frame": stem + ".png"} for stem in self.stems])
        clips = [f"dataset/source/fixture_5s/{i}.mp4" for i in range(120)]
        self.write_lines("training_metadata/round2/round2_val.csv", [])
        with (self.root / "training_metadata/round2/round2_val.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["video"])
            writer.writeheader()
            writer.writerows({"video": c} for c in clips)
        for index, model in enumerate((*report.CONTROLS, *report.MODELS, *report.BASELINES)):
            gt = {case: {"dreamsim": 0.2 + index * 0.01, "dreamsim_last": 0.3 + index * 0.01,
                         "motion_ratio": 0.5 + index * 0.01, "dur": 5.0} for case in self.cases}
            self.write_json(f"eval/wm_bench/gtref_{model}.json", gt)
            if model in report.BASELINES:
                continue
            for suite, ids in (("wm_bench", self.cases), ("vbench_i2v", self.public)):
                if suite == "vbench_i2v" and model == "full1":
                    continue  # FT1 raw public was not measured; _po is not a substitute.
                for dim in report.DIMS:
                    self.write_json(f"eval/{suite}/scores/{model}__{dim}_eval_results.json",
                                    {dim: [0.8, [{"video_path": c + "-0.mp4", "video_results": 0.8}
                                                for c in ids]]})
            value = 1.0 - index * 0.05
            loss = {"n": 120, "num_frames": 121, "t_fracs": [0.1, 0.3, 0.5, 0.7, 0.9],
                    "t_ids": [900, 700, 500, 300, 100], "mean_loss": value,
                    "mean_loss_per_t": [value] * 5, "per_dataset": {"fixture": value},
                    "clips": [{"video": c, "dataset": "fixture", "loss_t": [value] * 5, "loss": value}
                              for c in clips]}
            self.write_json(f"{report.LOSS_DIR}/loss_{model}.json", loss)
            self.write_json(f"eval/physiq/results/physics-IQ-benchmark/results/{model}-op-run_01_metrics.json",
                            {"final_score_orig": 0.3013037868892134, "final_score_origround": 30.13})
            csv_path = self.root / f"eval/physiq/results/physics-IQ-benchmark/results/{model}-op-run_01.csv"
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["scenario"])
                writer.writeheader()
                writer.writerows({"scenario": s} for s in self.scenarios)
            wb = {"preds": {s: {"instruction": ["score: 2"], "physical_laws": ["no"] * 5,
                                  "common_sense": ["no"] * 2} for s in self.stems},
                  "accs": {"instruction": [2.0] * 350, "physical_laws": [True] * 1750,
                           "common_sense": [True] * 700}}
            self.write_json(f"eval/wmb/results/{model}.json", wb)
        lb = self.root / "eval/vbench_i2v_leaderboard_20261011.csv"
        with lb.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["Model Name (clickable)", *report.LB_COL.values()])
            writer.writeheader()
            writer.writerows({"Model Name (clickable)": p, **{c: "90.0%" for c in report.LB_COL.values()}}
                             for p in report.PEERS)
        inventory = []
        for name in report.CHECKPOINTS.values():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture checkpoint")
            inventory.append({"relative_path": name, "size": path.stat().st_size, "sha256": "a" * 64})
        self.write_json(report.MIGRATION, {"files": inventory, "root": "private-root-not-for-report"})
        for name in ("report_merge_results.py", "merge_final_gate.py", "score_merge_cpu.py", "val_merge_score.py",
                     "wmb_merge_score.py", "summarize_eval.py", "build_teacher_review.py", "analyze_merge_metrics.py"):
            self.write_json("scripts/" + name, {"fixture": True})
        for name in ("jobs_merge_all.jsonl", "jobs_ft2_all.jsonl"):
            self.write_lines("eval/" + name, [{"fixture": True}])
        for model in report.MODELS:
            self.write_json(f"eval/diag/decode_{model}_20261018.json", {"fixture": True})
        self.make_analysis()

    def write_json(self, path, data):
        path = Path(path)
        path = path if path.is_absolute() else self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def write_lines(self, path, data):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(x) + "\n" for x in data))

    def make_analysis(self):
        models = (*report.MODELS, *report.CONTROLS)
        paired_sources = {"eval/wm_bench/cases.jsonl": report.sha256(self.root / "eval/wm_bench/cases.jsonl"),
                          "Physics-IQ/descriptions/descriptions_original.csv": report.sha256(
                              self.bench / "descriptions/descriptions_original.csv")}
        observed = {m: json.loads((self.root / f"eval/wm_bench/gtref_{m}.json").read_text())[self.cases[0]]
                    for m in models}
        pairs = [m + " minus " + c for m in report.MODELS for c in report.CONTROLS] + [
            report.MODELS[i] + " minus " + report.MODELS[j] for i in range(3) for j in range(i + 1, 3)]
        paired = {}
        for field in ("dreamsim", "dreamsim_last", "motion_ratio"):
            paired[field] = {}
            for pair in pairs:
                a, b = pair.split(" minus ")
                delta = observed[a][field] - observed[b][field]
                paired[field][pair] = {"delta": delta, "ci95": [delta - 0.01, delta + 0.01]}
        for model in models:
            for name in (f"eval/wm_bench/gtref_{model}.json",
                         f"eval/physiq/results/physics-IQ-benchmark/results/{model}-op-run_01.csv",
                         f"eval/physiq/results/physics-IQ-benchmark/results/{model}-op-run_01_metrics.json"):
                paired_sources[name] = report.sha256(self.root / name)
        self.write_json(report.ANALYSIS, {
            "models": list(report.MODELS), "references": dict.fromkeys(report.CONTROLS, "fixture"),
            "seed": 20261018, "bootstrap": 5000,
            "gtref": {"n_cases": 108, "case_ids": self.cases,
                      "observed": {k: {m: v[k] for m, v in observed.items()} for k in paired}, "paired": paired},
            "physics": {"n_scenarios": 66, "scenario_ids": self.scenarios, "views_per_scenario": 3,
                        "observed": dict.fromkeys(models, 30.13037868892134),
                        "paired": {p: {"delta": 0.0, "ci95": [-1.0, 1.0]} for p in pairs},
                        "formula_validation": {"iqtable_point_matches_saved_metric_abs_tolerance": 1e-8,
                                               "iqtable_resamples_checked_per_model": 3}},
            "source_sha256": paired_sources})

    def load(self, gate=None):
        return report.load_report(self.root, gate=gate or mock.Mock(),
                                  physics_bench=self.bench, wmb_bench=self.wmb_bench)

    def test_missing_default_gate_writes_nothing(self):
        with mock.patch.object(report.merge_final_gate, "check_complete", side_effect=RuntimeError("pending")) as gate:
            with self.assertRaises(RuntimeError):
                report.generate_report(self.root, self.out)
            gate.assert_called_once_with(self.root)
        self.assertFalse(self.out.exists())

    def test_complete_fixture_units_original_total_and_pending_fail(self):
        gate = mock.Mock()
        data = self.load(gate)
        self.assertEqual(gate.call_count, 2)
        text = report.render_report(data)
        self.assertIn("Teacher 自动条件：FAIL", text)
        self.assertIn("用户门禁：pending", text)
        self.assertIn("| 0.301304 | 30.13% | 9.000 | 2.000 | 5.000 | 2.000 |", text)
        self.assertIn("80.00%", text)
        self.assertIn(report.number(report.total(dict.fromkeys(report.DIMS, 0.8))[0], True, 2), text)
        self.assertIn("未测", text)
        self.assertIn("Physics（pp）", text)
        self.assertNotIn("private-root-not-for-report", text)
        self.assertNotIn(str(self.root), text)
        self.assertTrue(all(not Path(p).is_absolute() for p in data["source_sha256"]))

    def test_imaging_original_scale_not_clamped(self):
        data = self.load()
        data["rows"][report.MODELS[0]]["vbench"]["imaging_quality"] = 60.0
        self.assertIn("6000.00%", report.render_report(data))

    def test_nonfinite_aggregate_and_per_case_rejected(self):
        for index in (0, 1):
            with self.subTest(index=index):
                row = {"subject_consistency": [0.8, [{"video_path": "case-0.mp4", "video_results": 0.8}]]}
                if index == 0:
                    row["subject_consistency"][0] = float("nan")
                else:
                    row["subject_consistency"][1][0]["video_results"] = float("inf")
                with self.assertRaises(RuntimeError):
                    report.vbench_values(row, "subject_consistency", {"case-0.mp4"})
        with self.assertRaises(RuntimeError):
            report.number(float("nan"))

    def test_wrong_score_case_rejected(self):
        model = report.MODELS[0]
        self.write_json(f"eval/vbench_i2v/scores/{model}__i2v_subject_eval_results.json",
                        {"i2v_subject": [0.8, [{"video_path": "wrong-0.mp4", "video_results": 0.8}] * 355]})
        with self.assertRaisesRegex(RuntimeError, "case coverage"):
            self.load()

    def test_wrong_gtref_case_rejected(self):
        path = self.root / f"eval/wm_bench/gtref_{report.BASELINES[0]}.json"
        gt = json.loads(path.read_text())
        gt["wrong"] = gt.pop(self.cases[0])
        self.write_json(path, gt)
        with self.assertRaisesRegex(RuntimeError, "GTref108"):
            self.load()

    def test_missing_required_baseline_loss_rejected(self):
        (self.root / report.LOSS_DIR / "loss_round4.json").unlink()
        with self.assertRaises((RuntimeError, OSError)):
            self.load()

    def test_partial_optional_public_is_not_unmeasured(self):
        self.write_json("eval/vbench_i2v/scores/full1__i2v_subject_eval_results.json", {})
        with self.assertRaisesRegex(RuntimeError, "Partial raw"):
            self.load()

    def test_stale_paired_hash_rejected(self):
        path = self.root / report.ANALYSIS
        data = json.loads(path.read_text())
        data["source_sha256"]["eval/wm_bench/cases.jsonl"] = "0" * 64
        self.write_json(path, data)
        with self.assertRaisesRegex(RuntimeError, "Stale paired source"):
            self.load()

    def test_nonfinite_paired_ci_rejected(self):
        path = self.root / report.ANALYSIS
        data = json.loads(path.read_text())
        next(iter(data["physics"]["paired"].values()))["ci95"][0] = float("nan")
        self.write_json(path, data)
        with self.assertRaisesRegex(RuntimeError, "paired CI"):
            self.load()

    def test_unmeasured_teacher_baseline_is_not_new_scoring_requirement(self):
        missing = report.BASELINES[0]
        (self.root / f"eval/wm_bench/gtref_{missing}.json").unlink()
        data = self.load()
        self.assertIsNone(data["rows"][missing]["dreamsim"])
        self.assertTrue(all(missing not in t["compared"] for t in data["teacher"].values()))
        self.assertIn("未测", report.render_report(data))

    def test_all_unmeasured_related_baselines_compare_only_existing_base(self):
        for model in report.BASELINES:
            (self.root / f"eval/wm_bench/gtref_{model}.json").unlink()
        data = self.load()
        self.assertTrue(all(t["compared"] == ["base"] for t in data["teacher"].values()))
        self.assertIn("不宣称优于未测者", report.render_report(data))

    def test_loaded_pareto_excludes_controls_even_if_complete_and_dominating(self):
        data = self.load()
        self.assertTrue(data["pareto"])
        self.assertTrue(set(data["pareto"]) <= set(report.MODELS))
        self.assertNotIn("base", data["pareto"])
        self.assertIn("仅三个完整raw插值候选", report.render_report(data))

    def test_wrong_physics_scenario_rejected(self):
        path = self.root / f"eval/physiq/results/physics-IQ-benchmark/results/{report.MODELS[0]}-op-run_01.csv"
        text = path.read_text().replace(self.scenarios[0], "wrong-scene.mp4")
        path.write_text(text)
        with self.assertRaisesRegex(RuntimeError, "Physics66"):
            self.load()

    def test_nonfinite_physics_raw_metric_rejected(self):
        path = f"eval/physiq/results/physics-IQ-benchmark/results/{report.MODELS[0]}-op-run_01_metrics.json"
        self.write_json(path, {"final_score_orig": 0.3, "score_spatial": float("inf")})
        with self.assertRaisesRegex(RuntimeError, "Non-finite raw Physics"):
            self.load()

    def test_physics_percent_cannot_replace_fraction(self):
        path = f"eval/physiq/results/physics-IQ-benchmark/results/{report.MODELS[0]}-op-run_01_metrics.json"
        self.write_json(path, {"final_score_orig": 30.13})
        with self.assertRaisesRegex(RuntimeError, "fraction outside"):
            self.load()

    def test_display_overflow_rejected(self):
        with self.assertRaises(RuntimeError):
            report.number(1e308, percent=True)

    def test_missing_paired_metric_rejected(self):
        path = self.root / report.ANALYSIS
        data = json.loads(path.read_text())
        del data["gtref"]["paired"]["motion_ratio"]
        self.write_json(path, data)
        with self.assertRaisesRegex(RuntimeError, "Missing paired GT"):
            self.load()

    def test_inconsistent_paired_delta_rejected(self):
        path = self.root / report.ANALYSIS
        data = json.loads(path.read_text())
        next(iter(data["physics"]["paired"].values()))["delta"] = 50.0
        self.write_json(path, data)
        with self.assertRaisesRegex(RuntimeError, "delta differs"):
            self.load()

    def test_wrong_fixed120_order_rejected(self):
        path = self.root / report.LOSS_DIR / f"loss_{report.MODELS[0]}.json"
        data = json.loads(path.read_text())
        data["clips"] = list(reversed(data["clips"]))
        self.write_json(path, data)
        with self.assertRaisesRegex(RuntimeError, "videos/order"):
            self.load()

    def test_pareto_directions_ties_missing_dynamic_ignored(self):
        a = {"vbench_total": 0.9, "dreamsim": 0.1, "physics": 0.3, "wmb": 8.0, "val": 0.5}
        b = {"vbench_total": 0.8, "dreamsim": 0.2, "physics": 0.2, "wmb": 7.0, "val": 0.6,
             "dynamic_degree": 1e9}
        c = {**a, "vbench_total": 0.7, "dreamsim": 0.05}
        rows = {"a": a, "b": b, "c": c, "tie": copy.copy(a), "missing": {**a, "val": None}}
        self.assertEqual(report.pareto(rows), ["a", "c", "tie"])

    def test_automatic_pass_still_human_pending(self):
        data = self.load()
        for model in report.MODELS:
            data["rows"][model].update({"dreamsim": 0.01, "motion_err": 0.01,
                                       "clip_losses": [0.01] * 120, "vbench": dict.fromkeys(report.DIMS, 0.95)})
        data["teacher"] = report.teacher_checks(data["rows"], data["peers"])
        text = report.render_report(data)
        self.assertIn("Teacher 自动条件：PASS", text)
        self.assertIn("用户门禁：pending", text)
        self.assertIn("不得宣称 Teacher 通过或 SOTA", text)
        self.assertTrue(all(x["human_review"] == "pending" for x in data["teacher"].values()))

    def test_output_not_overwritten_or_symlink_followed(self):
        self.out.write_text("original")
        with self.assertRaises(RuntimeError):
            report.write_no_clobber(self.out, "new")
        self.assertEqual(self.out.read_text(), "original")
        self.out.unlink()
        self.out.symlink_to(self.root / "missing")
        with self.assertRaises(RuntimeError):
            report.write_no_clobber(self.out, "new")
        self.assertFalse((self.root / "missing").exists())

    def test_atomic_racing_writer_preserved(self):
        real_link = report.os.link

        def competing_writer(source, destination):
            Path(destination).write_text("other writer")
            real_link(source, destination)

        with mock.patch.object(report.os, "link", side_effect=competing_writer):
            with self.assertRaisesRegex(RuntimeError, "concurrently"):
                report.write_no_clobber(self.out, "my report")
        self.assertEqual(self.out.read_text(), "other writer")
        self.assertEqual(list(self.root.glob(".new.md.*.tmp")), [])

    def test_complete_mock_report_new_output_only(self):
        data = report.generate_report(self.root, self.out, gate=mock.Mock(),
                                      physics_bench=self.bench, wmb_bench=self.wmb_bench)
        self.assertTrue(self.out.is_file())
        self.assertEqual(self.out.read_text(), report.render_report(data))
        self.assertEqual(list(self.root.glob(".new.md.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
