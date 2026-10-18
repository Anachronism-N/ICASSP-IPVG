import csv
import importlib.util
import json
import sys
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "analyze_merge_metrics", Path(__file__).resolve().parents[1] / "scripts/analyze_merge_metrics.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class FakeIQTable:
    views = ["perspective-left", "perspective-center", "perspective-right"]
    spatiotemporal_iou_key = "spatiotemporal_iou_v1"
    variance_spatiotemporal_iou_key = "variance_spatiotemporal_iou"
    spatial_iou_key = "spatial_iou_v1"
    variance_spatial_key = "variance_spatial"
    weighted_spatial_iou_key = "weighted_spatial_iou_v1"
    variance_weighted_spatial_key = "variance_weighted_spatial"
    mse_key = "v1_mse"
    variance_mse_key = "variance_mse"

    @classmethod
    def get_list_keys(cls):
        return [cls.spatiotemporal_iou_key, cls.variance_spatiotemporal_iou_key, cls.mse_key, cls.variance_mse_key]

    @classmethod
    def get_scalar_keys(cls):
        return [cls.spatial_iou_key, cls.variance_spatial_key, cls.weighted_spatial_iou_key, cls.variance_weighted_spatial_key]

    @classmethod
    def get_list_columns(cls):
        return [f"{metric}_{view}" for metric in cls.get_list_keys() for view in cls.views]

    def __init__(self, frame):
        self.df = frame.copy()
        for metric in self.get_list_keys():
            arrays = [np.array(self.df[f"{metric}_{view}"].tolist()) for view in self.views]
            self.df[metric] = np.concatenate(arrays, axis=1).mean(axis=1)
        for metric in self.get_scalar_keys():
            self.df[metric] = self.df[[f"{metric}_{view}" for view in self.views]].mean(axis=1)

    @classmethod
    def from_csv(cls, path):
        frame = pd.read_csv(path)
        for column in cls.get_list_columns():
            frame[column] = frame[column].apply(lambda value: [round(x, 4) for x in json.loads(value)])
        return cls(frame)

    def compute_final_score_orig(self):
        means = self.df.mean(numeric_only=True)
        return np.clip((means[self.spatiotemporal_iou_key] / means[self.variance_spatiotemporal_iou_key] +
                        means[self.spatial_iou_key] / means[self.variance_spatial_key] +
                        means[self.weighted_spatial_iou_key] / means[self.variance_weighted_spatial_key]) / 3 -
                        (means[self.mse_key] - means[self.variance_mse_key]), 0, 1)


class MergeAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bench = self.root / "bench"
        self.models = ["m_full1_a03", "m_full1_a05"]
        self.cases = [f"case-{i:03}" for i in range(108)]
        self.scenarios = [f"scene-{i:02}.mp4" for i in range(66)]
        wm = self.root / "eval/wm_bench"
        wm.mkdir(parents=True)
        (wm / "cases.jsonl").write_text("".join(json.dumps({"id": case}) + "\n" for case in self.cases))
        descriptions = self.bench / "descriptions/descriptions_original.csv"
        descriptions.parent.mkdir(parents=True)
        with descriptions.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["scenario", "generated_video_name"])
            writer.writeheader()
            for i, scenario in enumerate(self.scenarios):
                for j, view in enumerate(FakeIQTable.views):
                    writer.writerow({"scenario": f"{i * 3 + j + 1:04}_{view}_take-1_{scenario}",
                                     "generated_video_name": f"{i * 3 + j + 1:04}_{view}_{scenario}"})
        for i, model in enumerate(self.models + list(analysis.REFERENCES)):
            result = {case: {"dreamsim": 0.2 + i * 0.01 + j * 0.001,
                             "dreamsim_last": 0.3 + i * 0.01 + j * 0.001,
                             "motion_ratio": 0.8 + i * 0.01 + j * 0.001, "dur": 5.0}
                      for j, case in enumerate(self.cases)}
            (wm / f"gtref_{model}.json").write_text(json.dumps(result))
            csv_path, metrics_path = self.paths(model)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            fields = ["scenario"] + FakeIQTable.get_list_columns() + [
                f"{metric}_{view}" for metric in FakeIQTable.get_scalar_keys() for view in FakeIQTable.views]
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for j, scenario in enumerate(self.scenarios):
                    row = {"scenario": scenario}
                    for metric in FakeIQTable.get_list_keys() + FakeIQTable.get_scalar_keys():
                        value = 0.5 if "variance" in metric else 0.1 + 0.01 * i + 0.001 * j
                        for view in FakeIQTable.views:
                            row[f"{metric}_{view}"] = json.dumps([value, value]) if metric in FakeIQTable.get_list_keys() else value
                    writer.writerow(row)
            point = float(FakeIQTable.from_csv(csv_path).compute_final_score_orig())
            metrics_path.write_text(json.dumps({"final_score_orig": point}))

    def paths(self, model):
        base = self.root / "eval/physiq/results/physics-IQ-benchmark/results" / (model + "-op-run_01")
        return base.with_suffix(".csv"), base.with_name(base.name + "_metrics.json")

    def run_analysis(self, **kwargs):
        return analysis.analyze(self.root, self.models, bench=self.bench, iq_class=FakeIQTable,
                                bootstrap=100, seed=20261018, **kwargs)

    def rewrite_csv(self, transform, model="m_full1_a03"):
        path, _ = self.paths(model)
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        fields, rows = transform(fields, rows)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_deterministic_paired_report_and_units(self):
        first, second = self.run_analysis(), self.run_analysis()
        first.pop("created_at")
        second.pop("created_at")
        self.assertEqual(first, second)
        self.assertEqual(first["gtref"]["n_cases"], 108)
        self.assertEqual(first["physics"]["n_scenarios"], 66)
        self.assertEqual(first["references"], {"base": "base", "round4": "R4", "full1": "FT1", "ft2": "FT2"})
        self.assertEqual(len(first["physics"]["paired"]), 9)
        paired = first["gtref"]["paired"]["dreamsim"]["m_full1_a03 minus m_full1_a05"]
        self.assertAlmostEqual(paired["delta"], -0.01)
        self.assertTrue(all(abs(bound + 0.01) < 1e-12 for bound in paired["ci95"]))
        self.assertIn("unadjusted", first["caveat"])
        self.assertEqual(first["physics"]["formula_validation"]["iqtable_resamples_checked_per_model"], 3)
        self.assertEqual(len(first["source_sha256"]), 20)

    def test_shuffled_physics_rows_align_by_original_scenario(self):
        before = self.run_analysis()
        self.rewrite_csv(lambda fields, rows: (fields, rows[::-1]))
        after = self.run_analysis()
        self.assertEqual(before["physics"], after["physics"])

    def test_gtref_missing_or_wrong_case_nonfinite_and_missing_key_fail(self):
        path = self.root / "eval/wm_bench/gtref_m_full1_a03.json"
        original = json.loads(path.read_text())
        variants = []
        missing = dict(original)
        missing.pop(self.cases[0])
        variants.append(missing)
        wrong = dict(original)
        wrong["wrong"] = wrong.pop(self.cases[0])
        variants.append(wrong)
        for value in (None, float("nan"), float("inf"), True):
            invalid = json.loads(json.dumps(original))
            invalid[self.cases[0]]["dreamsim"] = value
            variants.append(invalid)
        missing_value = json.loads(json.dumps(original))
        missing_value[self.cases[0]].pop("dur")
        variants.append(missing_value)
        for invalid in variants:
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(RuntimeError, "GTref"):
                self.run_analysis()

    def test_missing_reference_fails_no_analysis(self):
        self.paths("ft2")[0].unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_analysis()

    def test_duplicate_wrong_or_missing_physics_scenario_fails(self):
        for scenario in (self.scenarios[0], "unexpected.mp4"):
            self.rewrite_csv(lambda fields, rows: (fields, [dict(row, scenario=scenario) if i == 65 else row
                                                          for i, row in enumerate(rows)]))
            with self.assertRaisesRegex(RuntimeError, "Physics66"):
                self.run_analysis()
        self.rewrite_csv(lambda fields, rows: (fields, rows[:-1]))
        with self.assertRaisesRegex(RuntimeError, "Physics66"):
            self.run_analysis()

    def test_nonfinite_or_empty_physics_lists_rejected_before_parser(self):
        column = FakeIQTable.get_list_columns()[0]
        for value in ("[NaN]", "[Infinity]", "[]", "[1, null]", "[true]", "[1, garbage]"):
            self.rewrite_csv(lambda fields, rows: (fields, [dict(row, **{column: value}) if i == 0 else row
                                                          for i, row in enumerate(rows)]))
            with self.assertRaisesRegex(RuntimeError, "Physics list"):
                self.run_analysis()

    def test_missing_view_metric_and_nonfinite_scalar_rejected(self):
        column = FakeIQTable.get_scalar_keys()[0] + "_" + FakeIQTable.views[0]
        self.rewrite_csv(lambda fields, rows: (fields, [dict(row, **{column: "nan"}) if i == 0 else row
                                                      for i, row in enumerate(rows)]))
        with self.assertRaisesRegex(RuntimeError, "Physics scalar"):
            self.run_analysis()
        self.rewrite_csv(lambda fields, rows: ([field for field in fields if field != column],
                                              [{key: value for key, value in row.items() if key != column} for row in rows]))
        with self.assertRaisesRegex(RuntimeError, "metric/view columns"):
            self.run_analysis()

    def test_iqtable_saved_point_match_and_nonfinite_rejected(self):
        _, path = self.paths(self.models[0])
        for value in (0.123, None, float("nan"), True):
            path.write_text(json.dumps({"final_score_orig": value}))
            with self.assertRaises(RuntimeError):
                self.run_analysis()

    def test_original_score_ratio_of_means_not_mean_scenario_ratios_and_clips(self):
        values = np.array([[0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.05],
                           [0.3, 0.8, 0.3, 0.8, 0.3, 0.8, 0.1, 0.05]])
        self.assertAlmostEqual(float(analysis.original_score(values.mean(axis=0))), 0.35)
        self.assertNotAlmostEqual(float(analysis.original_score(values).mean()), 0.35)
        self.assertEqual(float(analysis.original_score(np.array([9, 1, 9, 1, 9, 1, 0, 0]))), 1)
        with self.assertRaisesRegex(RuntimeError, "zero-denominator"):
            analysis.original_score(np.array([1, 0, 1, 1, 1, 1, 0, 0]))

    def test_iqtable_resample_mismatch_fails(self):
        real_score = FakeIQTable.compute_final_score_orig
        calls = 0
        def score(table):
            nonlocal calls
            calls += 1
            return real_score(table) + (0.01 if calls > 6 else 0)
        with mock.patch.object(FakeIQTable, "compute_final_score_orig", score):
            with self.assertRaisesRegex(RuntimeError, "bootstrap formula"):
                self.run_analysis()

    def test_defaults_and_invalid_model_names(self):
        self.assertEqual((analysis.BOOTSTRAP, analysis.SEED), (5000, 20261018))
        for models in ([], ["../bad"], ["base"], [self.models[0], self.models[0]]):
            with self.assertRaisesRegex(RuntimeError, "merge model names"):
                analysis.analyze(self.root, models, bench=self.bench, iq_class=FakeIQTable)

    def test_cli_defaults_are_three_models_four_references_5000_and_fixed_seed(self):
        out = self.root / "cli.json"
        with mock.patch.object(sys, "argv", ["analysis", "--out", str(out)]), \
                mock.patch.object(analysis, "analyze", return_value={"complete": True}) as analyze, \
                mock.patch.object(analysis, "write_atomic") as write:
            analysis.main()
        analyze.assert_called_once_with(analysis.ROOT.resolve(),
            ["m_full1_a03", "m_full1_a05", "m_full1_a07"], bench=analysis.BENCH.resolve(),
            references=["base", "round4", "full1", "ft2"], bootstrap=5000, seed=20261018)
        write.assert_called_once_with(out, {"complete": True})

    def test_cli_custom_parameters_are_forwarded_and_existing_out_skips_analysis(self):
        out = self.root / "cli-custom.json"
        arguments = ["analysis", "--out", str(out), "--models", "m_full1_a03,m_full1_a05",
                     "--references", "base,ft2", "--bootstrap", "31", "--seed", "17",
                     "--root", str(self.root), "--bench", str(self.bench)]
        with mock.patch.object(sys, "argv", arguments), \
                mock.patch.object(analysis, "analyze", return_value={"custom": True}) as analyze, \
                mock.patch.object(analysis, "write_atomic"):
            analysis.main()
        analyze.assert_called_once_with(self.root.resolve(), self.models, bench=self.bench.resolve(),
                                         references=["base", "ft2"], bootstrap=31, seed=17)
        out.write_text("existing")
        with mock.patch.object(sys, "argv", arguments), mock.patch.object(analysis, "analyze") as analyze:
            with self.assertRaisesRegex(RuntimeError, "overwrite"):
                analysis.main()
        analyze.assert_not_called()
        self.assertEqual(out.read_text(), "existing")

    def test_reference_subset_and_custom_bootstrap_seed_change_only_requested_analysis(self):
        report = analysis.analyze(self.root, self.models, bench=self.bench, references=["ft2", "base"],
                                  bootstrap=31, seed=17, iq_class=FakeIQTable)
        self.assertEqual(report["references"], {"ft2": "FT2", "base": "base"})
        self.assertEqual((report["bootstrap"], report["seed"]), (31, 17))
        self.assertEqual(set(report["physics"]["observed"]), set(self.models + ["ft2", "base"]))
        self.assertEqual(len(report["physics"]["paired"]), 5)
        self.assertEqual(len(report["source_sha256"]), 14)

    def test_invalid_references_bootstrap_or_seed_fail_before_loading(self):
        for references in ([], ["base", "base"], ["../escape"], ["m_full1_a03"]):
            with self.assertRaises(RuntimeError):
                analysis.analyze(self.root, self.models, bench=self.bench,
                                 references=references, iq_class=FakeIQTable)
        for parameters in ({"bootstrap": 0}, {"bootstrap": -1}, {"bootstrap": 2.5}, {"seed": -1}, {"seed": 1.5}):
            with self.assertRaisesRegex(RuntimeError, "Bootstrap"):
                analysis.analyze(self.root, self.models, bench=self.bench, iq_class=FakeIQTable, **parameters)

    def test_atomic_output_existing_and_concurrent_file_not_overwritten(self):
        output = self.root / "analysis.json"
        analysis.write_atomic(output, {"complete": True})
        self.assertEqual(json.loads(output.read_text()), {"complete": True})
        self.assertFalse(list(self.root.glob(".analysis.json.*")))
        with self.assertRaisesRegex(RuntimeError, "overwrite"):
            analysis.write_atomic(output, {"other": True})
        self.assertEqual(json.loads(output.read_text()), {"complete": True})
        raced = self.root / "raced.json"
        def concurrent_writer(*args):
            raced.write_text('{"other": true}')
            raise FileExistsError()
        with mock.patch.object(analysis.os, "link", side_effect=concurrent_writer):
            with self.assertRaisesRegex(RuntimeError, "concurrently"):
                analysis.write_atomic(raced, {"ours": True})
        self.assertEqual(json.loads(raced.read_text()), {"other": True})

    def test_original_manifest_requires_all_three_distinct_views(self):
        path = self.bench / "descriptions/descriptions_original.csv"
        path.write_text(path.read_text().replace("0002_perspective-center", "0002_perspective-left"))
        with self.assertRaisesRegex(RuntimeError, "left/center/right"):
            self.run_analysis()

    def test_duplicate_json_keys_and_inconsistent_lists_fail(self):
        path = self.root / "eval/wm_bench/gtref_m_full1_a03.json"
        original = path.read_text()
        path.write_text('{"case": {}, "case": {}}')
        with self.assertRaisesRegex(RuntimeError, "Duplicate JSON key"):
            self.run_analysis()
        path.write_text(original)
        column = FakeIQTable.get_list_columns()[0]
        self.rewrite_csv(lambda fields, rows: (fields, [dict(row, **{column: "[0.2]"}) if i == 0 else row
                                                      for i, row in enumerate(rows)]))
        with self.assertRaisesRegex(RuntimeError, "frame-list lengths"):
            self.run_analysis()

    def test_broken_existing_output_symlink_is_preserved(self):
        output = self.root / "broken.json"
        output.symlink_to(self.root / "missing.json")
        with self.assertRaisesRegex(RuntimeError, "overwrite"):
            analysis.write_atomic(output, {"ours": True})
        self.assertTrue(output.is_symlink())

    def test_nonfinite_report_never_published(self):
        output = self.root / "invalid.json"
        with self.assertRaises(ValueError):
            analysis.write_atomic(output, {"bad": float("nan")})
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
