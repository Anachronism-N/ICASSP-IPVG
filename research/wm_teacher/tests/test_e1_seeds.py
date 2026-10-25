import contextlib
import importlib.util
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("summarize_e1_seeds", SCRIPTS / "summarize_e1_seeds.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class FourSeedTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.roots = {seed: self.root / rel for seed, rel in summary.SEED_DIRS.items()}
        self.ids = [f"private_case_{i:03d}" for i in range(94)]
        self.cases = [{"id": case_id, "dataset": "fixture", "prompt": "private prompt",
                       "image": "/private/images/frame.png", "gt": "/private/gt/video.mp4",
                       "video_sha256": "private-video-hash", "parent_keys": ["private-parent"]}
                      for case_id in self.ids]
        self.shifts = [-0.01, 0.02, -0.03, 0.04]
        self.dynamic_flips = [2, -2, 1, 0]
        for seed_index, (seed, root) in enumerate(self.roots.items()):
            (root / "scores").mkdir(parents=True)
            (root / "gtref").mkdir()
            self.write_cases(root, self.cases)
            for config in summary.CONFIGS:
                delta = self.shifts[seed_index] if config == summary.CANDIDATE_CONFIG else 0
                flips = self.dynamic_flips[seed_index] if config == summary.CANDIDATE_CONFIG else 0
                for dim in summary.DIMS:
                    values = []
                    for i, case_id in enumerate(self.ids):
                        if dim == "dynamic_degree":
                            value = i < 47 + flips
                        elif dim == "imaging_quality":
                            value = 65 + 100 * delta
                        elif dim == "aesthetic_quality":
                            value = 0.5 + delta + i / 10000
                        else:
                            value = 0.95 + delta
                        values.append({"video_path": f"/private/{case_id}-0.mp4", "video_results": value})
                    aggregate = sum(v["video_results"] for v in values) / len(values)
                    if dim == "imaging_quality":
                        aggregate /= 100
                    self.write_json(self.score_path(seed, config, dim), {dim: [aggregate, values]})
                gt = {case_id: {"dataset": "fixture", "dreamsim": 0.2 + delta / 10,
                                "dreamsim_last": 0.25 + delta / 10, "motion_ratio": 0.4 + delta,
                                "dur": 5.0} for case_id in self.ids}
                self.write_json(root / "gtref" / f"{config}.json", gt)

    def write_json(self, path, payload):
        path.write_text(json.dumps(payload), encoding="utf-8")

    def write_cases(self, root, cases):
        (root / "cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases), encoding="utf-8")

    def score_path(self, seed="seed3", config="s12_c5", dim="aesthetic_quality"):
        return self.roots[seed] / "scores" / f"{config}__{dim}_eval_results.json"

    def run_summary(self):
        return summary.summarize(self.roots, bootstrap_samples=80)

    def test_complete_paired_means_guards_and_cross_seed_sample_std(self):
        result = self.run_summary()
        self.assertTrue(result["complete"])
        self.assertEqual((result["n_groups"], result["n_cases_per_group"], result["n_seeds"]), (8, 94, 4))
        self.assertEqual(set(result["cross_seed_delta"]), set(summary.METRICS))
        for seed_index, seed in enumerate(self.roots):
            entry = result["seeds"][seed]
            self.assertEqual(set(entry["means"]), set(summary.CONFIGS))
            for config in summary.CONFIGS:
                self.assertEqual(entry["means"][config]["n"], 94)
            paired = entry["paired_delta"]["imaging_quality"]
            delta = self.shifts[seed_index]
            self.assertEqual(paired["n"], 94)
            self.assertAlmostEqual(paired["mean"], delta, places=12)
            for boundary in paired["ci95"]:
                self.assertAlmostEqual(boundary, delta, places=12)
            self.assertEqual(paired["better_cases"], 94 if delta > 0 else 0)
            self.assertEqual(set(entry["guard_checks"]), {"motion_smoothness", "subject_consistency", "dreamsim", "motion_ratio"})
            expected = {"motion_smoothness": delta >= -0.003, "subject_consistency": delta >= -0.01,
                        "dreamsim": delta / 10 <= 0.005, "motion_ratio": delta <= 0.05}
            self.assertEqual(entry["guard_checks"], expected)
            self.assertEqual(entry["all_guards_pass"], all(expected.values()))
        mean = sum(self.shifts) / 4
        independent_std = math.sqrt(sum((d - mean) ** 2 for d in self.shifts) / 3)
        actual = result["cross_seed_delta"]["imaging_quality"]
        self.assertAlmostEqual(actual["mean"], mean, places=12)
        self.assertAlmostEqual(actual["sample_std"], independent_std, places=12)
        for got, expected in zip(actual["range"], [-0.03, 0.04]):
            self.assertAlmostEqual(got, expected, places=12)
        self.assertNotAlmostEqual(actual["sample_std"], math.sqrt(sum((d - mean) ** 2 for d in self.shifts) / 4), places=6)

    def test_historical_three_seed_dynamic_std_is_2_214538_not_3_29_pp(self):
        deltas_pp = [2.127659574468077, -2.127659574468077, 1.0638297872340385]
        mean = sum(deltas_pp) / 3
        independent_std = math.sqrt(sum((d - mean) ** 2 for d in deltas_pp) / 2)
        result = summary.cross_seed_stats(deltas_pp)
        self.assertAlmostEqual(independent_std, 2.2145382973043883, places=12)
        self.assertAlmostEqual(result["sample_std"], independent_std, places=12)
        self.assertNotAlmostEqual(result["sample_std"], 3.29, places=2)
        actual = self.run_summary()
        for seed, expected in zip(("seed0", "seed1", "seed2"), deltas_pp):
            self.assertAlmostEqual(actual["seeds"][seed]["paired_delta"]["dynamic_degree"]["mean"] * 100, expected, places=11)

    def test_summary_contains_no_case_metadata_or_selection(self):
        result = self.run_summary()
        text = json.dumps(result)
        for private in ("private_case", "private prompt", "/private/", "private-video-hash", "private-parent"):
            self.assertNotIn(private, text)
        def check_keys(value):
            if isinstance(value, dict):
                self.assertFalse({"id", "prompt", "image", "gt", "video_sha256", "parent_keys", "selected", "selection"} & set(value))
                for child in value.values():
                    check_keys(child)
            elif isinstance(value, list):
                for child in value:
                    check_keys(child)
        check_keys(result)
        self.assertIn("descriptive", result["methodology"]["paired_ci"])
        self.assertIn("not a strict population-significance", result["methodology"]["inference"])

    def test_missing_score_refuses_output(self):
        self.score_path().unlink()
        with self.assertRaisesRegex(RuntimeError, "seed3: s12_c5/aesthetic_quality: missing score"):
            summary.main(["--root", str(self.root)])
        self.assertFalse((self.root / summary.OUTPUT).exists())
        self.assertFalse((self.root / summary.OUTPUT).with_suffix(".json.tmp").exists())

    def test_incomplete_duplicate_and_extra_score_coverage_rejected(self):
        path = self.score_path()
        original = json.loads(path.read_text())
        dim = "aesthetic_quality"
        for change in ("missing", "duplicate", "extra"):
            with self.subTest(change=change):
                data = json.loads(json.dumps(original))
                if change == "missing":
                    data[dim][1].pop()
                elif change == "duplicate":
                    data[dim][1][-1] = dict(data[dim][1][0])
                else:
                    data[dim][1].append({"video_path": "/private/extra-0.mp4", "video_results": 0.5})
                self.write_json(path, data)
                with self.assertRaisesRegex(RuntimeError, "coverage"):
                    self.run_summary()

    def test_missing_and_nonfinite_gt_rejected(self):
        path = self.roots["seed3"] / "gtref/s12_c5.json"
        original = json.loads(path.read_text())
        for change in ("missing", "missing_metric", float("nan"), float("inf")):
            with self.subTest(change=change):
                data = json.loads(json.dumps(original))
                if change == "missing":
                    data.pop(self.ids[0])
                elif change == "missing_metric":
                    data[self.ids[0]].pop("dreamsim")
                else:
                    data[self.ids[0]]["dreamsim"] = change
                self.write_json(path, data)
                with self.assertRaisesRegex(RuntimeError, "GT-reference"):
                    self.run_summary()

    def test_nonfinite_vbench_per_video_and_aggregate_rejected(self):
        path = self.score_path()
        original = json.loads(path.read_text())
        for value in (float("nan"), float("inf")):
            for location in ("video", "aggregate"):
                with self.subTest(value=value, location=location):
                    data = json.loads(json.dumps(original))
                    if location == "video":
                        data["aesthetic_quality"][1][0]["video_results"] = value
                    else:
                        data["aesthetic_quality"][0] = value
                    self.write_json(path, data)
                    with self.assertRaises(RuntimeError):
                        self.run_summary()

    def test_changed_case_metadata_rejected_before_output(self):
        for field, value in (("prompt", "changed"), ("dataset", "other"), ("image", "changed"),
                             ("video_sha256", "changed"), ("gt", "changed"), ("parent_keys", ["other"])):
            with self.subTest(field=field):
                cases = [dict(c) for c in self.cases]
                cases[0][field] = value
                self.write_cases(self.roots["seed3"], cases)
                with self.assertRaisesRegex(ValueError, "case metadata differ"):
                    summary.main(["--root", str(self.root)])
                self.assertFalse((self.root / summary.OUTPUT).exists())

    def test_metadata_row_order_does_not_break_pairing(self):
        self.write_cases(self.roots["seed3"], list(reversed(self.cases)))
        result = self.run_summary()
        self.assertAlmostEqual(result["seeds"]["seed3"]["paired_delta"]["imaging_quality"]["mean"], 0.04)

    def test_missing_duplicate_nonfinite_case_metadata_rejected(self):
        bad_cases = [self.cases[:-1], self.cases[:-1] + [self.cases[0]]]
        for field in ("id", "dataset"):
            cases = [dict(c) for c in self.cases]
            cases[0].pop(field)
            bad_cases.append(cases)
        cases = [dict(c) for c in self.cases]
        cases[0]["fps"] = float("nan")
        bad_cases.append(cases)
        for cases in bad_cases:
            with self.subTest(first=cases[0]):
                self.write_cases(self.roots["seed3"], cases)
                with self.assertRaises(ValueError):
                    self.run_summary()

    def test_exactly_four_roots_required(self):
        for roots in ({k: v for k, v in self.roots.items() if k != "seed3"}, dict(self.roots, seed4=self.root)):
            with self.assertRaisesRegex(ValueError, "Exactly"):
                summary.summarize(roots)

    def test_main_supports_project_root_and_testable_root_mapping(self):
        for roots in (None, self.roots):
            with self.subTest(mapped=roots is not None):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = summary.main(["--root", str(self.root)], roots=roots)
                output = self.root / summary.OUTPUT
                self.assertEqual(json.loads(output.read_text()), result)
                self.assertTrue(result["complete"])
                self.assertFalse(output.with_name(output.name + ".tmp").exists())

    def test_no_partial_flag_is_supported(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            summary.main(["--partial", "--root", str(self.root)])
        self.assertEqual(error.exception.code, 2)
        self.assertFalse((self.root / summary.OUTPUT).exists())


if __name__ == "__main__":
    unittest.main()
