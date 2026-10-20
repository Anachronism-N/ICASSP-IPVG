import importlib.util
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("score_e1", SCRIPTS / "score_e1.py")
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)
spec2 = importlib.util.spec_from_file_location("summarize_eval_ref", SCRIPTS / "summarize_eval.py")
se = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(se)

BASE = {"total": 0.80, "motion_smoothness": 0.980, "subject_consistency": 0.940, "dreamsim": 0.200, "motion_ratio": 0.400}


def cfg(**kw):
    return dict(BASE, **kw)


class SelectionRuleTests(unittest.TestCase):
    def test_constants_match_docs43(self):
        self.assertEqual(sc.RULE, {"motion_smoothness_min_delta": -0.003, "subject_consistency_min_delta": -0.010,
                                   "dreamsim_max_delta": 0.005, "motion_ratio_max_delta": 0.05})
        self.assertEqual(sc.DEFAULT_CONFIG, "s5_c5")
        self.assertEqual(len(sc.CONFIGS), 12)

    def test_exact_boundaries_pass(self):
        s = {"s5_c5": cfg(), "s8_c5": cfg(total=0.81, motion_smoothness=0.977, subject_consistency=0.930,
                                          dreamsim=0.205, motion_ratio=0.45)}
        r = sc.select_config(s)
        self.assertEqual(r["selected"], "s8_c5")
        self.assertTrue(all(r["checks"]["s8_c5"].values()))

    def test_each_guard_just_beyond_fails(self):
        for k, v in (("motion_smoothness", 0.9769), ("subject_consistency", 0.9299), ("dreamsim", 0.2051),
                     ("motion_ratio", 0.4501)):
            s = {"s5_c5": cfg(), "s12_c6": cfg(total=0.90, **{k: v})}
            r = sc.select_config(s)
            self.assertEqual(r["selected"], "s5_c5", k)
            self.assertFalse(r["checks"]["s12_c6"][k], k)

    def test_highest_total_among_passing(self):
        s = {"s5_c5": cfg(), "s3_c4": cfg(total=0.95, dreamsim=0.3), "s8_c5": cfg(total=0.82), "s8_c6": cfg(total=0.83)}
        self.assertEqual(sc.select_config(s)["selected"], "s8_c6")

    def test_ties_prefer_default_then_closest(self):
        self.assertEqual(sc.select_config({"s5_c5": cfg(), "s8_c5": cfg()})["selected"], "s5_c5")
        s = {"s5_c5": cfg(total=0.7), "s12_c6": cfg(total=0.9), "s8_c5": cfg(total=0.9)}
        self.assertEqual(sc.select_config(s)["selected"], "s8_c5")

    def test_default_lower_total_but_only_passing_and_missing_default(self):
        s = {"s5_c5": cfg(), "s8_c5": cfg(total=0.9, dreamsim=0.3)}
        r = sc.select_config(s)
        self.assertEqual((r["selected"], r["passing"]), ("s5_c5", ["s5_c5"]))
        with self.assertRaises(ValueError):
            sc.select_config({"s8_c5": cfg()})


def write_scores(root, config, ids, values, dataset="noxi", gt=None, aggregate_shift=0.0):
    os.makedirs(f"{root}/scores", exist_ok=True)
    os.makedirs(f"{root}/gtref", exist_ok=True)
    for dim in sc.DIMS:
        per = [{"video_path": f"/w/{i}-0.mp4", "video_results": values[i][dim]} for i in ids]
        agg = sum(float(v["video_results"]) for v in per) / len(per)
        if dim == "imaging_quality":
            agg /= 100
        json.dump({dim: [agg + aggregate_shift, per]}, open(f"{root}/scores/{config}__{dim}_eval_results.json", "w"))
    gt = gt or {i: {"dataset": dataset, "dreamsim": 0.2, "dreamsim_last": 0.25, "motion_ratio": 0.4, "dur": 5.0} for i in ids}
    json.dump(gt, open(f"{root}/gtref/{config}.json", "w"))


def vals(k):
    return {"i2v_subject": 0.97 + k, "i2v_background": 0.98, "subject_consistency": 0.93, "background_consistency": 0.94,
            "motion_smoothness": 0.98, "dynamic_degree": bool(k > 0), "aesthetic_quality": 0.5 + k,
            "imaging_quality": 64.0 + 100 * k}


class AggregationTests(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.root = t.name
        self.ids = ["a", "b", "c"]
        self.cases = [{"id": i, "dataset": "noxi"} for i in self.ids]
        self.values = {"a": vals(0.0), "b": vals(0.01), "c": vals(0.02)}

    def test_rows_scale_and_total_matches_summarize_eval(self):
        write_scores(self.root, "s5_c5", self.ids, self.values)
        rows = sc.load_config_rows("s5_c5", self.cases, self.root)
        self.assertAlmostEqual(rows["b"]["imaging_quality"], 0.65)
        self.assertEqual(rows["a"]["dynamic_degree"], 0.0)
        m = sc.mean_stats(rows)
        means = {d: sum(rows[i][d] for i in self.ids) / 3 for d in sc.DIMS}
        self.assertAlmostEqual(m["total"], se.total(means)[0], places=12)
        self.assertAlmostEqual(m["total"], sum(rows[i]["total"] for i in self.ids) / 3, places=12)  # linear
        # independent hand computation of Total* (camera_motion excluded, dynamic weight 0.5)
        n = lambda d, lo, hi: (means[d] - lo) / (hi - lo)
        q = (n("subject_consistency", .1462, 1) + n("background_consistency", .2615, 1)
             + n("motion_smoothness", .7060, .9975) + 0.5 * means["dynamic_degree"] + means["aesthetic_quality"]
             + means["imaging_quality"]) / 5.5
        i2v = (n("i2v_subject", .1462, 1) + n("i2v_background", .2615, 1)) / 2
        self.assertAlmostEqual(m["total"], (q + i2v) / 2, places=12)

    def test_coverage_errors(self):
        write_scores(self.root, "s5_c5", self.ids[:2], self.values)
        with self.assertRaises(RuntimeError):
            sc.load_config_rows("s5_c5", self.cases, self.root)
        write_scores(self.root, "s5_c5", self.ids, self.values, aggregate_shift=0.01)
        with self.assertRaises(RuntimeError):
            sc.load_config_rows("s5_c5", self.cases, self.root)
        bad = {i: {"dreamsim": float("nan"), "dreamsim_last": 0, "motion_ratio": 0, "dur": 5} for i in self.ids}
        write_scores(self.root, "s5_c5", self.ids, self.values, gt=bad)
        with self.assertRaises(RuntimeError):
            sc.load_config_rows("s5_c5", self.cases, self.root)

    def test_paired_deterministic(self):
        write_scores(self.root, "s5_c5", self.ids, self.values)
        base = sc.load_config_rows("s5_c5", self.cases, self.root)
        other = {i: dict(r, total=r["total"] + 0.01, dreamsim=r["dreamsim"] - 0.02) for i, r in base.items()}
        p1 = sc.paired(other, base, ["total", "dreamsim"], b=200)
        p2 = sc.paired(other, base, ["total", "dreamsim"], b=200)
        self.assertEqual(p1, p2)
        self.assertAlmostEqual(p1["total"]["mean"], 0.01)
        self.assertEqual(p1["dreamsim"]["better_cases"], 3)

    def test_summarize_full_grid_with_egoexo4d_sensitivity(self):
        ids = ["a", "b", "c", "e"]
        with open(f"{self.root}/cases.jsonl", "w") as f:
            for i in ids:
                f.write(json.dumps({"id": i, "dataset": "egoexo4d" if i == "e" else "noxi"}) + "\n")
        values = dict(self.values, e=vals(0.03))
        for k, name in enumerate(sc.CONFIGS):
            gt = {i: {"dataset": "x", "dreamsim": 0.2, "dreamsim_last": 0.25, "motion_ratio": 0.4, "dur": 5.0} for i in ids}
            write_scores(self.root, name, ids, values, gt=gt)
        sc.cmd_summarize(False, self.root)
        out = json.load(open(f"{self.root}/e1_summary.json"))
        self.assertEqual(out["selection"]["selected"], "s5_c5")  # all identical -> tie -> default
        e = out["sensitivity_excluding"]["egoexo4d"]
        self.assertEqual(e["n_cases"], 3)
        self.assertEqual(e["selection"]["selected"], "s5_c5")
        self.assertLess(e["means"]["s5_c5"]["aesthetic_quality"], out["means"]["s5_c5"]["aesthetic_quality"])
        self.assertEqual(set(out["per_dataset"]["s5_c5"]), {"noxi", "egoexo4d"})

    def test_real_vbench_files_per_video_mean_equals_aggregate(self):
        d = Path(sc.ROOT) / "eval/wm_bench/scores"
        if not (d / "base__imaging_quality_eval_results.json").exists():
            self.skipTest("no local VBench results")
        for dim in sc.DIMS:
            r = json.load(open(d / f"base__{dim}_eval_results.json"))[dim]
            mean = sum(sc.per_video_value(dim, v["video_results"]) for v in r[1]) / len(r[1])
            self.assertTrue(math.isclose(mean, r[0], rel_tol=1e-9), dim)


if __name__ == "__main__":
    unittest.main()
