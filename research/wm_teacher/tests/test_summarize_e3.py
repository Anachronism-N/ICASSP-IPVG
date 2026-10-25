import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("score_e1", SCRIPTS / "score_e1.py")
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)
spec3 = importlib.util.spec_from_file_location("summarize_e3", SCRIPTS / "summarize_e3.py")
s3 = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(s3)

from test_e1_score import vals, write_scores  # noqa: E402

IDS = [f"c{i:02d}" for i in range(6)]
CASES = [{"id": i, "dataset": "noxi" if i[1] < "5" else "egoexo4d", "gt": f"/g/{i}.mp4",
          "image": f"/img/{i}.png", "prompt": "p"} for i in IDS]


def arm_vals(motion_smoothness=0.9800, subject_consistency=0.9400, dynamic=1):
    def v(k):
        d = vals(0.01)
        d["motion_smoothness"] = motion_smoothness
        d["subject_consistency"] = subject_consistency
        d["dynamic_degree"] = bool(dynamic)
        return d
    return {i: v(i) for i in IDS}


def gtref(dreamsim=0.2000, motion=0.4000):
    return {i: {"dataset": "noxi", "dreamsim": dreamsim + (0.001 if i >= "c05" else 0),
                "dreamsim_last": 0.25, "motion_ratio": motion + (0.001 if i >= "c05" else 0), "dur": 5.0}
            for i in IDS}


class E3SummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.e3, self.base = root / "e3", root / "e1"
        for d, name in ((self.e3, "e3_devset_eval"), (self.base, "e1_devset")):
            d.mkdir()
            with open(d / "cases.jsonl", "w") as f:
                for c in CASES:
                    f.write(json.dumps(c) + "\n")
        self._paths = s3.E3, s3.BASE_ROOT
        s3.E3, s3.BASE_ROOT = str(self.e3), str(self.base)

    def tearDown(self):
        s3.E3, s3.BASE_ROOT = self._paths
        self.tmp.cleanup()

    def test_constants_match_docs43_47(self):
        self.assertEqual(s3.E3_ARMS, ["armA_orig", "armB_fixed", "armC_fixed_anchor"])
        self.assertEqual(s3.BASE_CONFIG, "s5_c5")
        self.assertEqual(s3.RULE, {"motion_smoothness_min_delta": -0.003, "subject_consistency_min_delta": -0.010,
                                   "dreamsim_max_delta": 0.005, "motion_ratio_max_delta": 0.05})

    def test_passing_arm_with_highest_total_selected(self):
        write_scores(str(self.e3), "armA_orig", IDS, arm_vals(0.9800, 0.9400, dynamic=0), gt=gtref())
        write_scores(str(self.e3), "armB_fixed", IDS, arm_vals(0.9800, 0.9400, dynamic=1), gt=gtref())
        write_scores(str(self.e3), "armC_fixed_anchor", IDS, arm_vals(0.9770, 0.9300, dynamic=1), gt=gtref(0.205, 0.45))
        write_scores(str(self.base), "s5_c5", IDS, arm_vals(0.9800, 0.9400), gt=gtref())
        s3.main([])
        j = json.load(open(self.e3 / "e3_summary.json"))["judgment"]
        # C sits exactly at the guard boundaries, A/B equal to base -> all pass; B total beats A via dynamic
        self.assertEqual(j["passing_arms"], ["armA_orig", "armB_fixed", "armC_fixed_anchor"])
        self.assertEqual(j["selected"], "armB_fixed")
        self.assertFalse(j["negative_result"])

    def test_all_fail_gives_negative_result(self):
        # A fails motion_smoothness (-0.004 < -0.003); B fails subject_consistency (-0.011 < -0.010);
        # C fails dreamsim (+0.006 > +0.005).
        write_scores(str(self.e3), "armA_orig", IDS, arm_vals(0.9760, 0.9400), gt=gtref(0.2002))
        write_scores(str(self.e3), "armB_fixed", IDS, arm_vals(0.9800, 0.9290), gt=gtref(0.2002))
        write_scores(str(self.e3), "armC_fixed_anchor", IDS, arm_vals(0.9800, 0.9400), gt=gtref(0.206))
        write_scores(str(self.base), "s5_c5", IDS, arm_vals(0.9800, 0.9400), gt=gtref())
        s3.main([])
        j = json.load(open(self.e3 / "e3_summary.json"))["judgment"]
        self.assertEqual(j["passing_arms"], [])
        self.assertIsNone(j["selected"])
        self.assertTrue(j["negative_result"])
        self.assertIn("no full-benchmark run", j["next_step"])

    def test_changed_metadata_and_duplicate_cases_rejected(self):
        with open(self.e3 / "cases.jsonl", "w") as f:
            for c in CASES:
                f.write(json.dumps(dict(c, prompt="changed")) + "\n")
        with self.assertRaisesRegex(ValueError, "metadata"):
            s3.main(["--partial"])
        for path in (self.e3, self.base):
            with open(path / "cases.jsonl", "w") as f:
                for c in CASES + [CASES[0]]:
                    f.write(json.dumps(c) + "\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            s3.main(["--partial"])

    def test_partial_reports_base_and_missing_arms(self):
        write_scores(str(self.base), "s5_c5", IDS, arm_vals(), gt=gtref())
        s3.main(["--partial"])
        summary = json.load(open(self.e3 / "e3_summary_partial.json"))
        self.assertEqual(set(summary["missing"]), set(s3.E3_ARMS))
        self.assertNotIn("judgment", summary)
        self.assertIn("s5_c5", summary["means"])

    def test_paired_vs_base_uses_same_cases(self):
        write_scores(str(self.e3), "armA_orig", IDS, arm_vals(0.9800, 0.9350), gt=gtref(0.19))
        write_scores(str(self.e3), "armB_fixed", IDS, arm_vals(), gt=gtref())
        write_scores(str(self.e3), "armC_fixed_anchor", IDS, arm_vals(), gt=gtref())
        write_scores(str(self.base), "s5_c5", IDS, arm_vals(), gt=gtref())
        s3.main([])
        p = json.load(open(self.e3 / "e3_summary.json"))["paired_vs_base"]
        self.assertEqual(p["armA_orig"]["subject_consistency"]["n"], len(IDS))
        self.assertAlmostEqual(p["armA_orig"]["dreamsim"]["mean"], -0.01, places=6)


if __name__ == "__main__":
    unittest.main()
