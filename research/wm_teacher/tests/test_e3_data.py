import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

# Public snapshot guard: these tests need private helpers (prepare_e1_devset) not in the snapshot.
_HAS_PRIVATE_DEPS = (SCRIPTS / "prepare_e1_devset.py").is_file() and (SCRIPTS / "score_e3_motion.py").is_file()

import unittest as _u
if not _HAS_PRIVATE_DEPS:
    raise _u.SkipTest('e3_data tests require private helpers (prepare_e1_devset/score_e3_motion) absent from the public snapshot')


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pool = _load("build_e3_pool")
mot = _load("score_e3_motion")
P = "dataset/wan22_5s/"


def good_indomain(**kw):
    s = {"n_frames": 121, "scene_cut": False, "diff_median": 2.0, "aes": 6.0, "cam_shift": 1.0, "cuts": 0, "flicker": 0.1}
    s.update(kw)
    return s


def good_ov(**kw):
    s = {"n_frames": 121, "scene_cut": False, "cam_shift": 1.0, "cuts": 0, "flicker": 0.1, "ov_aes": 7.0, "ov_motion": 5.0}
    s.update(kw)
    return s


Q = {"flicker_q80": 1.0, "ov_motion_q40": 2.0}


def fails(ds, s, q=Q):
    ok, out, first = pool.evaluate(ds, s, q)
    return sorted(r for r, o in out.items() if o != "pass") if out else [first]


class RuleTests(unittest.TestCase):
    def test_indomain_inclusive_boundaries(self):
        self.assertEqual(fails("noxi", good_indomain(aes=5.8, diff_median=1.5, cam_shift=8.0, flicker=1.0)), [])
        self.assertEqual(fails("noxi", good_indomain(aes=5.7999)), ["aes"])
        self.assertEqual(fails("noxi", good_indomain(diff_median=1.4999)), ["diff_median"])
        self.assertEqual(fails("noxi", good_indomain(cam_shift=8.0001)), ["cam_shift"])
        self.assertEqual(fails("noxi", good_indomain(flicker=1.0001)), ["flicker"])
        self.assertEqual(fails("noxi", good_indomain(cuts=1)), ["cuts"])
        self.assertEqual(fails("noxi", good_indomain(scene_cut=True)), ["scene_cut"])
        self.assertEqual(fails("noxi", good_indomain(n_frames=120)), ["n_frames"])

    def test_openvid_uses_openvid_thresholds_only(self):
        self.assertEqual(fails("openvidgen", good_ov(ov_aes=6.3, ov_motion=2.0)), [])
        self.assertEqual(fails("openvidgen", good_ov(ov_aes=6.2999)), ["ov_aes"])
        self.assertEqual(fails("openvidgen", good_ov(ov_motion=1.9999)), ["ov_motion"])
        self.assertEqual(fails("openvidgen", good_ov(ov_aes=5.9)), ["ov_aes"])  # in-domain 5.8 not used
        ok, out, _ = pool.evaluate("openvidgen", good_indomain(), Q)  # in-domain aes does not substitute
        self.assertFalse(ok)
        self.assertEqual((out["ov_aes"], out["ov_motion"]), ("missing", "missing"))

    def test_missing_signal_never_passes(self):
        for k in ("aes", "cam_shift", "cuts", "diff_median", "scene_cut"):
            s = good_indomain()
            s[k] = None
            ok, out, first = pool.evaluate("noxi", s, Q)
            self.assertFalse(ok)
            self.assertEqual(out[k], "missing")
            self.assertTrue(first.startswith("missing:"))
        ok, out, _ = pool.evaluate("noxi", good_indomain(), {})  # no per-source flicker threshold
        self.assertEqual((ok, out["flicker"]), (False, "missing"))

    def test_mira_excluded(self):
        ok, out, first = pool.evaluate("mira", good_indomain(aes=9.0), Q)
        self.assertEqual((ok, out, first), (False, {}, "excluded_dataset"))

    def test_quantile_matches_numpy_linear(self):
        import numpy as np
        xs = [0.3, 1.2, 0.7, 5.0, 2.2, 0.0, 0.9]
        for q in (0.0, 0.4, 0.8, 1.0):
            self.assertAlmostEqual(pool.quantile(xs, q), float(np.quantile(xs, q)))
        self.assertIsNone(pool.quantile([], 0.5))

    def test_source_thresholds_are_per_dataset(self):
        rows = [{"video": f"a{i}", "dataset": "noxi"} for i in range(10)] + \
               [{"video": f"b{i}", "dataset": "seamless"} for i in range(10)]
        sig = {f"a{i}": {"flicker": float(i), "ov_motion": None} for i in range(10)}
        sig.update({f"b{i}": {"flicker": float(i * 10), "ov_motion": None} for i in range(10)})
        q = pool.source_thresholds(rows, sig)
        self.assertAlmostEqual(q["noxi"]["flicker_q80"], 7.2)
        self.assertAlmostEqual(q["seamless"]["flicker_q80"], 72.0)
        self.assertIsNone(q["noxi"]["ov_motion_q40"])

    def test_openvid_name_mapping(self):
        self.assertEqual(pool.openvid_name("dataset/wan22_5s_extra/openvidgen_5s/OpenVidHD_part_3.zip_abc_12_0to163_w000.mp4"),
                         "abc_12_0to163.mp4")


class CapTests(unittest.TestCase):
    def rows(self, spec):
        return [{"video": v, "dataset": ds, "parent_keys": set(k), "aes": a} for v, ds, k, a in spec]

    def test_per_parent_cap_shared_keys_and_priority(self):
        spec = [(f"v{i}", "celebvhq", ["youtube:A"], 6.0 + i / 10) for i in range(5)]
        spec += [("x", "easycom", ["easycom:S1", "easycom:S2"], 7.0), ("y", "easycom", ["easycom:S2"], 6.5),
                 ("z", "easycom", ["easycom:S2"], 6.4)]
        kept, cap = pool.apply_caps(self.rows(spec), 2, 1.0)
        vids = {r["video"] for r in kept}
        self.assertEqual({v for v in vids if v.startswith("v")}, {"v4", "v3"})  # highest aes kept
        self.assertEqual(vids - {"v4", "v3"}, {"x", "y"})
        self.assertEqual(cap, {"parent_cap": 4, "matrix_cap": 0})
        kept2, _ = pool.apply_caps(list(reversed(self.rows(spec))), 2, 1.0)
        self.assertEqual({r["video"] for r in kept2}, vids)  # input-order independent

    def test_matrix_share_at_most_ten_percent(self):
        spec = [(f"o{i}", "noxi", [f"noxi:{i}"], 6.0) for i in range(45)]
        spec += [(f"m{i}", "matrix", [f"matrix:{i}"], 6.0 + i / 100) for i in range(20)]
        kept, cap = pool.apply_caps(self.rows(spec), 2, 0.10)
        m = [r for r in kept if r["dataset"] == "matrix"]
        self.assertEqual((len(m), cap["matrix_cap"]), (5, 15))
        self.assertLessEqual(len(m) / len(kept), 0.10)
        self.assertEqual({r["video"] for r in m}, {f"m{i}" for i in range(15, 20)})  # best-aes matrix kept


class RaftParamTests(unittest.TestCase):
    def test_vbench_params_full_res(self):
        n = mot.n_sampled_frames(121, 24.0)
        self.assertEqual((mot.frame_interval(24.0), n), (3, 41))
        p = mot.vbench_params(704, 1280, n)
        self.assertAlmostEqual(p["thres"], 16.5)
        self.assertEqual(p["count_num"], 10)
        self.assertEqual(mot.vbench_params(256, 256, 16), {"thres": 6.0, "count_num": 4})

    def test_threshold_scales_with_resolution(self):
        full, half = mot.vbench_params(704, 1280, 41), mot.vbench_params(352, 640, 41)
        self.assertAlmostEqual(half["thres"], 0.5 * full["thres"])
        self.assertEqual(half["count_num"], full["count_num"])  # count does not depend on scale

    def test_check_move_and_aggregate(self):
        self.assertTrue(mot.check_move([7, 0, 7], 6.0, 2))
        self.assertFalse(mot.check_move([6.0, 6.0, 7], 6.0, 2))  # strict '>' as in VBench
        self.assertEqual(mot.count_over([6.0, 6.1, 9], 6.0), 2)
        r = mot.aggregate([20.0] * 10 + [1.0] * 30, [1.0] * 40, [2.0] * 40, 704, 1280, 1.0)
        self.assertTrue(r["dynamic"])
        self.assertEqual((r["n_over"], r["n_pairs"], r["n_sampled"], r["count_num"]), (10, 40, 41, 10))
        self.assertAlmostEqual(r["top5_mean"], 5.75)
        r = mot.aggregate([20.0] * 9 + [1.0] * 31, [0.0] * 40, [0.0] * 40, 704, 1280, 1.0)
        self.assertFalse(r["dynamic"])

    def test_flow_stats_separates_camera_and_residual(self):
        import numpy as np
        u = np.full((20, 20), 3.0); v = np.zeros((20, 20))
        top, cam, resid = mot.flow_stats(u, v)
        self.assertAlmostEqual((top, cam, resid), (3.0, 3.0, 0.0))
        u[:2, :] = 13.0  # 10% of pixels move extra 10 px on top of global pan
        top, cam, resid = mot.flow_stats(u, v)
        self.assertAlmostEqual(cam, 3.0)
        self.assertAlmostEqual(resid, 10.0)

    def test_shards_partition_and_resume(self):
        vids = [f"v{i}" for i in range(103)]
        parts = [mot.shard(vids, i, 32) for i in range(32)]
        self.assertEqual(sorted(sum(parts, [])), sorted(vids))
        with self.assertRaises(ValueError):
            mot.shard(vids, 32, 32)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.jsonl")
            with open(p, "w") as f:
                f.write(json.dumps({"video": "a", "dynamic": True}) + "\n")
                f.write(json.dumps({"video": "b", "error": "x"}) + "\n")
                f.write('{"video": "c", "dyn')
            self.assertEqual(set(mot.load_records([p])), {"a"})


if __name__ == "__main__":
    unittest.main()
