import csv
import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("prepare_e1_devset", SCRIPTS / "prepare_e1_devset.py")
dev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dev)

P = "dataset/wan22_5s/"


def keys(rel, easycom=None):
    return dev.parent_keys(P + rel, easycom)


class ParentKeyTests(unittest.TestCase):
    def test_youtube_namespace_shared_across_datasets(self):
        k = {keys("celebvhq_5s/60/EBdYT9I7YhI_w080.mp4")[0], keys("openvidgen_5s/EBdYT9I7YhI_35_0to168.mp4")[0],
             keys("vfhq_5s/e0/Clip+EBdYT9I7YhI+P0+C0+F159135-159453_w001.mp4")[0],
             keys("hoigen1m_5s/3f/EBdYT9I7YhI-0_10_06.572-0_10_26.392_w000.mp4")[0],
             keys("openvidhd_5s/89/OpenVidHD_part_13.zip_EBdYT9I7YhI_3_0to191_w000.mp4")[0]}
        self.assertEqual(k, {frozenset({"youtube:EBdYT9I7YhI"})})
        self.assertEqual(keys("openvidgen_5s/Gv_cqCBJ1Bw_2_0to163.mp4")[0], {"youtube:Gv_cqCBJ1Bw"})

    def test_windows_cameras_and_views_merge(self):
        self.assertEqual(keys("robomind_5s/ur_1rgb/camera_top_ep000229_w0.mp4")[0],
                         keys("robomind_5s/ur_1rgb/camera_front_ep000229_w1.mp4")[0])
        self.assertNotEqual(keys("robomind_5s/ur_1rgb/camera_top_ep000229_w0.mp4")[0],
                            keys("robomind_5s/franka_1rgb/camera_top_ep000229_w0.mp4")[0])
        self.assertEqual(keys("noxi_5s/a9/073_novice_w110.mp4")[0], keys("noxi_5s/073_expert_w001.mp4")[0])
        self.assertEqual(keys("ukdash_5s/241220_134704_020_RH_w003.mp4")[0], {"ukdash:day:241220"})
        self.assertEqual(keys("matrix_5s/1e/part_100_2024-06-05_20-54-23_000021_w000.mp4")[0],
                         keys("matrix_5s/1f/part_7_2024-06-05_20-54-23_000003_w001.mp4")[0])
        self.assertEqual(keys("h2o_5s/1b/subject1_ego_h2_3_cam4_w003.mp4")[0], keys("h2o_5s/09/subject1_h2_5_cam4_w001.mp4")[0])
        self.assertEqual(keys("egodex_5s/train_push_pop_toy-part2/camera_ep000072_w1.mp4")[0],
                         {"egodex:train_push_pop_toy:ep000072"})
        self.assertEqual(keys("direct_5s/01/AriaEverydayActivities_1.0.0_loc2_script5_seq1_rec1_preview_rgb_w021.mp4")[0],
                         keys("direct_5s/02/AriaEverydayActivities_1.0.0_loc2_script5_seq1_rec2_preview_rgb_w003.mp4")[0])
        self.assertEqual(keys("direct_5s/39/ADT_Apartment_release_multiskeleton_party_seq115_71292_preview_rgb_w018.mp4")[0],
                         {"aria:ADT:seq115"})
        self.assertEqual(keys("direct_5s/d9/Hot3DAria_v4.0.0_P0004_abaa9885_preview_rgb_w009.mp4")[0],
                         keys("direct_5s/cf/Hot3DQuest_v4.0.0_P0004_0ec32d10_preview_rgb_w009.mp4")[0])
        self.assertEqual(keys("seamless_5s/7b/V01_S0172_I00001228_P1316_w020.mp4")[0], {"seamless:session_token:V01:S0172"})
        self.assertEqual(keys("mira_5s/b2/dataset_wan22_training_mira_processed_2026-05-13T22-52-32Z-b0d167_p2_c00000_w002.mp4")[0],
                         keys("mira_5s/b3/dataset_wan22_training_mira_processed_2026-05-13T22-52-32Z-b0d167_p0_c00000_w9.mp4")[0])
        self.assertEqual(keys("charades_5s/db/WM336_1280x704_24fps_w003.mp4")[0], {"charades:video:WM336"})

    def test_easycom_uses_all_sessions_and_egoexo_unresolved(self):
        ec = {"07-00-382": {"Session_1", "Session_4"}}
        self.assertEqual(keys("easycom_5s/ed/07-00-382_1280x704_24fps_w004.mp4", ec),
                         (frozenset({"easycom:Session_1", "easycom:Session_4"}), True))
        self.assertFalse(keys("easycom_5s/ed/09-00-001_1280x704_24fps_w004.mp4", ec)[1])
        k, resolved = keys("egoexo4d_5s/ee/cam05_w050.mp4")
        self.assertFalse(resolved)
        self.assertEqual(k, keys("egoexo4d_5s/aa/cam05_w001.mp4")[0])

    def test_extract_paths_normalises_absolute_and_relative(self):
        text = json.dumps({"a": "/x/wm_dataset/dataset/wan22_5s/noxi_5s/073_novice_w110.mp4",
                           "b": ["dataset/wan22_5s/vfhq_5s/e0/Clip+u+P0+C0+F1-2_w001.mp4"]}) + "\nvideo,prompt\n" \
            "dataset/wan22_5s/charades_5s/db/WM336_1280x704_24fps_w003.mp4,\"text\"\n"
        self.assertEqual(dev.extract_paths(text), {P + "noxi_5s/073_novice_w110.mp4",
                                                   P + "vfhq_5s/e0/Clip+u+P0+C0+F1-2_w001.mp4",
                                                   P + "charades_5s/db/WM336_1280x704_24fps_w003.mp4"})


def pool():
    rows = []
    for i in range(30):
        for w in range(3):
            rows.append({"video": P + f"robomind_5s/ur_1rgb/camera_top_ep{i:06d}_w{w}.mp4", "prompt": f"r{i}"})
    for i in range(4):
        rows.append({"video": P + f"noxi_5s/{i:03d}_novice_w001.mp4", "prompt": f"n{i}"})
    rows.append({"video": P + "celebvhq_5s/aa/SAMEVIDEO01_w001.mp4", "prompt": "c"})
    rows.append({"video": P + "openvidgen_5s/SAMEVIDEO01_3_0to100.mp4", "prompt": "o"})
    return rows


class SelectionTests(unittest.TestCase):
    def test_exclusion_cap_distinct_parents_and_small_dataset(self):
        excluded = {"robomind:ur_1rgb:ep000003", "noxi:session:1"}
        sel, stats = dev.select_devset(pool(), excluded, seed=7, per_dataset=6)
        by = {}
        for s in sel:
            by.setdefault(s["dataset"], []).append(s)
        self.assertEqual(len(by["robomind"]), 6)
        self.assertEqual(len(by["noxi"]), 3)  # fewer than 6 eligible -> take all
        self.assertNotIn("noxi:session:1", {k for s in sel for k in s["parent_keys"]})
        self.assertNotIn("robomind:ur_1rgb:ep000003", {k for s in sel for k in s["parent_keys"]})
        all_keys = [k for s in sel for k in s["parent_keys"]]
        self.assertEqual(len(all_keys), len(set(all_keys)))
        self.assertEqual(stats["excluded_windows_by_dataset"]["robomind"], 3)
        # same YouTube video under two datasets is used at most once
        self.assertEqual(sum("SAMEVIDEO01" in s["video"] for s in sel), 1)

    def test_deterministic_and_order_independent(self):
        a, _ = dev.select_devset(pool(), set(), seed=dev.SEED)
        shuffled = pool()
        random.Random(1).shuffle(shuffled)
        b, _ = dev.select_devset(shuffled, set(), seed=dev.SEED)
        self.assertEqual([s["video"] for s in a], [s["video"] for s in b])
        c, _ = dev.select_devset(pool(), set(), seed=dev.SEED + 1)
        self.assertNotEqual([s["video"] for s in a], [s["video"] for s in c])

    def test_unusable_windows_skipped_and_recorded(self):
        bad = P + "noxi_5s/000_novice_w001.mp4"
        sel, stats = dev.select_devset(pool(), set(), usable=lambda r: r["video"] != bad)
        self.assertNotIn(bad, [s["video"] for s in sel])
        self.assertIn(bad, stats["unusable_skipped"])


class SourceTests(unittest.TestCase):
    def test_fixed_val120_matches_eval_val_loss_slicing(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "val.csv"
            with open(p, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["video", "prompt"])
                for i in range(10):
                    w.writerow([P + f"noxi_5s/{i:03d}_novice_w001.mp4", "x"])
            self.assertEqual(dev.fixed_val120_paths(p, n=3), [P + f"noxi_5s/{i:03d}_novice_w001.mp4" for i in (0, 3, 6)])

    def test_missing_or_empty_source_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                dev.load_exclusions({"x": [f"{d}/nothing*.json"]})
            (Path(d) / "empty.json").write_text("{}")
            with self.assertRaises(ValueError):
                dev.load_exclusions({"x": [f"{d}/empty.json"]})

    def test_declared_sources_cover_required_sets(self):
        for name in ("wm108", "fixed_val120", "teacher_P1", "teacher_C1", "teacher_M1", "training_smoke"):
            self.assertIn(name, dev.EXCLUSION_SOURCES)
        self.assertEqual((dev.SEED, dev.PER_DATASET), (20261021, 6))


if __name__ == "__main__":
    unittest.main()
