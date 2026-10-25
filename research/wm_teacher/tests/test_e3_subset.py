import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_e3_subset.py"
SPEC = importlib.util.spec_from_file_location("prepare_e3_subset", SCRIPT)
subset = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subset)


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.eval = self.root / "eval"
        self.eval.mkdir()
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.checkpoint = self.root / "models/train/e3_arms_20261023/armC_fixed_anchor.safetensors"
        self.put(self.checkpoint, b"synthetic checkpoint")
        self.vb, self.wm, self.phys, self.wmb = [], [], [], []
        self.jobs = {name: [] for name in subset.SOURCE_SHA256}
        for i in range(355):
            cid = f"sample {i:03d}"
            image = self.eval / f"vbench_i2v/images_16-9/{cid}.jpg"
            self.put(image, f"image {i}".encode())
            dims = ["i2v_subject", "subject_consistency"] if i < 246 else ["i2v_background", "background_consistency"]
            self.vb.append({"prompt_en": cid, "dimension": dims, "image_type": f"type{i % 5}", "image_name": image.name})
            for suite, model, name, prompt in [("vbench_i2v", "base", "jobs_vbench.jsonl", cid),
                                                ("vbench_i2v", "base_po", "jobs_vbench_po.jsonl", f"Detailed {cid}")]:
                self.jobs[name].append(self.job(cid, image, prompt, suite, model, f"{cid}-0.mp4"))
        datasets = [(f"dataset{i}", 10) for i in range(10)] + [("small", 6), ("tiny", 2)]
        for ds, count in datasets:
            for i in range(count):
                cid = f"{ds}__{i}"
                image = self.root / f"dataset/{ds}/{i}.mp4"
                self.put(image, f"WM input {cid}".encode())
                prompt = f"Neutral activity {cid}"
                j = self.job(cid, image, prompt, "wm_bench", "base", f"{cid}.mp4")
                self.jobs["jobs_wm.jsonl"].append(j)
                self.wm.append(dict(id=cid, image=str(image), gt=str(image), dataset=ds, prompt=prompt))
        for scene in range(66):
            for view_index, view in enumerate(("left", "center", "right")):
                cid = f"{scene * 3 + view_index + 1:04d}"
                filename = f"{cid}_perspective-{view}_trimmed-scene-{scene}.mp4"
                image = self.eval / f"physiq/inputs/{cid}.jpg"
                self.put(image, f"physics input {cid}".encode())
                prompt = f"Neutral physics scene {scene}"
                self.jobs["jobs_extra_base.jsonl"].append(self.job(cid, image, prompt, "physiq", "base-op-run_01", filename))
                self.phys.append({"scenario": filename.replace("_trimmed-", "_take-1_trimmed-"),
                                  "description": prompt, "category": f"category{scene % 5}", "generated_video_name": filename})
        for i in range(350):
            cid = f"world{i:03d}"
            image = self.eval / f"wmb/inputs/{cid}.jpg"
            self.put(image, f"world input {i}".encode())
            prompt = f"Neutral world action {i}"
            self.jobs["jobs_extra_base.jsonl"].append(self.job(cid, image, prompt, "wmb", "base", f"{cid}.mp4"))
            self.wmb.append({"first_frame": f"images/{cid}.jpg", "text_instruction": prompt, "domain": f"domain{i % 7}"})
        self.vb_path, self.phys_path, self.wmb_path = [self.root / name for name in ("vb.json", "physics.csv", "world.json")]
        self.write_metadata()
        self.write_sources()
        scripts = self.root / "scripts"
        self.put(scripts / "prepare_e3_subset.py", SCRIPT.read_bytes())
        self.put(scripts / "gen_jobs.py", b'W, H, NUM_FRAMES = 1280, 704, 121\nNEG = "default negative"\ndef build_pipe():\n    pipe.load_lora(pipe.dit, ckpt, alpha=1)\n')
        for name in ("build_eval_jobs.py", "build_extra_bench_jobs.py"):
            self.put(scripts / name, b"# synthetic builder\n")
        self.put(self.root / "DiffSynth-Studio/diffsynth/__init__.py", b"# synthetic import\n")
        self.put(self.eval / "scores_DO_NOT_READ.json", b"not JSON: forbidden selection input")

    @staticmethod
    def put(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def job(self, cid, image, prompt, suite, model, filename):
        out = self.eval / suite / "videos" / model / filename
        self.put(out, f"reference video {suite} {model} {cid}".encode())
        return dict(id=cid, model="base_po" if model == "base_po" else "base", ckpt="NONE",
                    image=str(image), prompt=prompt, out=str(out), seed=0)

    def write_sources(self):
        self.shas = {}
        for name, rows in self.jobs.items():
            data = b"".join(subset.encoded(j) for j in rows)
            self.put(self.eval / name, data)
            self.shas[name] = subset.sha_bytes(data)

    def write_metadata(self):
        import csv
        import io
        self.put(self.vb_path, subset.encoded(self.vb))
        self.put(self.wmb_path, subset.encoded(self.wmb))
        self.put(self.eval / "wm_bench/cases.jsonl", b"".join(subset.encoded(c) for c in self.wm))
        text = io.StringIO()
        writer = csv.DictWriter(text, fieldnames=["scenario", "description", "category", "generated_video_name"])
        writer.writeheader()
        writer.writerows(self.phys)
        self.put(self.phys_path, text.getvalue().encode())

    def prepare(self):
        with patch.object(subset, "SOURCE_SHA256", self.shas), patch.object(subset, "runtime_snapshot", return_value=({"python": str(Path(subset.sys.executable).absolute())}, [])):
            return subset.prepare(self.root, vb_info=self.vb_path, phys_info=self.phys_path,
                                  wmb_info=self.wmb_path, cache_dir=self.cache)

    @property
    def plan_path(self):
        return self.eval / "e3_subset_20261025/plan.json"


class SubsetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f = Fixture(self.tmp.name)

    def test_full_cpu_fixture_counts_contract_and_read_only_base(self):
        before = {j["out"]: Path(j["out"]).read_bytes() for rows in self.f.jobs.values() for j in rows}
        plan = self.f.prepare()
        p, jobs, cases = subset.verify_plan(self.f.plan_path)
        self.assertEqual(p, plan)
        self.assertEqual(len(jobs), 276)
        self.assertEqual({j["model"] for j in jobs}, {subset.TAG})
        self.assertEqual({c["model"] for c in cases if c["suite"] == "vbench_po"}, {subset.TAG + "_po"})
        self.assertEqual(plan["selection"]["counts"], subset.EXPECTED_COUNTS)
        self.assertEqual(len({(c["suite"], c["id"]) for c in cases}), 276)
        raw = {c["id"] for c in cases if c["suite"] == "vbench"}
        po = {c["id"] for c in cases if c["suite"] == "vbench_po"}
        self.assertEqual(raw, po)
        self.assertEqual(len(raw), 71)
        physics = [c for c in cases if c["suite"] == "physiq"]
        self.assertEqual(len({c["scene"] for c in physics}), 14)
        for scene in {c["scene"] for c in physics}:
            self.assertEqual({c["view"] for c in physics if c["scene"] == scene}, {"left", "center", "right"})
        self.assertEqual({r["selected"] for r in plan["selection"]["strata"]["wmb"].values()}, {10})
        self.assertEqual(len(plan["selection"]["strata"]["wm"]), 12)
        self.assertEqual(sum(r["selected"] for r in plan["selection"]["strata"]["wm"].values()), 22)
        self.assertFalse(plan["launch"]["authorized_to_launch"])
        self.assertEqual(plan["selection"]["score_files_read"], 0)
        self.assertEqual(plan["budget"]["total_gpu_hours"], 40)
        self.assertAlmostEqual(plan["budget"]["scheduled_generation_gpu_hours"], 26 + 2 / 3)
        for j in jobs:
            self.assertEqual((j["cfg"], j["shift"], j["steps"], j["seed"]), (5, 5, 50, 0))
            self.assertNotIn("neg", j)
            self.assertFalse(Path(j["out"]).exists())
        for path, data in before.items():
            self.assertEqual(Path(path).read_bytes(), data)
        for name in ("plan.json", "plan.sha256", "jobs.jsonl", "cases.jsonl"):
            self.assertEqual((self.f.plan_path.parent / name).stat().st_mode & 0o777, 0o600)

    def test_selection_deterministic_and_score_blind(self):
        sources = {}
        for suite, name in (("vbench", "jobs_vbench.jsonl"), ("vbench_po", "jobs_vbench_po.jsonl"), ("wm", "jobs_wm.jsonl")):
            sources[suite] = {j["id"]: j for j in self.f.jobs[name]}
        for suite in ("physiq", "wmb"):
            sources[suite] = {j["id"]: j for j in self.f.jobs["jobs_extra_base.jsonl"] if f"/{suite}/" in j["out"]}
        first = subset.select_cases(sources, self.f.vb, self.f.wm, self.f.phys, self.f.wmb)
        self.f.put(self.f.eval / "scores_DO_NOT_READ.json", b'{"candidate": 1e308, "base": -1e308}')
        second = subset.select_cases(sources, self.f.vb, self.f.wm, self.f.phys, self.f.wmb)
        self.assertEqual(first, second)

    def test_safety_union_exclusion_and_nonsexual_art(self):
        self.assertTrue(subset.unsafe_reasons("intimate moment"))
        self.assertTrue(subset.unsafe_reasons("intimate close-up"))
        self.assertTrue(subset.unsafe_reasons("romantic scene"))
        self.assertTrue(subset.unsafe_reasons("nude"))
        self.assertTrue(subset.unsafe_reasons("kiss passionately"))
        self.assertTrue(subset.unsafe_reasons("seductive"))
        self.assertFalse(subset.unsafe_reasons("A marble sculpture in a museum"))
        self.f.jobs["jobs_vbench_po.jsonl"][0]["prompt"] = "seductive"
        self.f.write_sources()
        plan = self.f.prepare()
        _, _, cases = subset.verify_plan(self.f.plan_path)
        self.assertNotIn(self.f.vb[0]["prompt_en"], {c["id"] for c in cases if c["suite"].startswith("vbench")})
        exclusion = plan["selection"]["safety_exclusions"]["vbench_paired"]
        self.assertEqual(exclusion["excluded_cases"], 1)
        self.assertNotIn("id", json.dumps(exclusion))
        self.assertNotIn("prompt", json.dumps(exclusion))

    def test_source_hash_mismatch_before_writes(self):
        with (self.f.eval / "jobs_wm.jsonl").open("ab") as f:
            f.write(b"\n")
        with self.assertRaisesRegex(ValueError, "source manifest SHA256"):
            self.f.prepare()
        self.assertFalse(self.f.plan_path.parent.exists())

    def test_duplicate_source_id_failclosed(self):
        self.f.jobs["jobs_wm.jsonl"][1]["id"] = self.f.jobs["jobs_wm.jsonl"][0]["id"]
        self.f.write_sources()
        with self.assertRaisesRegex(ValueError, "Duplicate job"):
            self.f.prepare()

    def test_duplicate_source_output_failclosed(self):
        self.f.jobs["jobs_wm.jsonl"][1]["out"] = self.f.jobs["jobs_wm.jsonl"][0]["out"]
        self.f.write_sources()
        with self.assertRaisesRegex(ValueError, "Duplicate job"):
            self.f.prepare()

    def test_missing_input_failclosed(self):
        Path(self.f.jobs["jobs_wm.jsonl"][0]["image"]).unlink()
        with self.assertRaisesRegex(ValueError, "Missing input"):
            self.f.prepare()

    def test_nonfinite_json_failclosed(self):
        for value in ("NaN", "Infinity", "-Infinity", "1e999"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                subset.loads('{"value": ' + value + '}')
        with self.assertRaises(ValueError):
            subset.validate_jobs([dict(self.f.jobs["jobs_wm.jsonl"][0], cfg=float("nan"))])

    def test_sampler_change_failclosed(self):
        for field, value in (("cfg", 4), ("shift", 8), ("steps", 20), ("seed", 1), ("cfg_zero_star", True)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                subset.validate_jobs([dict(self.f.jobs["jobs_wm.jsonl"][0], **{field: value})])

    def test_missing_physics_view_failclosed(self):
        self.f.phys.pop()
        self.f.write_metadata()
        with self.assertRaisesRegex(ValueError, "Physics take-1 coverage"):
            self.f.prepare()

    def test_input_mutation_after_plan_failclosed(self):
        self.f.prepare()
        _, jobs, _ = subset.verify_plan(self.f.plan_path)
        Path(jobs[0]["image"]).write_bytes(b"mutated image")
        with self.assertRaisesRegex(ValueError, "Frozen SHA256"):
            subset.verify_plan(self.f.plan_path)

    def test_reference_mutation_after_plan_failclosed(self):
        self.f.prepare()
        _, _, cases = subset.verify_plan(self.f.plan_path)
        Path(cases[0]["base_video"]).write_bytes(b"mutated reference")
        with self.assertRaisesRegex(ValueError, "Frozen SHA256"):
            subset.verify_plan(self.f.plan_path)

    def test_manifest_mutation_failclosed(self):
        self.f.prepare()
        with (self.f.plan_path.parent / "jobs.jsonl").open("ab") as f:
            f.write(b"\n")
        with self.assertRaisesRegex(ValueError, "Frozen SHA256"):
            subset.verify_plan(self.f.plan_path)

    def test_plan_mutation_failclosed(self):
        self.f.prepare()
        with self.f.plan_path.open("ab") as f:
            f.write(b"\n")
        with self.assertRaisesRegex(ValueError, "plan SHA256"):
            subset.verify_plan(self.f.plan_path)

    def test_code_import_mutation_failclosed(self):
        self.f.prepare()
        self.f.put(self.f.root / "DiffSynth-Studio/diffsynth/__init__.py", b"modified import")
        with self.assertRaisesRegex(ValueError, "Frozen SHA256"):
            subset.verify_plan(self.f.plan_path)

    def test_cache_same_size_wrong_content_failclosed(self):
        cache_path = self.f.cache / (subset.sha_bytes(str(self.f.checkpoint).encode()) + ".safetensors")
        cache_path.write_bytes(b"X" * self.f.checkpoint.stat().st_size)
        with self.assertRaisesRegex(ValueError, "cache content SHA256"):
            self.f.prepare()

    def test_cache_created_after_plan_is_checked(self):
        self.f.prepare()
        cache_path = self.f.cache / (subset.sha_bytes(str(self.f.checkpoint).encode()) + ".safetensors")
        cache_path.write_bytes(self.f.checkpoint.read_bytes())
        subset.verify_plan(self.f.plan_path)
        cache_path.write_bytes(b"X" * self.f.checkpoint.stat().st_size)
        with self.assertRaisesRegex(ValueError, "cache content SHA256"):
            subset.verify_plan(self.f.plan_path)

    def test_existing_output_directory_never_overwritten(self):
        self.f.prepare()
        original = self.f.plan_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "new private"):
            self.f.prepare()
        self.assertEqual(self.f.plan_path.read_bytes(), original)

    def test_generator_resolution_and_alpha_guard(self):
        path = self.f.root / "scripts/gen_jobs.py"
        data = path.read_bytes()
        for before, after in ((b"1280", b"640"), (b"alpha=1", b"alpha=0.5")):
            path.write_bytes(data.replace(before, after))
            with self.assertRaises(ValueError):
                subset.generator_settings(path)
        path.write_bytes(data)


if __name__ == "__main__":
    unittest.main()
