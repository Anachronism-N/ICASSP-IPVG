import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("build_e1_jobs", SCRIPTS / "build_e1_jobs.py")
jb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jb)

CASES = [{"id": f"noxi__{i:03d}_novice_w001", "image": f"/img/{i}.png", "prompt": f"p{i}"} for i in range(5)]


class JobTests(unittest.TestCase):
    def test_grid_fields_and_count(self):
        jobs = jb.build_jobs(CASES, "/v")
        self.assertEqual(len(jobs), 5 * 12)
        self.assertEqual({(j["shift"], j["cfg"]) for j in jobs},
                         {(float(s), float(c)) for s in (3, 5, 8, 12) for c in (4, 5, 6)})
        self.assertIn((5.0, 5.0), {(j["shift"], j["cfg"]) for j in jobs})
        for j in jobs:
            self.assertEqual(set(j), {"id", "model", "ckpt", "image", "prompt", "out", "seed", "cfg", "shift", "steps"})
            self.assertEqual((j["model"], j["ckpt"], j["seed"], j["steps"]), ("base", "NONE", 0, 50))
            self.assertNotIn("neg", j)  # gen_jobs default negative prompt
            self.assertTrue(j["out"].startswith(f"/v/s{int(j['shift'])}_c{int(j['cfg'])}/"))
        self.assertEqual(len({j["out"] for j in jobs}), len(jobs))
        self.assertEqual(len({j["id"] for j in jobs}), len(jobs))
        self.assertEqual(len({(j["model"], j["ckpt"]) for j in jobs}), 1)  # one pipeline load per shard
        # case-major: first 12 jobs are all configs of case 0
        self.assertEqual({j["image"] for j in jobs[:12]}, {"/img/0.png"})

    def test_prompt_and_image_passthrough(self):
        for j in jb.build_jobs(CASES, "/v"):
            i = int(j["image"].split("/")[-1].split(".")[0])
            self.assertEqual(j["prompt"], f"p{i}")

    def test_write_refuses_existing_file_or_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            jobs = jb.build_jobs(CASES[:1], f"{d}/v")
            jb.write_jobs(jobs, f"{d}/jobs.jsonl")
            self.assertEqual(len(open(f"{d}/jobs.jsonl").readlines()), 12)
            with self.assertRaises(FileExistsError):
                jb.write_jobs(jobs, f"{d}/jobs.jsonl")
            Path(jobs[3]["out"]).parent.mkdir(parents=True)
            Path(jobs[3]["out"]).write_bytes(b"x")
            with self.assertRaises(RuntimeError):
                jb.write_jobs(jobs, f"{d}/jobs2.jsonl")

    def test_smoke_configs(self):
        jobs = jb.build_jobs(CASES[:1], "/s", jb.SMOKE_CONFIGS)
        self.assertEqual([(j["shift"], j["cfg"]) for j in jobs], [(5.0, 5.0), (12.0, 6.0)])
        json.dumps(jobs)


if __name__ == "__main__":
    unittest.main()
