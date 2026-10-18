#!/usr/bin/env python3
"""Wait for local generators/GPU 0..2, then evaluate the unchanged fixed120 loss protocol."""
import argparse
import concurrent.futures
import csv
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

from score_merge_cpu import read_json, require, finite, publish, suite_lock

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("m_full1_a03", "m_full1_a05", "m_full1_a07")
T_FRACS = [0.1, 0.3, 0.5, 0.7, 0.9]
LOSS_DIR = "models/train/round2_5s_lora_r32/eval"


def is_generator(args):
    words = [x.decode(errors="replace") if isinstance(x, bytes) else x for x in args if x]
    if not words or not re.fullmatch(r"(?:python|pypy)(?:\d+(?:\.\d+)*)?", Path(words[0]).name):
        return False
    # Interpreter options may consume a value; neither that value nor a body is a script.
    index = 1
    while index < len(words):
        word = words[index]
        if word in ("-", "--help", "--version"):
            return False
        if word == "--":
            index += 1
            return index < len(words) and Path(words[index]).name == "gen_jobs.py"
        if re.match(r"-[bBdEiIOPqRsSuv]*(?:c|m|h|V|\?)", word):
            return False
        if word in ("-W", "-X", "--check-hash-based-pycs"):
            index += 2
            continue
        if not word.startswith("-"):
            return Path(word).name == "gen_jobs.py"
        index += 1
    return False


def generators_active(proc=Path("/proc")):
    for process in proc.iterdir():
        if not process.name.isdigit():
            continue
        try:
            if is_generator((process / "cmdline").read_bytes().split(b"\0")):
                return True
        except FileNotFoundError:
            continue
    return False


def gpu_idle(run=subprocess.run):
    result = run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                 capture_output=True, text=True, check=True)
    used = {int(index.strip()): int(memory.strip()) for index, memory in
            (line.split(",") for line in result.stdout.splitlines() if line.strip())}
    return all(index in used and used[index] < 100 for index in (0, 1, 2))


def wait_for_release(interval=60, active=generators_active, idle=gpu_idle, sleep=time.sleep):
    while active() or not idle():
        sleep(interval)
    print("WORKER_GENERATION_EXIT_GPU0_2_RELEASED", flush=True)


def validate_loss(path, base):
    result = read_json(path)
    fields = {"n", "clips", "num_frames", "t_fracs", "t_ids", "mean_loss",
              "mean_loss_per_t", "per_dataset"}
    for data in (base, result):
        require(isinstance(data, dict) and fields <= data.keys() and
                isinstance(data["clips"], list) and isinstance(data["t_fracs"], list) and
                isinstance(data["t_ids"], list) and isinstance(data["mean_loss_per_t"], list) and
                isinstance(data["per_dataset"], dict),
                f"Missing/invalid fixed120 loss fields; preserve and review: {path}")
        require(all(isinstance(c, dict) and {"video", "dataset", "loss_t", "loss"} <= c.keys() and
                    isinstance(c["video"], str) and isinstance(c["dataset"], str) and
                    isinstance(c["loss_t"], list) for c in data["clips"]),
                f"Missing/invalid clip loss fields; preserve and review: {path}")
    require(type(result["n"]) is int and result["n"] == 120 and len(result["clips"]) == 120 and
            type(result["num_frames"]) is int and result["num_frames"] == 121,
            f"Incomplete fixed120/121f loss: {path}")
    videos = [clip["video"] for clip in result["clips"]]
    expected = [clip["video"] for clip in base["clips"]]
    # Order matters: the original evaluator seeds noise by clip index, not video id.
    require(len(set(videos)) == 120 and videos == expected,
            f"Loss videos/order differ from original fixed120 base: {path}")
    require(result["t_fracs"] == T_FRACS == base["t_fracs"] and
            result["t_ids"] == base["t_ids"] and len(result["t_ids"]) == 5 and
            all(type(t) is int for t in result["t_ids"]), f"Changed fixed timesteps: {path}")
    require(finite(result["mean_loss"]) and
            all(len(c["loss_t"]) == 5 and all(finite(v) for v in c["loss_t"]) and
                finite(c["loss"]) and math.isclose(c["loss"], sum(c["loss_t"]) / 5, rel_tol=1e-9)
                for c in result["clips"]), f"Non-finite/inconsistent clip loss: {path}")
    require(math.isclose(result["mean_loss"], sum(c["loss"] for c in result["clips"]) / 120,
                         rel_tol=1e-9), f"Inconsistent mean loss: {path}")
    require(len(result["mean_loss_per_t"]) == 5 and all(finite(v) for v in result["mean_loss_per_t"]) and
            all(finite(v) for v in result["per_dataset"].values()), f"Non-finite loss aggregates: {path}")
    require(all(math.isclose(result["mean_loss_per_t"][k],
                             sum(c["loss_t"][k] for c in result["clips"]) / 120, rel_tol=1e-9)
                for k in range(5)), f"Inconsistent per-t loss aggregates: {path}")
    require([c["dataset"] for c in result["clips"]] == [c["dataset"] for c in base["clips"]] and
            all(len(c["video"].split("/")) > 2 and
                c["dataset"] == c["video"].split("/")[2].rsplit("_5s", 1)[0]
                for c in result["clips"]), f"Changed fixed120 dataset membership: {path}")
    by_dataset = {}
    for clip in result["clips"]:
        by_dataset.setdefault(clip["dataset"], []).append(clip["loss"])
    require(set(result["per_dataset"]) == set(by_dataset) and
            all(math.isclose(result["per_dataset"][dataset], sum(values) / len(values), rel_tol=1e-9)
                for dataset, values in by_dataset.items()), f"Inconsistent dataset loss aggregates: {path}")
    return result


def fixed_base(root):
    path = root / LOSS_DIR / "loss_base.json"
    base = read_json(path)
    validate_loss(path, base)
    with (root / "training_metadata/round2/round2_val.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    original = [r["video"] for r in rows[::max(1, len(rows) // 120)][:120]]
    require(original == [c["video"] for c in base["clips"]],
            "Current validation CSV no longer matches original fixed120 base; preserve and review")
    return base


def score_model(root, model, gpu, base, python, run=subprocess.run):
    require(model in MODELS, f"Unknown fixed merge model: {model}")
    destination = root / LOSS_DIR / ("loss_" + model + ".json")
    with suite_lock(root, model, "val_loss"):
        if destination.exists():
            validate_loss(destination, base)
            print("SKIP_VALID_FIXED120_LOSS", model, flush=True)
            return
        require(destination.parent.is_dir(), f"Missing original loss directory: {destination.parent}")
        work = Path(tempfile.mkdtemp(prefix="." + model + ".loss-", dir=destination.parent))
        output = work / destination.name
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                   PYTHONPATH=str(root / "DiffSynth-Studio") + ":" + str(root / "scripts"),
                   DIFFSYNTH_MODEL_BASE_PATH="/dockerdata/models", DIFFSYNTH_SKIP_DOWNLOAD="true",
                   WM_CHECKPOINT_CACHE_DIR="/dockerdata/models/wm_checkpoints",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false")
        log = root / "dataset/download_logs" / ("val_loss_" + model + "_20261018.log")
        with log.open("a") as handle:
            run([python, "-u", str(root / "scripts/eval_val_loss.py"), "--ckpt",
                 str(root / "models/train/merge" / (model.removeprefix("m_") + ".safetensors")),
                 "--val-csv", str(root / "training_metadata/round2/round2_val.csv"),
                 "--n", "120", "--num-frames", "121", "--out", str(output)],
                cwd=root / "DiffSynth-Studio", env=env, stdout=handle,
                stderr=subprocess.STDOUT, check=True)
        result = validate_loss(output, base)
        publish(output, destination)
        print("FIXED120_VAL_LOSS_PASS", model, result["mean_loss"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--python", default="/opt/conda/envs/torch-base/bin/python")
    parser.add_argument("--interval", type=float, default=60)
    args = parser.parse_args()
    require(args.interval > 0, "Interval must be positive")
    root = args.root.resolve()
    base = fixed_base(root)
    wait_for_release(args.interval)
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        list(pool.map(lambda model: score_model(root, model, MODELS.index(model), base, args.python), args.models))
    print("THREE_MERGE_FIXED120_VAL_LOSS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
