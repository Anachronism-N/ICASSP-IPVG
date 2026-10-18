#!/usr/bin/env python3
"""
Sharded TI2V generation for Wan2.2-TI2V-5B (+ optional LoRA) over a job list. One process per GPU.

Job list (JSONL), one line per video:
  {"id", "model", "ckpt" (LoRA path or "NONE"), "image" (.png/.jpg, or .mp4 -> frame 0),
   "prompt", "out", "seed"}
Jobs are grouped by model; process k of N takes every N-th job of each model (static round robin),
loads the pipeline once per model, skips outputs that already exist (resume-safe, atomic rename).

Usage (from DiffSynth-Studio, PYTHONPATH set):
  python gen_jobs.py JOBS.jsonl SHARD NSHARDS [--models base,round2]
"""
import argparse
import hashlib
import json
import os
import time
from collections import defaultdict

import imageio.v2 as imageio
import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

W, H, NUM_FRAMES = 1280, 704, 121
NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
       "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
       "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")


def load_image(path):
    if path.endswith(".mp4"):
        r = imageio.get_reader(path)
        img = Image.fromarray(r.get_data(0))
        r.close()
    else:
        img = Image.open(path).convert("RGB")
    # resize to cover 1280x704, then center crop (16:9 inputs lose 8px top/bottom)
    s = max(W / img.width, H / img.height)
    img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    l, t = (img.width - W) // 2, (img.height - H) // 2
    return img.crop((l, t, l + W, t + H))


def local_checkpoint(ckpt):
    cache = os.environ.get("WM_CHECKPOINT_CACHE_DIR", "/dockerdata/models/wm_checkpoints")
    key = hashlib.sha256(os.path.abspath(ckpt).encode()).hexdigest() + ".safetensors"
    path = os.path.join(cache, key)
    if os.path.isfile(path):
        if os.path.getsize(path) != os.path.getsize(ckpt):
            raise RuntimeError(f"Incomplete local checkpoint cache: {path}")
        print(f"[checkpoint cache] {ckpt} -> {path}", flush=True)
        return path
    return ckpt


def build_pipe(ckpt):
    if ckpt != "NONE":
        ckpt = local_checkpoint(ckpt)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
    )
    if ckpt != "NONE":
        from safetensors import safe_open
        with safe_open(ckpt, "pt") as f:
            full = not any("lora" in k for k in f.keys())
        if full:  # full-DiT fine-tune checkpoint
            from safetensors.torch import load_file
            missing, unexpected = pipe.dit.load_state_dict(load_file(ckpt), strict=False)
            assert not unexpected and not missing, (len(missing), len(unexpected))
        else:
            pipe.load_lora(pipe.dit, ckpt, alpha=1)
    return pipe


def wait_same_gpu_valloss():
    """eval_val_loss.py launched on our GPU (pipeline runs it next to the gen shards) -> wait for it, else OOM."""
    time.sleep(20)  # the pipeline starts val loss a moment before the gen shards
    me = os.environ.get("CUDA_VISIBLE_DEVICES")
    while True:
        busy = False
        for pid in filter(str.isdigit, os.listdir("/proc")):
            try:
                cmd = open(f"/proc/{pid}/cmdline", "rb").read()
                if b"eval_val_loss.py" not in cmd:
                    continue
                env = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
                busy = f"CUDA_VISIBLE_DEVICES={me}".encode() in env
            except OSError:
                continue
            if busy:
                break
        if not busy:
            return
        print("[wait] eval_val_loss.py is using this GPU", flush=True)
        time.sleep(120)


STALE_CLAIM_S = 40 * 60  # a 121f video takes ~6 min; an older claim without output = crashed worker


def active_claim(out):
    try:
        return time.time() - os.path.getmtime(out + ".claim") <= STALE_CLAIM_S
    except FileNotFoundError:
        return False


def claim(out):
    """Atomic per-job claim on the shared fs so idle shards can take over a straggler's remaining jobs."""
    c = out + ".claim"
    try:
        os.close(os.open(c, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        try:
            if time.time() - os.path.getmtime(c) > STALE_CLAIM_S and not os.path.exists(out):
                os.remove(c)
                return claim(out)
        except OSError:
            pass
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("shard", type=int)
    ap.add_argument("nshards", type=int)
    ap.add_argument("--models", default="")
    a = ap.parse_args()
    wait_same_gpu_valloss()

    by_model = defaultdict(list)
    for line in open(a.jobs):
        j = json.loads(line)
        by_model[(j["model"], j["ckpt"])].append(j)
    want = set(a.models.split(",")) if a.models else None
    failures = 0
    for (model, ckpt), jobs in by_model.items():
        if want and model not in want:
            continue
        todo = [j for j in jobs if not os.path.exists(j["out"])]
        mine = [j for i, j in enumerate(jobs) if i % a.nshards == a.shard and not os.path.exists(j["out"])]
        print(f"[{model}] {len(mine)} jobs for shard {a.shard}/{a.nshards}", flush=True)
        if not todo:
            continue
        # own round-robin share first, then (in reverse, to meet the owner from the other end) everyone else's
        ids = {id(j) for j in mine}
        order = mine + [j for j in reversed(todo) if id(j) not in ids]
        if all(active_claim(j["out"]) for j in order):
            continue  # everything left is already being generated elsewhere; don't load the pipeline
        pipe = build_pipe(ckpt)
        t0 = time.time()
        k = 0
        for j in order:
            if os.path.exists(j["out"]):
                continue
            os.makedirs(os.path.dirname(j["out"]), exist_ok=True)
            if not claim(j["out"]):
                continue
            k += 1
            try:
                # optional per-job sampler overrides (inference sweeps); defaults = Wan2.2-TI2V-5B official
                video = pipe(prompt=j["prompt"], negative_prompt=j.get("neg", NEG), seed=j.get("seed", 0), tiled=True,
                             height=H, width=W, num_frames=NUM_FRAMES, input_image=load_image(j["image"]),
                             cfg_scale=j.get("cfg", 5.0), sigma_shift=j.get("shift", 5.0),
                             num_inference_steps=j.get("steps", 50),
                             cfg_zero_star=j.get("cfg_zero_star", False), cfg_zero_steps=j.get("cfg_zero_steps", 0))
                tmp = j["out"] + ".tmp.mp4"
                save_video(video, tmp, fps=24, quality=5)
                os.replace(tmp, j["out"])
            except Exception as e:
                failures += 1
                print(f"[{model}] FAIL {j['id']}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            finally:
                try:
                    os.remove(j["out"] + ".claim")
                except OSError:
                    pass
            el = time.time() - t0
            print(f"[{model}] {k} done (own share {len(mine)}) {j['id']} {el / k:.0f}s/video", flush=True)
        del pipe
        torch.cuda.empty_cache()
    if failures:
        raise SystemExit(f"[shard failed] {failures} generation errors; check logs and missing outputs")
    print("[shard done]", flush=True)


if __name__ == "__main__":
    main()
