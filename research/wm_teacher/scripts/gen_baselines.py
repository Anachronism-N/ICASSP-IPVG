#!/usr/bin/env python3
"""
Related-work I2V baselines on the WM-Bench cases (diffusers, each model at its official default
resolution / length / sampler; same GT first frame + caption as our models; seed 0).

  cogvideox  CogVideoX-5b-I2V      720x480,  49 frames @ 8 fps,  50 steps, cfg 6.0
  hunyuan    HunyuanVideo-I2V      1280x720, 129 frames @ 24 fps, 50 steps, true-cfg off (official i2v default)
  wan21      Wan2.1-I2V-14B-720P   1280x720, 81 frames @ 16 fps,  50 steps, cfg 5.0

40GB A100: big models use model CPU offload + VAE tiling. One process per GPU, sharded like
gen_jobs.py. Output: eval/wm_bench/videos/<model>/<id>.mp4 (native fps).
Usage: /dockerdata/venv_bl/bin/python gen_baselines.py MODEL SHARD NSHARDS
"""
import json
import os
import sys
import time

import torch
from diffusers.utils import export_to_video
from PIL import Image
import imageio.v2 as imageio

ROOT = "/srv/wm_teacher"
EV = f"{ROOT}/eval/wm_bench"
BL = "/dockerdata/models_bl"
CFG = {
    "cogvideox": dict(path=f"{BL}/CogVideoX-5b-I2V", w=720, h=480, frames=49, fps=8, steps=50, cfg=6.0),
    # 720p x 129 frames needs ~60GB+; 544p (its other supported tier) is what fits a 40GB A100
    "hunyuan": dict(path=f"{BL}/HunyuanVideo-I2V", w=960, h=544, frames=129, fps=24, steps=50, cfg=1.0),
    "wan21": dict(path=f"{BL}/Wan2.1-I2V-14B-720P", w=1280, h=720, frames=81, fps=16, steps=50, cfg=5.0),
}
WAN_NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
           "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
           "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")


def first_frame(path, w, h):
    r = imageio.get_reader(path)
    img = Image.fromarray(r.get_data(0))
    r.close()
    s = max(w / img.width, h / img.height)  # cover + center crop to the model's native aspect
    img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    l, t = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((l, t, l + w, t + h))


def load(model):
    c = CFG[model]
    if model == "cogvideox":
        from diffusers import CogVideoXImageToVideoPipeline
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(c["path"], torch_dtype=torch.bfloat16)
        pipe.to("cuda")
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
    elif model == "hunyuan":
        from diffusers import HunyuanVideoImageToVideoPipeline
        pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(c["path"], torch_dtype=torch.bfloat16)
        pipe.vae.enable_tiling()
        pipe.enable_model_cpu_offload()
    else:
        from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
        vae = AutoencoderKLWan.from_pretrained(c["path"], subfolder="vae", torch_dtype=torch.float32)
        pipe = WanImageToVideoPipeline.from_pretrained(c["path"], vae=vae, torch_dtype=torch.bfloat16)
        # 14B DiT in bf16 (28GB) + 720p x 81f activations OOMs on 40GB and the sequential-offload
        # fallback takes ~4.7h/video. Store DiT weights in fp8, upcast per layer to bf16 for compute
        # (diffusers layerwise casting; norms/embeddings stay bf16) -> 14GB of weights.
        pipe.transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
        pipe.vae.enable_tiling()
        pipe.enable_model_cpu_offload()
    return pipe


def run(pipe, model, img, prompt):
    c = CFG[model]
    g = torch.Generator("cuda").manual_seed(0)
    kw = dict(image=img, prompt=prompt, height=c["h"], width=c["w"], num_frames=c["frames"],
              num_inference_steps=c["steps"], generator=g)
    if model == "cogvideox":
        kw.update(guidance_scale=c["cfg"], use_dynamic_cfg=True)
    elif model == "wan21":
        kw.update(guidance_scale=c["cfg"], negative_prompt=WAN_NEG)
    return pipe(**kw).frames[0]


LOCK_TTL = 3 * 3600  # a claim older than this is considered abandoned (crashed worker)


def claim(out):
    """Atomically claim a job via O_EXCL lock file, so any number of workers can join or leave."""
    lock = out + ".lock"
    if os.path.exists(out):
        return False
    try:
        if time.time() - os.path.getmtime(lock) > LOCK_TTL:
            os.remove(lock)
    except OSError:
        pass
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return not os.path.exists(out)
    except FileExistsError:
        return False


def remaining(model):
    """number of cases with neither an output nor a live claim (used by gpu_filler.sh)"""
    cases = [json.loads(l) for l in open(f"{EV}/cases.jsonl")]
    d = f"{EV}/videos/{model}"
    n = 0
    for c in cases:
        o = f"{d}/{c['id']}.mp4"
        lk = o + ".lock"
        if os.path.exists(o):
            continue
        if os.path.exists(lk) and time.time() - os.path.getmtime(lk) <= LOCK_TTL:
            continue
        n += 1
    return n


def main():
    if sys.argv[1] == "--remaining":
        print(remaining(sys.argv[2]))
        return
    # SHARD only rotates the starting point; jobs are taken dynamically via claim()
    model, shard, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    cases = [json.loads(l) for l in open(f"{EV}/cases.jsonl")]
    out_dir = f"{EV}/videos/{model}"
    os.makedirs(out_dir, exist_ok=True)
    start = len(cases) * shard // max(n, 1)
    order = cases[start:] + cases[:start]
    print(f"[{model}] worker {shard}/{n}: {remaining(model)} unclaimed jobs", flush=True)
    if not remaining(model):
        return
    pipe = load(model)
    sequential = False
    t0 = time.time()
    k = 0
    for c in order:
        out = f"{out_dir}/{c['id']}.mp4"
        if not claim(out):
            continue
        k += 1
        img = first_frame(c["image"], CFG[model]["w"], CFG[model]["h"])
        try:
            try:
                frames = run(pipe, model, img, c["prompt"])
            except torch.OutOfMemoryError:
                if sequential:
                    raise
                # weights-streaming fallback: slower, but fits any of these models in 40GB
                print(f"[{model}] OOM -> switching to sequential CPU offload", flush=True)
                torch.cuda.empty_cache()
                pipe.remove_all_hooks()
                pipe.enable_sequential_cpu_offload()
                sequential = True
                frames = run(pipe, model, img, c["prompt"])
            tmp = out + ".tmp.mp4"
            export_to_video(frames, tmp, fps=CFG[model]["fps"])
            os.replace(tmp, out)
        except Exception as e:
            print(f"[{model}] FAIL {c['id']}: {type(e).__name__}: {str(e)[:300]}", flush=True)
            continue
        finally:
            try:
                os.remove(out + ".lock")
            except OSError:
                pass
        print(f"[{model}] {k} {c['id']} {(time.time() - t0) / k:.0f}s/video", flush=True)
    print("[shard done]", flush=True)


if __name__ == "__main__":
    main()
