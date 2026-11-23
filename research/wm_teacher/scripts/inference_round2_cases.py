#!/usr/bin/env python3
"""
TI2V inference on held-out round-2 val clips (5s windows, Qwen3.5 captions) for one checkpoint.
GT first frame -> input_image, val caption -> prompt, 121 frames @24fps, seed 0, so outputs of
base / round-1 / round-2 checkpoints are directly comparable side by side.

Usage (from DiffSynth-Studio, PYTHONPATH set):
  python inference_round2_cases.py CKPT|NONE OUT_DIR [DATASETS]
"""
import csv
import os
import shutil
import sys

import imageio.v2 as imageio
import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

ROOT = "/srv/wm_teacher"
CKPT, OUT_DIR = sys.argv[1], sys.argv[2]
WANT = (sys.argv[3] if len(sys.argv) > 3 else "hoigen1m,celebvhq,mira,charades,noxi,matrix,direct,vfhq").split(",")
NUM_FRAMES = 121
os.makedirs(OUT_DIR, exist_ok=True)

cases = {}
for row in csv.DictReader(open(f"{ROOT}/training_metadata/round2/round2_val.csv")):
    ds = row["video"].split("/")[2].rsplit("_5s", 1)[0]
    if ds in WANT and ds not in cases:
        cases[ds] = row

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth"),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
)
if CKPT != "NONE":
    pipe.load_lora(pipe.dit, CKPT, alpha=1)
print(f"ckpt={CKPT} cases={list(cases)}", flush=True)

NEG = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

with open(os.path.join(OUT_DIR, "cases.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "video", "prompt"])
    for ds, row in cases.items():
        w.writerow([ds, row["video"], row["prompt"]])

for ds, row in cases.items():
    gt = os.path.join(ROOT, row["video"])
    reader = imageio.get_reader(gt)
    first = Image.fromarray(reader.get_data(0)).resize((1280, 704))
    reader.close()
    video = pipe(prompt=row["prompt"], negative_prompt=NEG, seed=0, tiled=True,
                 height=704, width=1280, num_frames=NUM_FRAMES, input_image=first)
    save_video(video, os.path.join(OUT_DIR, f"{ds}_gen.mp4"), fps=24, quality=5)
    shutil.copyfile(gt, os.path.join(OUT_DIR, f"{ds}_gt.mp4"))
    print(f"[{ds}] done", flush=True)
print("ALL DONE", flush=True)
