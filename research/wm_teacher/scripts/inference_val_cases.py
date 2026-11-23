#!/usr/bin/env python3
"""
Mid-training validation inference for Wan2.2-TI2V-5B + round-1 LoRA.

Runs 3 TI2V cases from round1_val.csv (hoigen1m / noxi / matrix), using the
ground-truth first frame as input_image and the val caption as prompt.
Generates 49 frames (matches training), saves MP4 next to the GT video.

Run from DiffSynth-Studio dir with wan env + PYTHONPATH. Uses one GPU
(default 7) alongside training (~33GB, training uses ~44GB of 96GB).
"""
import csv
import os
import sys
import imageio.v2 as imageio
import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

ROOT = "/srv/wm_teacher"
CKPT = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/models/train/round1_lora_r32/step-8000.safetensors"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/models/train/round1_lora_r32/inference_val"
NUM_FRAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 49  # must be 4k+1; 121 = ~5s @24fps

WANT = ["hoigen1m", "noxi", "matrix"]
os.makedirs(OUT_DIR, exist_ok=True)

# pick cases
cases = {}
with open(f"{ROOT}/training_metadata/round1_val.csv") as f:
    for row in csv.DictReader(f):
        ds = row["video"].split("/")[2].split("_")[0]
        if ds in WANT and ds not in cases:
            cases[ds] = row
assert len(cases) == 3, f"expected 3 cases, got {list(cases)}"

print(f"Loading pipeline (bf16, cuda) ...", flush=True)
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth"),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
)
if CKPT == "NONE":
    print("Baseline: no LoRA", flush=True)
else:
    print(f"Loading LoRA: {CKPT}", flush=True)
    pipe.load_lora(pipe.dit, CKPT, alpha=1)
    print("LoRA loaded.", flush=True)

NEG = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

for ds, row in cases.items():
    gt_path = os.path.join(ROOT, row["video"])
    prompt = row["prompt"]
    out_path = os.path.join(OUT_DIR, f"{ds}_lora.mp4")

    # first frame of GT video as input image
    reader = imageio.get_reader(gt_path)
    first = Image.fromarray(reader.get_data(0)).resize((1280, 704))
    reader.close()

    print(f"[{ds}] generating ... prompt: {prompt[:70]}...", flush=True)
    video = pipe(
        prompt=prompt,
        negative_prompt=NEG,
        seed=0, tiled=True,
        height=704, width=1280,
        num_frames=NUM_FRAMES,
        input_image=first,
    )
    save_video(video, out_path, fps=24, quality=5)
    print(f"[{ds}] saved -> {out_path}", flush=True)

print("ALL DONE", flush=True)
