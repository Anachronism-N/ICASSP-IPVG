#!/usr/bin/env python3
"""
Held-out validation loss for Wan2.2-TI2V-5B (+ optional LoRA) on 5s windows.

Same objective as training (FlowMatchSFTLoss, first-frame conditioned TI2V) but deterministic:
for every val clip the loss is averaged over a FIXED set of timesteps with FIXED per-(clip,
timestep) noise, so numbers are directly comparable across checkpoints (base / round1 / round2).

Run from DiffSynth-Studio with PYTHONPATH set (see eval_round2.sh). One process per GPU:
  python eval_val_loss.py --ckpt PATH|none --val-csv CSV --n 120 --out OUT.json
"""
import argparse
import csv
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.getcwd(), "examples/wanvideo/model_training"))
from train import WanTrainingModule  # noqa: E402
from diffsynth.core import UnifiedDataset  # noqa: E402

ROOT = "/srv/wm_teacher"
M = ("Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,"
     "Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth")
T_FRACS = [0.1, 0.3, 0.5, 0.7, 0.9]  # positions in the scheduler's training timestep table


def is_full_ckpt(path):
    from safetensors import safe_open
    with safe_open(path, "pt") as f:
        return not any("lora" in k for k in f.keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LoRA safetensors, or 'none' for the base model")
    ap.add_argument("--val-csv", default=f"{ROOT}/training_metadata/round2/round2_val.csv")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--num-frames", type=int, default=121)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ckpt = None if a.ckpt == "none" else a.ckpt
    if ckpt is not None:
        from gen_jobs import local_checkpoint
        ckpt = local_checkpoint(ckpt)
    full = ckpt is not None and is_full_ckpt(ckpt)  # full-DiT fine-tune checkpoint (no LoRA keys)
    lora = {} if full or ckpt is None else dict(lora_base_model="dit", lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32)
    model = WanTrainingModule(
        model_id_with_origin_paths=M, **lora,
        resume_from_checkpoint=ckpt, remove_prefix_in_ckpt="pipe.dit." if ckpt else None,
        extra_inputs="input_image", task="sft", device="cuda",
    )
    model.eval()
    pipe = model.pipe

    # evenly spread, dataset-stratified subset of the val split (val csv is grouped by dataset)
    rows = list(csv.DictReader(open(a.val_csv)))
    step = max(1, len(rows) // a.n)
    rows = rows[::step][:a.n]
    op = UnifiedDataset.default_video_operator(
        base_path=ROOT, max_pixels=1280 * 704, height=704, width=1280,
        height_division_factor=16, width_division_factor=16, num_frames=a.num_frames,
        time_division_factor=4, time_division_remainder=1)

    n_t = len(pipe.scheduler.timesteps)
    t_ids = [min(n_t - 1, int(f * n_t)) for f in T_FRACS]
    per_clip, t0 = [], time.time()
    for i, r in enumerate(rows):
        data = {"video": op(r["video"]), "prompt": r["prompt"]}
        with torch.no_grad():
            inputs = model.get_pipeline_inputs(data)
            inputs = model.transfer_data_to_device(inputs, pipe.device, pipe.torch_dtype)
            for unit in pipe.units:
                inputs = pipe.unit_runner(unit, pipe, *inputs)
            shared, posi, _ = inputs
            kw = {**shared, **posi}
            x0 = kw["input_latents"]
            models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
            losses = []
            for tid in t_ids:
                g = torch.Generator(device=x0.device).manual_seed(1000 * i + tid)
                noise = torch.randn(x0.shape, generator=g, device=x0.device, dtype=x0.dtype)
                t = pipe.scheduler.timesteps[tid].to(dtype=pipe.torch_dtype, device=pipe.device)
                kw["latents"] = pipe.scheduler.add_noise(x0, noise, t)
                target = pipe.scheduler.training_target(x0, noise, t)
                if "first_frame_latents" in kw:
                    kw["latents"][:, :, 0:1] = kw["first_frame_latents"]
                pred = pipe.model_fn(**models, **kw, timestep=t)
                if "first_frame_latents" in kw:
                    pred, target = pred[:, :, 1:], target[:, :, 1:]
                losses.append(torch.nn.functional.mse_loss(pred.float(), target.float()).item())
        ds = r["video"].split("/")[2].rsplit("_5s", 1)[0]
        per_clip.append({"video": r["video"], "dataset": ds, "loss_t": losses,
                         "loss": sum(losses) / len(losses)})
        if (i + 1) % 10 == 0:
            m = sum(c["loss"] for c in per_clip) / len(per_clip)
            print(f"[{i + 1}/{len(rows)}] mean={m:.4f} {(time.time() - t0) / (i + 1):.1f}s/clip", flush=True)

    by_ds = {}
    for c in per_clip:
        by_ds.setdefault(c["dataset"], []).append(c["loss"])
    summary = {
        "ckpt": a.ckpt, "n": len(per_clip), "num_frames": a.num_frames,
        "t_fracs": T_FRACS, "t_ids": t_ids,
        "mean_loss": sum(c["loss"] for c in per_clip) / len(per_clip),
        "mean_loss_per_t": [sum(c["loss_t"][k] for c in per_clip) / len(per_clip) for k in range(len(t_ids))],
        "per_dataset": {d: sum(v) / len(v) for d, v in sorted(by_ds.items())},
        "clips": per_clip,
    }
    json.dump(summary, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "clips"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
