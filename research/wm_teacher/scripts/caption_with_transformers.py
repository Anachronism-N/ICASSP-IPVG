#!/usr/bin/env python3
"""
Caption videos with Qwen2.5-VL-7B via transformers (no vLLM required).

Targets DexYCB (2,400) + H2O (60): hand-object interaction videos that
currently lack captions. Produces an augmented manifest with a `caption`
field, suitable for inclusion in the unified training manifest.

Model: Qwen2.5-VL-7B-Instruct (~16GB bf16, fits a single A100-40GB).
Sampling: 8 frames per video, uniformly spaced.
Output: dataset/wan22_training/<name>_processed/manifest_captioned.jsonl

Usage:
    python3 scripts/caption_with_transformers.py --datasets dexycb h2o
    python3 scripts/caption_with_transformers.py --datasets dexycb --batch-size 8
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

PROJECT_ROOT = Path("/srv/wm_teacher")
MODEL_PATH = PROJECT_ROOT / "models" / "Qwen2.5-VL-7B-Instruct"

DATASET_CONFIGS = {
    "dexycb": {
        "manifest": PROJECT_ROOT / "dataset" / "wan22_training" / "dexycb_processed" / "manifest.jsonl",
        "output": PROJECT_ROOT / "dataset" / "wan22_training" / "dexycb_processed" / "manifest_captioned.jsonl",
        "video_field": "video_path",
        "source_id_field": "session",
    },
    "h2o": {
        "manifest": PROJECT_ROOT / "dataset" / "wan22_training" / "h2o_processed" / "manifest.jsonl",
        "output": PROJECT_ROOT / "dataset" / "wan22_training" / "h2o_processed" / "manifest_captioned.jsonl",
        "video_field": "video_path",
        "source_id_field": "frames_source",
    },
}

PROMPT = (
    "You are a video captioning expert. Describe this hand-object interaction video "
    "in 1-3 concise English sentences. Focus on: (1) what the hand(s) are doing, "
    "(2) the object being manipulated, (3) the action sequence and visible result. "
    "Be factual and specific. Do not mention the video ID or filename."
)

NUM_FRAMES = 8


def sample_frames(video_path, num_frames=8):
    """Sample N uniformly-spaced frames from a video. Returns list of PIL Images."""
    import os
    try:
        from PIL import Image
    except ImportError:
        return []

    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True, timeout=15,
    )
    try:
        dur = float(r.stdout.strip()) if r.stdout.strip() else 5.0
    except ValueError:
        dur = 5.0
    if dur <= 0:
        dur = 5.0

    timestamps = [i * dur / (num_frames + 1) for i in range(1, num_frames + 1)]
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, t in enumerate(timestamps):
            out = f"{tmp}/f_{i:03d}.jpg"
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-ss", str(t), "-i", video_path,
                 "-vframes", "1", "-q:v", "2", out],
                capture_output=True, timeout=15,
            )
            if os.path.exists(out) and os.path.getsize(out) > 0:
                try:
                    frames.append(Image.open(out).convert("RGB"))
                except Exception:
                    pass
    return frames


def build_messages(frames):
    """Build Qwen2.5-VL chat messages with N frames + text prompt."""
    content = []
    for img in frames:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": PROMPT})
    return [{"role": "user", "content": content}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["dexycb", "h2o"],
                    choices=list(DATASET_CONFIGS.keys()))
    ap.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="Limit per dataset (0 = all)")
    ap.add_argument("--model-path", type=Path, default=MODEL_PATH,
                    help="Path to Qwen2.5-VL model (use /dev/shm copy for fast loading)")
    args = ap.parse_args()

    if not args.model_path.exists():
        raise SystemExit(f"Model not found at {args.model_path}. Download it first:\n"
                         f"  modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct "
                         f"--local_dir {MODEL_PATH}")

    print(f"Loading model from {args.model_path} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(str(args.model_path))
    print(f"Model loaded. Device: {next(model.parameters()).device}")

    for ds_name in args.datasets:
        cfg = DATASET_CONFIGS[ds_name]
        in_path = cfg["manifest"]
        out_path = cfg["output"]
        vfield = cfg["video_field"]
        sid_field = cfg["source_id_field"]

        if not in_path.exists():
            print(f"\n[{ds_name}] SKIP: manifest not found {in_path}")
            continue

        rows = [json.loads(l) for l in open(in_path, encoding="utf-8") if l.strip()]
        if args.limit > 0:
            rows = rows[: args.limit]
        print(f"\n[{ds_name}] {len(rows)} videos to caption -> {out_path.name}")

        # Resume: load already-captioned entries.
        done = set()
        if out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        done.add(d["video_path"])
                    except Exception:
                        pass
            print(f"  resume: {len(done)} already captioned")

        todo = [r for r in rows if r.get(vfield) not in done]
        print(f"  todo: {len(todo)}")

        ok = 0
        fail = 0
        with open(out_path, "a", encoding="utf-8") as fout:
            for i, r in enumerate(todo, 1):
                vp = r.get(vfield)
                if not vp or not os.path.exists(vp):
                    fail += 1
                    continue
                frames = sample_frames(vp, args.num_frames)
                if not frames:
                    print(f"  [{i}/{len(todo)}] FAIL no frames: {Path(vp).name}")
                    fail += 1
                    continue

                messages = build_messages(frames)
                try:
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=text, images=frames, return_tensors="pt", padding=True)
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
                    # Strip input tokens.
                    gen = out_ids[:, inputs["input_ids"].shape[1]:]
                    caption = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
                except Exception as e:
                    print(f"  [{i}/{len(todo)}] FAIL generate: {e}")
                    fail += 1
                    continue

                if not caption:
                    fail += 1
                    continue

                out_row = dict(r)
                out_row["caption"] = caption
                out_row["annotation_source"] = "qwen2.5-vl-7b"
                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                fout.flush()
                ok += 1
                if i % 20 == 0 or i == len(todo):
                    print(f"  [{i}/{len(todo)}] ok={ok} fail={fail} last: {caption[:80]}")

        print(f"[{ds_name}] done: ok={ok} fail={fail} (total captioned now: {ok + len(done)})")

    print("\nAll captioning complete.")


if __name__ == "__main__":
    main()
