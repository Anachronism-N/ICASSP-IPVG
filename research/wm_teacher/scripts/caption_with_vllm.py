#!/usr/bin/env python3
"""
Caption videos with Qwen3-VL-235B-A22B-Instruct-FP8 via vLLM OpenAI API.

Models on scripts/caption_with_transformers.py (preserves all manifest fields,
supports resume) but replaces the inference backend with a vLLM HTTP API call.

Targets:
  - DexYCB (hand-object interaction, resume from 281 existing Qwen2.5-VL captions)
  - H2O (hand-object interaction, fresh)
  - OpenVidHD (generic video description, fresh)

Output: <dataset>_processed/manifest_captioned.jsonl with all original fields
plus `caption` and `annotation_source`.

Usage:
    # Run all three datasets:
    python3 scripts/caption_with_vllm.py --datasets dexycb h2o openvidhd

    # Single dataset with higher concurrency:
    python3 scripts/caption_with_vllm.py --datasets openvidhd --concurrent 48

    # Limit (smoke test):
    python3 scripts/caption_with_vllm.py --datasets dexycb --limit 10
"""

import argparse
import base64
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

# Frame extraction via PyAV (avoids dependency on ffmpeg/ffprobe binaries in PATH).
import av

PROJECT_ROOT = Path("/srv/wm_teacher")
WT_DIR = PROJECT_ROOT / "dataset" / "wan22_training"

# vLLM endpoint (single replica, TP=8 on port 8000)
API_URL = "http://localhost:8000/v1/chat/completions"
# vLLM serves the model under the local dir path (not the HF repo ID).
MODEL_NAME = "/srv/wm_teacher/models/Qwen3-VL-235B-A22B-Instruct-FP8"
# Bypass Squid proxy for localhost requests (http_proxy is set in the env).
NO_PROXY_DICT = {"http": None, "https": None}

NUM_FRAMES = 8
ANNOTATION_SOURCE = "qwen3-vl-235b-a22b-fp8"

# Prompt strategies
HAND_OBJECT_PROMPT = (
    "You are a video captioning expert. Describe this hand-object interaction video "
    "in 1-3 concise English sentences. Focus on: (1) what the hand(s) are doing, "
    "(2) the object being manipulated, (3) the action sequence and visible result. "
    "Be factual and specific. Do not mention the video ID or filename."
)

GENERIC_VIDEO_PROMPT = (
    "You are a video captioning expert. Describe the video in natural English in 2-4 sentences. "
    "Focus on:\n"
    "1. What is happening (actions, interactions)\n"
    "2. Who/what is in the scene (people, objects)\n"
    "3. The setting/environment\n"
    "4. Any notable details (camera movement, lighting, mood)\n\n"
    "Keep it concise and factual. Do NOT start with \"The video shows\" - just describe directly."
)

GAMING_PROMPT = (
    "You are a video captioning expert. Describe this gameplay video in 2-4 concise English sentences. "
    "Focus on: (1) the game scene/setting, (2) what is happening on screen (driving, combat, "
    "exploration, UI interactions), (3) camera perspective (first-person, third-person), "
    "(4) notable visual details (lighting, weather, HUD elements). Be factual and specific."
)

DATASET_CONFIGS = {
    "dexycb": {
        "manifest": WT_DIR / "dexycb_processed" / "manifest.jsonl",
        "output": WT_DIR / "dexycb_processed" / "manifest_captioned.jsonl",
        "video_field": "video_path",
        "source_id_field": "session",
        "prompt": HAND_OBJECT_PROMPT,
    },
    "h2o": {
        "manifest": WT_DIR / "h2o_processed" / "manifest.jsonl",
        "output": WT_DIR / "h2o_processed" / "manifest_captioned.jsonl",
        "video_field": "video_path",
        "source_id_field": "frames_source",
        "prompt": HAND_OBJECT_PROMPT,
    },
    "openvidhd": {
        "manifest": WT_DIR / "openvidhd_processed" / "manifest.jsonl",
        "output": WT_DIR / "openvidhd_processed" / "manifest_captioned.jsonl",
        "video_field": "video_path",
        "source_id_field": "youtube_id",
        "prompt": GENERIC_VIDEO_PROMPT,
    },
    "matrix": {
        "manifest": WT_DIR / "matrix_extracted" / "manifest.jsonl",
        "output": WT_DIR / "matrix_extracted" / "manifest_captioned.jsonl",
        "video_field": "video_path",
        "source_id_field": "original_member",
        "prompt": GAMING_PROMPT,
    },
}


def sample_frames_base64(video_path, num_frames=NUM_FRAMES):
    """Sample N uniformly-spaced frames from a video, return as base64 JPEG list.

    Uses PyAV (no external ffmpeg/ffprobe binaries required).
    """
    try:
        container = av.open(video_path)
    except av.error.InvalidDataError:
        return []

    stream = container.streams.video[0]
    duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
    if duration <= 0:
        # Fall back to container duration
        if container.duration:
            duration = float(container.duration) / 1_000_000  # AV_TIME_BASE units
    if duration <= 0:
        duration = 5.0

    timestamps = [i * duration / (num_frames + 1) for i in range(1, num_frames + 1)]
    frames = []
    seen_pts = set()
    target_idx = 0

    # Seek approximately then decode linearly to collect the target timestamps.
    for t in timestamps:
        try:
            # Seek to slightly before the target (keyframe boundary).
            seek_ts = int(max(0.0, t - 0.5) / stream.time_base)
            container.seek(seek_ts, stream=stream)
        except Exception:
            pass
        best_frame = None
        best_diff = float("inf")
        for frame in container.decode(video=0):
            frame_t = float(frame.pts * stream.time_base) if frame.pts is not None else 0.0
            diff = abs(frame_t - t)
            if diff < best_diff:
                best_diff = diff
                best_frame = frame
            if frame_t >= t:
                break
        if best_frame is not None:
            # PyAV: convert to RGB ndarray then to JPEG via PIL.
            import numpy as np
            from PIL import Image
            arr = best_frame.to_ndarray(format="rgb24")
            img = Image.fromarray(arr)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            frames.append(base64.b64encode(buf.getvalue()).decode())
        if len(frames) >= num_frames:
            break

    container.close()
    return frames


def call_vllm(frames_b64, prompt, api_url=None, model=None, timeout=120):
    """Call vLLM OpenAI-compatible chat completions API with images + text."""
    if api_url is None:
        api_url = API_URL
    if model is None:
        model = MODEL_NAME
    content = []
    for fb64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{fb64}"},
        })
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 256,
        "temperature": 0.3,
    }

    try:
        r = requests.post(api_url, json=payload, timeout=timeout, proxies=NO_PROXY_DICT)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        # Non-200: log and return None
        print(f"    API error {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"    API exception: {e}")
    return None


def load_done(output_path):
    """Load set of already-captioned video_paths from output file (resume)."""
    done = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done.add(d["video_path"])
                except Exception:
                    pass
    return done


def caption_dataset(cfg, limit=0, concurrent=32, num_frames=NUM_FRAMES,
                    api_url=None, model=None):
    """Caption one dataset. Reads manifest, writes manifest_captioned.jsonl."""
    name = cfg.get("_name", "?")
    in_path = cfg["manifest"]
    out_path = cfg["output"]
    vfield = cfg["video_field"]
    prompt = cfg["prompt"]

    if not in_path.exists():
        print(f"\n[{name}] SKIP: manifest not found {in_path}")
        return

    rows = [json.loads(l) for l in open(in_path, encoding="utf-8") if l.strip()]
    if limit > 0:
        rows = rows[:limit]

    done = load_done(out_path)
    todo = [r for r in rows if r.get(vfield) not in done]
    print(f"\n[{name}] total={len(rows)} done={len(done)} todo={len(todo)} -> {out_path.name}")

    if not todo:
        print(f"[{name}] all done.")
        return

    ok = 0
    fail = 0

    def process_one(row):
        vp = row.get(vfield)
        if not vp or not os.path.exists(vp):
            return ("fail", row, "video missing")
        try:
            frames = sample_frames_base64(vp, num_frames)
            if not frames:
                return ("fail", row, "no frames")
            cap = call_vllm(frames, prompt, api_url=api_url, model=model)
            if not cap:
                return ("fail", row, "no caption")
            return ("ok", row, cap)
        except Exception as e:
            return ("fail", row, str(e))

    with ThreadPoolExecutor(max_workers=concurrent) as pool:
        futures = {pool.submit(process_one, r): r for r in todo}
        with open(out_path, "a", encoding="utf-8") as fout:
            with tqdm(total=len(todo), desc=name, unit="vid") as pbar:
                for fut in as_completed(futures):
                    status, row, info = fut.result()
                    if status == "ok":
                        out_row = dict(row)
                        out_row["caption"] = info
                        out_row["annotation_source"] = ANNOTATION_SOURCE
                        fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                        fout.flush()
                        ok += 1
                    else:
                        fail += 1
                        if fail <= 5 or fail % 20 == 0:
                            print(f"  FAIL ({info}): {Path(row.get(vfield, '?')).name}")
                    pbar.update(1)
                    pbar.set_postfix(ok=ok, fail=fail)

    print(f"[{name}] done: ok={ok} fail={fail} (total captioned now: {ok + len(done)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["dexycb", "h2o", "openvidhd", "matrix"],
                    choices=list(DATASET_CONFIGS.keys()))
    ap.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    ap.add_argument("--concurrent", type=int, default=32,
                    help="Concurrent API requests (single vLLM replica batches server-side)")
    ap.add_argument("--limit", type=int, default=0, help="Limit per dataset (0 = all)")
    ap.add_argument("--api-url", default=API_URL)
    ap.add_argument("--model", default=MODEL_NAME)
    args = ap.parse_args()

    # Health check
    print(f"Checking vLLM at {args.api_url} ...")
    try:
        # vLLM health endpoint is at /health (not /v1/health).
        # Strip /v1/chat/completions from the API URL to get the base URL.
        base_url = args.api_url.rsplit("/", 3)[0]  # http://localhost:8000
        health_url = base_url + "/health"
        r = requests.get(health_url, timeout=10, proxies=NO_PROXY_DICT)
        if r.status_code != 200:
            raise SystemExit(f"vLLM health check failed: {r.status_code}. Is the server running?")
        print("  vLLM healthy.")
    except requests.ConnectionError:
        raise SystemExit(f"Cannot reach vLLM at {args.api_url}. Start it with deploy_qwen3vl.sh first.")

    for ds_name in args.datasets:
        cfg = dict(DATASET_CONFIGS[ds_name])
        cfg["_name"] = ds_name
        caption_dataset(cfg, limit=args.limit, concurrent=args.concurrent,
                        num_frames=args.num_frames,
                        api_url=args.api_url, model=args.model)

    print("\nAll captioning complete.")


if __name__ == "__main__":
    main()
