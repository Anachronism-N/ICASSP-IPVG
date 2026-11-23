#!/usr/bin/env python3
"""
Re-caption unified manifest videos with Qwen3-VL-235B-A22B-Instruct-FP8 via vLLM.

Reads unified_train.jsonl / unified_val.jsonl, filters by source_dataset, and
re-captions each clip's video_path with a unified training-oriented prompt.
Supports multiple vLLM endpoints (load-balanced round-robin), resume, and
PyAV frame sampling (no ffmpeg/ffprobe dependency).

Output: <output_dir>/<dataset>_recaptioned.jsonl with fields:
  {video_path, source_dataset, source_sequence_id, prompt (new),
   annotation_source, model}

A separate merge step (merge_recaptioned.py) updates unified_train/val.jsonl
in place, preserving the old prompt as prompt_legacy.

Usage:
    # Stage 1: re-caption 4 low-quality datasets in unified_train + val
    python3 scripts/recaption_unified.py \\
        --manifest training_metadata/unified_train.jsonl \\
                   training_metadata/unified_val.jsonl \\
        --datasets mira charades noxi easycom \\
        --output-dir training_metadata/recaptioned \\
        --api-urls http://PRIVATE_NODE_ADDRESS:8000/v1/chat/completions \\
                   http://PRIVATE_NODE_ADDRESS:8000/v1/chat/completions \\
        --concurrent 64

    # Smoke test (10 per dataset)
    python3 scripts/recaption_unified.py ... --limit 10
"""

import argparse
import base64
import io
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

# Frame extraction via PyAV (avoids dependency on ffmpeg/ffprobe binaries).
import av

PROJECT_ROOT = Path("/srv/wm_teacher")
MODEL_NAME = "/srv/wm_teacher/models/Qwen3-VL-235B-A22B-Instruct-FP8"
ANNOTATION_SOURCE = "qwen3-vl-235b-a22b-fp8-recaption"
NUM_FRAMES = 8

# Bypass Squid proxy for intranet requests (http_proxy is set in the env).
NO_PROXY_DICT = {"http": None, "https": None}

# ---------------------------------------------------------------------------
# Unified training-oriented prompt
# ---------------------------------------------------------------------------
# Designed for world-model video generation training: emphasizes visible scene
# content, subjects, actions, interactions, environment and camera/lighting so
# the caption can serve as a strong text condition for a video diffusion model.
# Cross-dataset consistent (gameplay / daily activity / conversation / HOI all
# use the same prompt) to avoid distribution shifts in the prompt embedding.
UNIFIED_PROMPT = """You are an expert video describer for a world-model training pipeline. Carefully observe the sampled frames and write a precise, information-dense English description of this video clip in 3-5 sentences (about 60-120 words).

Cover these aspects:
1. Subjects: who or what is visible — people (appearance, clothing, posture), objects, characters, vehicles, or in-game elements. Note their positions and spatial relations.
2. Action: what is happening — movements, manipulations, gestures, interactions between subjects and objects, gameplay events, or conversational cues.
3. Environment: the setting and background — room, outdoor scene, game arena, surfaces, props, furniture.
4. Camera & visuals: perspective (first-person / third-person / overhead), framing, lighting, colors, weather, mood. For gameplay, note HUD/UI elements if visible.

Rules:
- Describe ONLY what is visible in the frames. Be factual and specific.
- Use concrete details: colors, sizes, positions, directions, speeds.
- Do NOT start with "The video shows" or "This video" — describe the scene directly.
- Do NOT mention filenames, video IDs, frame counts, or that this is a caption.
- Do NOT speculate beyond what is visible.
- Keep it to 3-5 concise but complete sentences."""


# ---------------------------------------------------------------------------
# Frame sampling (PyAV) — reused from caption_with_vllm.py
# ---------------------------------------------------------------------------
def sample_frames_base64(video_path, num_frames=NUM_FRAMES):
    """Sample N uniformly-spaced frames from a video, return as base64 JPEG list.

    Uses PyAV (no external ffmpeg/ffprobe binaries required).
    """
    try:
        container = av.open(video_path)
    except Exception:
        return []

    stream = container.streams.video[0]
    duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
    if duration <= 0:
        if container.duration:
            duration = float(container.duration) / 1_000_000  # AV_TIME_BASE units
    if duration <= 0:
        duration = 5.0

    timestamps = [i * duration / (num_frames + 1) for i in range(1, num_frames + 1)]
    frames = []

    for t in timestamps:
        try:
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


# ---------------------------------------------------------------------------
# Multi-endpoint vLLM client (thread-safe round-robin)
# ---------------------------------------------------------------------------
class EndpointPool:
    """Thread-safe round-robin over multiple vLLM endpoints."""

    def __init__(self, api_urls):
        self.urls = list(api_urls)
        self.idx = 0
        self.lock = threading.Lock()
        if not self.urls:
            raise ValueError("need at least one api url")

    def next(self):
        with self.lock:
            url = self.urls[self.idx % len(self.urls)]
            self.idx += 1
            return url


def call_vllm(frames_b64, prompt, endpoint_pool, model=None, timeout=150):
    """Call a vLLM OpenAI-compatible chat completions endpoint.

    Tries endpoints round-robin; on failure, retries the next endpoint once.
    """
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
        "max_tokens": 320,
        "temperature": 0.3,
    }

    tried = set()
    for _ in range(len(endpoint_pool.urls)):
        url = endpoint_pool.next()
        if url in tried:
            break
        tried.add(url)
        try:
            r = requests.post(url, json=payload, timeout=timeout, proxies=NO_PROXY_DICT)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def resolve_video_path(raw, project_root=PROJECT_ROOT):
    """Resolve a video_path that may be absolute or relative to PROJECT_ROOT."""
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute():
        return str(p) if p.exists() else None
    # Relative: try project_root / raw
    cand = project_root / raw
    if cand.exists():
        return str(cand)
    return None


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Main caption routine
# ---------------------------------------------------------------------------
def caption_subset(rows, dataset_name, output_path, endpoint_pool,
                   concurrent=64, num_frames=NUM_FRAMES, limit=0):
    """Caption a list of unified-manifest rows for one dataset."""
    done = load_done(output_path)
    todo = [r for r in rows if r.get("video_path") not in done]
    if limit > 0:
        todo = todo[:limit]

    print(f"\n[{dataset_name}] total={len(rows)} done={len(done)} todo={len(todo)} -> {output_path.name}")
    if not todo:
        print(f"[{dataset_name}] all done.")
        return 0, 0

    ok = 0
    fail = 0
    fail_lock = threading.Lock()

    def process_one(row):
        vp_raw = row.get("video_path")
        vp = resolve_video_path(vp_raw)
        if not vp:
            return ("fail", row, "video missing/unresolvable")
        # Fast pre-check: skip empty/corrupt files without opening them
        try:
            if os.path.getsize(vp) < 1024:
                return ("fail", row, f"empty file ({os.path.getsize(vp)}B)")
        except OSError:
            return ("fail", row, "stat failed")
        try:
            frames = sample_frames_base64(vp, num_frames)
            if not frames:
                return ("fail", row, "no frames")
            cap = call_vllm(frames, UNIFIED_PROMPT, endpoint_pool)
            if not cap:
                return ("fail", row, "no caption")
            return ("ok", row, cap)
        except Exception as e:
            return ("fail", row, str(e))

    with ThreadPoolExecutor(max_workers=concurrent) as pool:
        futures = {pool.submit(process_one, r): r for r in todo}
        with open(output_path, "a", encoding="utf-8") as fout:
            with tqdm(total=len(todo), desc=dataset_name, unit="vid") as pbar:
                for fut in as_completed(futures):
                    status, row, info = fut.result()
                    if status == "ok":
                        out_row = {
                            "video_path": row.get("video_path"),
                            "source_dataset": row.get("source_dataset"),
                            "source_sequence_id": row.get("source_sequence_id"),
                            "prompt": info,
                            "annotation_source": ANNOTATION_SOURCE,
                            "model": MODEL_NAME,
                        }
                        fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                        fout.flush()
                        ok += 1
                    else:
                        with fail_lock:
                            fail += 1
                            if fail <= 5 or fail % 20 == 0:
                                print(f"  FAIL ({info}): {row.get('video_path', '?')}")
                    pbar.update(1)
                    pbar.set_postfix(ok=ok, fail=fail)

    print(f"[{dataset_name}] done: ok={ok} fail={fail} (total captioned now: {ok + len(done)})")
    return ok, fail


def main():
    global MODEL_NAME
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", nargs="+", required=True,
                    help="Path(s) to unified manifest jsonl (train and/or val)")
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="source_dataset values to re-caption (e.g. mira charades noxi easycom)")
    ap.add_argument("--output-dir", default="training_metadata/recaptioned",
                    help="Output directory for per-dataset <ds>_recaptioned.jsonl")
    ap.add_argument("--api-urls", nargs="+", required=True,
                    help="vLLM chat completions endpoint URLs (one per replica)")
    ap.add_argument("--concurrent", type=int, default=64,
                    help="Concurrent API requests across all endpoints")
    ap.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    ap.add_argument("--limit", type=int, default=0,
                    help="Limit per dataset (0 = all). Useful for smoke tests.")
    ap.add_argument("--model", default=MODEL_NAME,
                    help="Model name served by vLLM (defaults to local FP8 path)")
    args = ap.parse_args()

    MODEL_NAME = args.model

    endpoint_pool = EndpointPool(args.api_urls)
    print(f"Endpoints: {endpoint_pool.urls}")
    print(f"Datasets: {args.datasets}")
    print(f"Output dir: {args.output_dir}")

    # Health check all endpoints
    for url in endpoint_pool.urls:
        base = url.rsplit("/", 3)[0]  # http://host:port
        health = base + "/health"
        try:
            r = requests.get(health, timeout=10, proxies=NO_PROXY_DICT)
            status = "healthy" if r.status_code == 200 else f"HTTP {r.status_code}"
        except Exception as e:
            status = f"UNREACHABLE ({e})"
        print(f"  {health} -> {status}")

    # Load + filter manifests
    target = set(args.datasets)
    by_ds = {}
    for mp in args.manifest:
        mp = Path(mp)
        if not mp.is_absolute():
            mp = PROJECT_ROOT / mp
        if not mp.exists():
            print(f"WARN: manifest not found: {mp}")
            continue
        with open(mp, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                ds = r.get("source_dataset")
                if ds in target:
                    by_ds.setdefault(ds, []).append(r)

    print("\nFiltered counts:")
    for ds in args.datasets:
        print(f"  {ds}: {len(by_ds.get(ds, []))}")

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    total_ok = total_fail = 0
    for ds in args.datasets:
        rows = by_ds.get(ds, [])
        if not rows:
            print(f"\n[{ds}] no rows, skip")
            continue
        out_path = out_dir / f"{ds}_recaptioned.jsonl"
        ok, fail = caption_subset(rows, ds, out_path, endpoint_pool,
                                  concurrent=args.concurrent,
                                  num_frames=args.num_frames,
                                  limit=args.limit)
        total_ok += ok
        total_fail += fail

    print(f"\nAll done. ok={total_ok} fail={total_fail}")
    print(f"Output dir: {out_dir}")
    print("Next: merge back with scripts/merge_recaptioned.py")


if __name__ == "__main__":
    main()
