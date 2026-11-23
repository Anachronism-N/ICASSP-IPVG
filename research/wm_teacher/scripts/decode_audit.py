#!/usr/bin/env python3
"""
Decode audit on unified manifest.

Randomly samples N videos from the unified train manifest, verifies:
  - ffprobe can decode the file (container readable)
  - reports width/height/fps/duration
  - first frame can be extracted (required for Wan2.2-TI2V input_image)

Outputs a JSON report at training_metadata/decode_audit_report.json.

Usage:
    python3 scripts/decode_audit.py --n 500
    python3 scripts/decode_audit.py --n 500 --manifest training_metadata/unified_train.jsonl
"""

import argparse
import json
import os
import random
import subprocess
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path("/srv/wm_teacher")


def ffprobe_video(path):
    """Return dict with width/height/fps/duration or None if undecodable."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate,duration",
             "-show_entries", "format=duration",
             "-of", "json", path],
            capture_output=True, text=True, timeout=20,
        )
        if r.returncode != 0:
            return None
        d = json.loads(r.stdout)
        s = (d.get("streams") or [{}])[0]
        fps_num, fps_den = (s.get("r_frame_rate") or "0/1").split("/")
        fps = float(fps_num) / float(fps_den) if float(fps_den) else 0.0
        dur = float(s.get("duration") or d.get("format", {}).get("duration") or 0)
        return {
            "width": s.get("width"),
            "height": s.get("height"),
            "fps": round(fps, 2),
            "duration": round(dur, 2),
        }
    except Exception:
        return None


def extract_first_frame(path, out_path):
    """Try to extract the first frame. Returns True on success."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", path,
             "-vframes", "1", "-q:v", "2", out_path],
            capture_output=True, text=True, timeout=20,
        )
        return r.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="Number of samples to audit")
    ap.add_argument("--manifest", type=Path,
                    default=PROJECT_ROOT / "training_metadata" / "unified_train.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "training_metadata" / "decode_audit_report.json")
    args = ap.parse_args()

    rows = []
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    sample = rows[: args.n]
    print(f"Auditing {len(sample)} of {len(rows)} videos from {args.manifest.name}")

    results = []
    by_source = Counter()
    ok_decode = 0
    ok_first_frame = 0
    missing_file = 0
    fail_decode = 0
    fail_first_frame = 0

    tmp_frame = "/tmp/_audit_frame.jpg"
    for i, r in enumerate(sample, 1):
        src = r.get("source_dataset", "?")
        path = r["video_path"]
        by_source[src] += 1

        if not os.path.exists(path):
            missing_file += 1
            results.append({"path": path, "source": src, "status": "missing_file"})
            if i % 50 == 0:
                print(f"  [{i}/{len(sample)}] missing={missing_file} ok_decode={ok_decode}")
            continue

        info = ffprobe_video(path)
        if info is None:
            fail_decode += 1
            results.append({"path": path, "source": src, "status": "fail_decode"})
        else:
            ok_decode += 1
            ff_ok = extract_first_frame(path, tmp_frame)
            if ff_ok:
                ok_first_frame += 1
                status = "ok"
            else:
                fail_first_frame += 1
                status = "fail_first_frame"
            results.append({
                "path": path, "source": src, "status": status,
                "width": info["width"], "height": info["height"],
                "fps": info["fps"], "duration": info["duration"],
            })

        if i % 50 == 0:
            print(f"  [{i}/{len(sample)}] ok={ok_decode} ff_ok={ok_first_frame} "
                  f"missing={missing_file} fail_dec={fail_decode} fail_ff={fail_first_frame}")

    if os.path.exists(tmp_frame):
        os.remove(tmp_frame)

    total = len(sample)
    report = {
        "total_audited": total,
        "ok_decode": ok_decode,
        "ok_first_frame": ok_first_frame,
        "missing_file": missing_file,
        "fail_decode": fail_decode,
        "fail_first_frame": fail_first_frame,
        "decode_rate": round(ok_decode / total, 4) if total else 0,
        "first_frame_rate": round(ok_first_frame / total, 4) if total else 0,
        "by_source": dict(by_source),
        "results": results,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Decode Audit Report ===")
    print(f"Total audited : {total}")
    print(f"OK decode     : {ok_decode} ({report['decode_rate']*100:.1f}%)")
    print(f"OK first frame: {ok_first_frame} ({report['first_frame_rate']*100:.1f}%)")
    print(f"Missing file  : {missing_file}")
    print(f"Fail decode   : {fail_decode}")
    print(f"Fail 1st frame: {fail_first_frame}")
    print(f"By source     : {dict(by_source)}")
    print(f"Report saved  : {args.out}")


if __name__ == "__main__":
    main()
