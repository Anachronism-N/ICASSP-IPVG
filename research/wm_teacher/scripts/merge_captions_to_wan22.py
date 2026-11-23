#!/usr/bin/env python3
"""
Merge existing captions into *_wan22/manifest_wan22.jsonl files.

The unified manifest builder reads from *_wan22/manifest_wan22.jsonl where the
`prompt` field is empty for most datasets. This script fills in the prompt by
joining with caption sources:

  - dexycb:   1:1 on video_path           (manifest_captioned.jsonl)
  - h2o:      many-to-1 via source_seq_id (strip "h2o:" prefix -> frames_source)
  - openvidhd: 1:1 on video_path          (manifest_captioned.jsonl, but no wan22 dir — skip, handled separately)
  - charades: many-to-1 via source_id     (strip "charades:" prefix -> video_id)
  - noxi:     many-to-1 via source_id     (strip "noxi:" prefix -> session_id stem)
  - easycom:  many-to-1 via source_id     (strip "easycom:" prefix -> video_id stem)
  - hoigen1m: from CSV                    (prompt field already filled? check)
  - matrix:   from manifest_captioned     (after VLM captioning completes)

Usage:
    python3 scripts/merge_captions_to_wan22.py
    python3 scripts/merge_captions_to_wan22.py --datasets dexycb h2o --dry-run
"""

import argparse
import csv
import json
import os
from pathlib import Path

PROJECT_ROOT = Path("/srv/wm_teacher")
WT_DIR = PROJECT_ROOT / "dataset" / "wan22_training"


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def strip_prefix(s, prefix):
    """Strip 'prefix' from start of s if present."""
    return s[len(prefix):] if s.startswith(prefix) else s


def merge_by_video_path(wan22_rows, captioned_rows, caption_field="caption"):
    """1:1 merge on video_path. Returns count of prompts filled."""
    cap_map = {}
    for r in captioned_rows:
        vp = r.get("video_path")
        if vp:
            cap_map[vp] = r.get(caption_field) or r.get("caption") or ""
    filled = 0
    for r in wan22_rows:
        vp = r.get("video_path")
        # Handle relative paths — try both relative and absolute
        cap = cap_map.get(vp) or cap_map.get(str(PROJECT_ROOT / vp)) or cap_map.get(os.path.abspath(vp))
        if cap and not r.get("prompt"):
            r["prompt"] = cap
            filled += 1
    return filled


def merge_by_source_id(wan22_rows, captioned_rows, wan22_prefix, captioned_id_field, caption_field="caption"):
    """Many-to-1 merge: strip wan22_prefix from source_sequence_id to get captioned_id_field value."""
    cap_map = {}
    for r in captioned_rows:
        sid = r.get(captioned_id_field)
        if sid:
            cap_map[sid] = r.get(caption_field) or r.get("caption") or ""
    filled = 0
    for r in wan22_rows:
        sid = r.get("source_sequence_id", "")
        if sid:
            stem = strip_prefix(sid, wan22_prefix + ":")
            cap = cap_map.get(stem)
            if cap and not r.get("prompt"):
                r["prompt"] = cap
                filled += 1
    return filled


def merge_hoigen1m_from_csv(wan22_rows):
    """HOIGen-1M captions come from a CSV (caption_info column)."""
    csv_path = PROJECT_ROOT / "training_metadata" / "hoigen1m_metadata.csv"
    if not csv_path.exists():
        print("  hoigen1m: CSV not found, skipping")
        return 0
    # Build video_path -> caption map from CSV
    cap_map = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # CSV columns: video_path, caption_info (adjust if different)
            vp = row.get("video_path") or row.get("path") or row.get("video")
            cap = row.get("caption_info") or row.get("caption") or row.get("prompt")
            if vp and cap:
                cap_map[vp] = cap
    print(f"  hoigen1m: loaded {len(cap_map)} captions from CSV")
    filled = 0
    for r in wan22_rows:
        vp = r.get("video_path", "")
        # Try matching by stem (CSV may have different path format)
        stem = Path(vp).stem
        cap = cap_map.get(vp) or cap_map.get(stem)
        if cap and not r.get("prompt"):
            r["prompt"] = cap
            filled += 1
    return filled


def process_dataset(name, wan22_manifest_path, merge_fn):
    """Process one dataset: load wan22, merge captions, write back."""
    if not wan22_manifest_path.exists():
        print(f"  {name}: SKIP — manifest not found at {wan22_manifest_path}")
        return 0
    rows = load_jsonl(wan22_manifest_path)
    empty_before = sum(1 for r in rows if not r.get("prompt", "").strip())
    filled = merge_fn(rows)
    empty_after = sum(1 for r in rows if not r.get("prompt", "").strip())
    print(f"  {name}: {len(rows)} rows, empty {empty_before} -> {empty_after}, filled {filled}")
    return rows, filled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["dexycb", "h2o", "charades", "noxi", "easycom", "hoigen1m", "matrix"],
                    help="Datasets to merge")
    ap.add_argument("--dry-run", action="store_true", help="Report stats only, don't write")
    args = ap.parse_args()

    results = {}

    if "dexycb" in args.datasets:
        print("\n[dexycb] 1:1 merge on video_path")
        wan22_path = WT_DIR / "dexycb_processed" / "manifest_wan22.jsonl"
        cap_path = WT_DIR / "dexycb_processed" / "manifest_captioned.jsonl"
        if wan22_path.exists() and cap_path.exists():
            rows = load_jsonl(wan22_path)
            caps = load_jsonl(cap_path)
            filled = merge_by_video_path(rows, caps)
            print(f"  dexycb: {len(rows)} rows, filled {filled}")
            if not args.dry_run:
                write_jsonl(rows, wan22_path)
            results["dexycb"] = filled

    if "h2o" in args.datasets:
        print("\n[h2o] many-to-1 merge via source_sequence_id -> frames_source")
        wan22_path = WT_DIR / "h2o_wan22" / "manifest_wan22.jsonl"
        cap_path = WT_DIR / "h2o_processed" / "manifest_captioned.jsonl"
        if wan22_path.exists() and cap_path.exists():
            rows = load_jsonl(wan22_path)
            caps = load_jsonl(cap_path)
            filled = merge_by_source_id(rows, caps, "h2o", "frames_source")
            print(f"  h2o: {len(rows)} rows, filled {filled}")
            if not args.dry_run:
                write_jsonl(rows, wan22_path)
            results["h2o"] = filled

    if "charades" in args.datasets:
        print("\n[charades] many-to-1 merge via source_sequence_id -> video_id")
        wan22_path = WT_DIR / "charades_wan22" / "manifest_wan22.jsonl"
        cap_path = WT_DIR / "charades_processed" / "manifest.jsonl"
        if wan22_path.exists() and cap_path.exists():
            rows = load_jsonl(wan22_path)
            caps = load_jsonl(cap_path)
            # charades source_id: "charades:001YG_1280x704_24fps" -> video_id "001YG"
            # but wan22 stem has resolution/fps suffix; charades_processed video = "001YG_1280x704_24fps.mp4"
            # and video_id = "001YG". Need to match on full stem (without "charades:" prefix).
            filled = merge_by_source_id(rows, caps, "charades", "video_id")
            # If that didn't fill many, try matching on the video filename stem
            if filled == 0:
                cap_map = {}
                for r in caps:
                    v = r.get("video", "")
                    if v:
                        stem = str(Path(v).stem)  # e.g. "001YG_1280x704_24fps"
                        cap_map[stem] = r.get("caption", "")
                filled = 0
                for r in rows:
                    sid = strip_prefix(r.get("source_sequence_id", ""), "charades:")
                    cap = cap_map.get(sid)
                    if cap and not r.get("prompt"):
                        r["prompt"] = cap
                        filled += 1
                print(f"  charades (fallback to full stem): filled {filled}")
            else:
                print(f"  charades: {len(rows)} rows, filled {filled}")
            if not args.dry_run:
                write_jsonl(rows, wan22_path)
            results["charades"] = filled

    if "noxi" in args.datasets:
        print("\n[noxi] many-to-1 merge via source_sequence_id -> session_id")
        wan22_path = WT_DIR / "noxi_wan22" / "manifest_wan22.jsonl"
        cap_path = WT_DIR / "noxi_processed" / "manifest.jsonl"
        if wan22_path.exists() and cap_path.exists():
            rows = load_jsonl(wan22_path)
            caps = load_jsonl(cap_path)
            # noxi source_id: "noxi:001_expert" -> session_id "001"
            # noxi_processed has session_id "001" but caption may be per-session
            # Try matching on session stem (strip _expert/_novice suffix)
            cap_map = {}
            for r in caps:
                sid = r.get("session_id", "")
                if sid:
                    cap_map[sid] = r.get("caption", "")
            filled = 0
            for r in rows:
                sid = strip_prefix(r.get("source_sequence_id", ""), "noxi:")
                # sid is like "001_expert" — strip the role suffix to get "001"
                stem = sid.split("_")[0] if "_" in sid else sid
                cap = cap_map.get(stem) or cap_map.get(sid)
                if cap and not r.get("prompt"):
                    r["prompt"] = cap
                    filled += 1
            print(f"  noxi: {len(rows)} rows, filled {filled}")
            if not args.dry_run:
                write_jsonl(rows, wan22_path)
            results["noxi"] = filled

    if "easycom" in args.datasets:
        print("\n[easycom] merge from CSV (match on filename stem)")
        wan22_path = WT_DIR / "easycom_wan22" / "manifest_wan22.jsonl"
        csv_path = PROJECT_ROOT / "training_metadata" / "easycom_metadata.csv"
        if wan22_path.exists() and csv_path.exists():
            rows = load_jsonl(wan22_path)
            cap_map = {}
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vp = row.get("video") or row.get("video_path")
                    cap = row.get("prompt") or row.get("caption")
                    if vp and cap:
                        stem = Path(vp).stem
                        cap_map[stem] = cap
            print(f"  loaded {len(cap_map)} captions from CSV")
            filled = 0
            for r in rows:
                sid = strip_prefix(r.get("source_sequence_id", ""), "easycom:")
                # sid is "00-00-000_1280x704_24fps" — matches CSV video stem
                cap = cap_map.get(sid)
                if cap and not r.get("prompt"):
                    r["prompt"] = cap
                    filled += 1
            print(f"  easycom: {len(rows)} rows, filled {filled}")
            if not args.dry_run:
                write_jsonl(rows, wan22_path)
            results["easycom"] = filled

    if "hoigen1m" in args.datasets:
        print("\n[hoigen1m] merge from CSV (match on filename stem)")
        wan22_path = WT_DIR / "hoigen1m_wan22" / "manifest_wan22.jsonl"
        if wan22_path.exists():
            rows = load_jsonl(wan22_path)
            csv_path = PROJECT_ROOT / "training_metadata" / "hoigen1m_metadata.csv"
            if csv_path.exists():
                cap_map = {}
                with open(csv_path, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    cols = reader.fieldnames
                    print(f"  CSV columns: {cols}")
                    for row in reader:
                        vp = row.get("video") or row.get("video_path") or row.get("path")
                        cap = row.get("prompt") or row.get("caption_info") or row.get("caption")
                        if vp and cap:
                            stem = Path(vp).stem  # e.g. "---dI8zrQJI-0:02:40.960-0:02:43.840"
                            cap_map[stem] = cap
                print(f"  loaded {len(cap_map)} captions from CSV")
                filled = 0
                for r in rows:
                    vp = r.get("video_path", "")
                    stem = Path(vp).stem
                    # wan22 clips have _NNN suffix (e.g. "video_000"); strip it to match CSV parent stem
                    parts = stem.rsplit("_", 1)
                    parent_stem = parts[0] if len(parts) == 2 and parts[1].isdigit() else stem
                    cap = cap_map.get(stem) or cap_map.get(parent_stem)
                    if cap and not r.get("prompt"):
                        r["prompt"] = cap
                        filled += 1
                print(f"  hoigen1m: {len(rows)} rows, filled {filled}")
                if not args.dry_run and filled > 0:
                    write_jsonl(rows, wan22_path)
                results["hoigen1m"] = filled
            else:
                print(f"  CSV not found: {csv_path}")

    if "matrix" in args.datasets:
        print("\n[matrix] many-to-1 merge via source_sequence_id -> video_path stem")
        wan22_path = WT_DIR / "matrix_wan22" / "manifest_wan22.jsonl"
        cap_path = WT_DIR / "matrix_extracted" / "manifest_captioned.jsonl"
        if wan22_path.exists() and cap_path.exists():
            rows = load_jsonl(wan22_path)
            caps = load_jsonl(cap_path)
            # wan22 source_id: "matrix:part_0_2024-06-03_23-28-42_000000"
            # captioned video_path: "part_0_2024-06-03_23-28-42_000000.mp4" (stem matches after stripping "matrix:")
            cap_map = {}
            for r in caps:
                vp = r.get("video_path", "")
                if vp:
                    stem = str(Path(vp).stem)  # "part_0_2024-06-03_23-28-42_000000"
                    cap_map[stem] = r.get("caption", "")
            print(f"  loaded {len(cap_map)} captions from captioned")
            filled = 0
            for r in rows:
                sid = strip_prefix(r.get("source_sequence_id", ""), "matrix:")
                # sid is "part_0_2024-06-03_23-28-42_000000" — matches captioned video_path stem
                cap = cap_map.get(sid)
                if cap and not r.get("prompt"):
                    r["prompt"] = cap
                    filled += 1
            print(f"  matrix: {len(rows)} rows, filled {filled}")
            if not args.dry_run and filled > 0:
                write_jsonl(rows, wan22_path)
            results["matrix"] = filled
        elif wan22_path.exists() and not cap_path.exists():
            print(f"  matrix: caption file not found at {cap_path} (VLM captioning may still be running)")

    print("\n=== Summary ===")
    total_filled = 0
    for ds, n in results.items():
        print(f"  {ds}: {n} captions merged")
        total_filled += n
    print(f"  TOTAL: {total_filled} captions merged")
    if args.dry_run:
        print("  (dry-run — no files written)")


if __name__ == "__main__":
    main()
