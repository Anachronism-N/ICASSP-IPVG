#!/usr/bin/env python3
"""
Merge re-captioned prompts back into unified_train/val.jsonl.

Reads <dataset>_recaptioned.jsonl files produced by recaption_unified.py and
updates the corresponding rows in unified manifest jsonl files:
  - The old `prompt` is preserved as `prompt_legacy`
  - `prompt` is replaced with the new caption
  - `annotation_source` is updated to the recaption source

A backup of each manifest is written to <manifest>.bak.<timestamp> before
any modification, so the operation is reversible.

Usage:
    python3 scripts/merge_recaptioned.py \\
        --manifest training_metadata/unified_train.jsonl \\
                   training_metadata/unified_val.jsonl \\
        --recaption-dir training_metadata/recaptioned \\
        --datasets mira charades noxi easycom
"""

import argparse
import json
import shutil
import time
from pathlib import Path

PROJECT_ROOT = Path("/srv/wm_teacher")


def load_recaptions(recaption_dir, datasets):
    """Load {video_path: new_prompt} for each dataset, plus annotation_source."""
    mapping = {}
    src_map = {}
    for ds in datasets:
        p = recaption_dir / f"{ds}_recaptioned.jsonl"
        if not p.exists():
            print(f"WARN: {p} not found, skip")
            continue
        n = 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                vp = r.get("video_path")
                if vp:
                    mapping[vp] = r.get("prompt")
                    src_map[vp] = r.get("annotation_source", "qwen3-vl-235b-a22b-fp8-recaption")
                    n += 1
        print(f"  loaded {n} recaptions from {p.name}")
    return mapping, src_map


def merge_manifest(manifest_path, mapping, src_map):
    """Update a manifest jsonl in place, backing up first. Returns (total, updated)."""
    rows = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                rows.append(line)
                continue
            r = json.loads(line)
            vp = r.get("video_path")
            if vp in mapping:
                # Preserve old prompt (only once; don't overwrite legacy on re-run)
                if "prompt_legacy" not in r:
                    r["prompt_legacy"] = r.get("prompt", "")
                r["prompt"] = mapping[vp]
                r["annotation_source"] = src_map[vp]
            rows.append(r)

    # Backup
    ts = time.strftime("%Y%m%d_%H%M%S")
    bak = manifest_path.with_suffix(manifest_path.suffix + f".bak.{ts}")
    shutil.copy2(manifest_path, bak)
    print(f"  backup -> {bak.name}")

    # Write back
    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in rows:
            if isinstance(r, str):
                f.write(r)
            else:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Count updated
    updated = sum(1 for r in rows if isinstance(r, dict) and r.get("annotation_source") == src_map.get(next(iter(src_map), ""), ""))
    return len(rows), updated


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", nargs="+", required=True,
                    help="Unified manifest jsonl file(s) to update in place")
    ap.add_argument("--recaption-dir", default="training_metadata/recaptioned",
                    help="Directory containing <ds>_recaptioned.jsonl")
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="Dataset names to merge (matches <ds>_recaptioned.jsonl)")
    args = ap.parse_args()

    rec_dir = Path(args.recaption_dir)
    if not rec_dir.is_absolute():
        rec_dir = PROJECT_ROOT / rec_dir

    print(f"Loading recaptions from {rec_dir} ...")
    mapping, src_map = load_recaptions(rec_dir, args.datasets)
    print(f"Total recaptioned video_paths: {len(mapping)}")
    if not mapping:
        print("Nothing to merge. Exit.")
        return

    for mp in args.manifest:
        mp = Path(mp)
        if not mp.is_absolute():
            mp = PROJECT_ROOT / mp
        if not mp.exists():
            print(f"WARN: {mp} not found, skip")
            continue
        print(f"\nMerging into {mp.name} ...")
        total, updated = merge_manifest(mp, mapping, src_map)
        print(f"  rows={total} updated={updated}")

    # Quick stats
    print("\nDone. Verifying prompt fill rate:")
    for mp in args.manifest:
        mp = Path(mp)
        if not mp.is_absolute():
            mp = PROJECT_ROOT / mp
        empty = 0
        total = 0
        with open(mp, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                total += 1
                if not (r.get("prompt") or "").strip():
                    empty += 1
        print(f"  {mp.name}: {total} rows, {empty} empty prompt ({100*empty/total:.2f}%)")


if __name__ == "__main__":
    main()
