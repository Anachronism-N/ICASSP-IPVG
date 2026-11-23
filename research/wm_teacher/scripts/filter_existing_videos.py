#!/usr/bin/env python3
"""
Filter unified manifest to only entries whose video file exists on disk.

Also reports per-source missing stats, so we can see which datasets have
extraction gaps (e.g. MIRA captions referencing not-yet-extracted videos).

Usage:
    python3 scripts/filter_existing_videos.py
"""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path("/srv/wm_teacher")
IN_FILES = [
    PROJECT_ROOT / "training_metadata" / "unified_train.jsonl",
    PROJECT_ROOT / "training_metadata" / "unified_val.jsonl",
]


def main():
    total_by_src = Counter()
    missing_by_src = Counter()

    for in_path in IN_FILES:
        if not in_path.exists():
            print(f"SKIP (not found): {in_path}")
            continue

        rows = []
        with open(in_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))

        kept = []
        for r in rows:
            src = r.get("source_dataset", "?")
            vp = r.get("video_path", "")
            total_by_src[src] += 1
            if vp and os.path.exists(vp):
                kept.append(r)
            else:
                missing_by_src[src] += 1

        out_path = in_path.with_name(in_path.name.replace(".jsonl", "_filtered.jsonl"))
        with open(out_path, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        n_total = len(rows)
        n_kept = len(kept)
        n_miss = n_total - n_kept
        print(f"{in_path.name}: {n_total} -> {n_kept} kept ({n_miss} missing) -> {out_path.name}")

    print("\n=== Per-source missing stats ===")
    print(f"{'source':<14} {'total':>8} {'missing':>8} {'miss%':>7}")
    for src in sorted(total_by_src):
        t = total_by_src[src]
        m = missing_by_src[src]
        pct = 100.0 * m / t if t else 0
        print(f"{src:<14} {t:>8} {m:>8} {pct:>6.1f}%")
    t = sum(total_by_src.values())
    m = sum(missing_by_src.values())
    print(f"{'TOTAL':<14} {t:>8} {m:>8} {100.0*m/t:>6.1f}%")


if __name__ == "__main__":
    main()
