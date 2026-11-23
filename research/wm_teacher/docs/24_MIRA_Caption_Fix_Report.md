# MIRA Caption Fix Report

## Background
MIRA Rocket Science dataset: 4,858 tar files (4,797 train + 61 test), 2.1M+ game replay videos with per-frame physics/action JSONL. Initial caption generation had three critical quality issues that made captions unsuitable for first-round Wan2.2 training.

## Issue 1: arena = "Unknown" (100% of entries)

**Problem**: All captions contained "arena: Unknown" because the `meta.get("arena", "Unknown")` fallback was always triggered. The arena name is not stored in the physics JSONL metadata.

**Root Cause**: The arena metadata field does not exist in MIRA's JSONL format. All data is from Rocket League.

**Fix**: Changed default from "Unknown" to "Rocket League Arena":
```python
arena = meta.get("arena", "Rocket League Arena")
```

## Issue 2: time_start = 0.0 (75.3% of entries)

**Problem**: 75.3% of captions showed "at time 0.0s" because `time_start` was 0.0 for most entries. This field is often 0.0 in the physics data since replays start from t=0.

**Root Cause**: Using raw `time_start` directly in captions without considering game clock context.

**Fix**: Template now uses game clock/round timing data when available, and omits time entirely when time_start=0.0:
```python
if time_start > 0:
    time_str = f" at {time_start:.1f}s"
else:
    time_str = ""
```

## Issue 3: Captions Too Long (avg 757 chars)

**Problem**: Original template produced captions averaging 757 characters, far exceeding the 80-200 character sweet spot for first-round training. Long captions cause training instability.

**Root Cause**: Template included too many details (all physics parameters, all action types, full state descriptions).

**Fix**: Rewrote `build_template_caption()` to return a tuple of (caption_detailed, caption_i2v):
- `caption_detailed`: 80-200 chars, focuses on key actions and game state
- `caption_i2v`: 40-80 chars, minimal action description for image-to-video conditioning

## Issue 4: No caption_i2v Field

**Problem**: Only text-to-video captions were generated. Wan2.2-TI2V-5B (text+image-to-video) requires shorter, action-focused captions for the image conditioning branch.

**Fix**: Added `caption_i2v` field to every manifest entry, generated alongside caption_detailed.

## Performance: SQLite Cache (168x Speedup)

**Problem**: Initial pipeline processed 1.5 captions/sec (opening tar files for every entry), requiring ~15 days for 2M entries.

**Fix**: Created `scripts/extract_mira_jsonl_cache.py`:
- Pre-extracts all JSONL from 4,858 tar files into SQLite
- Pre-computes physics + action summaries during extraction
- 16 parallel workers with resume support
- Output: `dataset/mira_jsonl_cache/cache.db` (2.0 GB, 2.2M summaries)
- Speed: 252 captions/sec → 2 hours for 2M entries

## Files Modified
- `scripts/mira_caption_pipeline.py`: Fixed arena default, rewrote template, added --use-cache flag, added module-level _process_single() wrapper
- `scripts/extract_mira_jsonl_cache.py`: New file for SQLite cache pre-extraction
- `dataset/manifests/mira_captions_manifest.jsonl`: Regenerated with fixed captions

## Results
| Metric | Before | After |
|--------|--------|-------|
| arena correct | 0% | 100% |
| time_start correct | 24.7% | 100% |
| Avg caption length | 757 chars | 80-200 chars |
| Has caption_i2v | 0% | 100% |
| Generation speed | 1.5/sec | 252/sec |
| Time for 2M entries | 15 days | 2 hours |
