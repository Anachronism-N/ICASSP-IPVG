#!/usr/bin/env python3
"""
MIRA JSONL Cache Extraction (Optimized)
========================================
Pre-extracts JSONL/meta data from MIRA tar files into a fast SQLite cache,
pre-computing physics and action summaries to eliminate redundant computation.

The caption pipeline previously spent most of its time:
  1. Opening tar files (I/O)
  2. Reading JSONL data (I/O + parsing)
  3. Computing physics/action summaries (CPU)

This script eliminates (1) and (2) by pre-extracting into SQLite.
It also pre-computes (3) so the caption pipeline is pure cache lookup.

Cache Schema:
  Table: summaries
    match_id TEXT, chunk INT, player INT,
    actions_json TEXT,      -- pre-computed action summary (small)
    physics_json TEXT,      -- pre-computed physics summary (small)
    meta_json TEXT,         -- chunk meta (arena, teams, player_ids)
    PRIMARY KEY (match_id, chunk, player)

Cache size: ~500 MB total (vs ~500 GB for raw frame storage)

Usage:
  python3 scripts/extract_mira_jsonl_cache.py
  python3 scripts/extract_mira_jsonl_cache.py --workers 16
  python3 scripts/extract_mira_jsonl_cache.py --limit 10 --drop-existing
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sqlite3
import tarfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/srv/wm_teacher")
MIRA_BASE = PROJECT_ROOT / "dataset/general_action/mira_rocket"
CACHE_DIR = PROJECT_ROOT / "dataset/mira_jsonl_cache"
CACHE_DB = CACHE_DIR / "cache.db"
TAR_INDEX_TRAIN = PROJECT_ROOT / "dataset/mira_tar_index_train.json"
TAR_INDEX_TEST = PROJECT_ROOT / "dataset/mira_tar_index_test.json"

KEY_TO_ACTION = {
    "W": "accelerating forward", "A": "turning left", "S": "braking or reversing",
    "D": "turning right", "Q": "air-rolling left", "E": "air-rolling right",
    "Space": "jumping", "LShiftKey": "boosting or power-sliding",
    "LControlKey": "handbraking",
}


# ---------------------------------------------------------------------------
# Summary Computation (inline, no external dependencies)
# ---------------------------------------------------------------------------
def compute_physics_summary(frames: list[dict]) -> dict:
    """Compute physics summary identical to mira_caption_pipeline.analyze_physics_summary."""
    n = len(frames)
    if n == 0:
        return {"n_frames": 0}

    car_pos_x, car_pos_y, car_pos_z = [], [], []
    car_vel_x, car_vel_y, car_vel_z = [], [], []
    ball_pos_x, ball_pos_y, ball_pos_z = [], [], []
    ball_vel_x, ball_vel_y, ball_vel_z = [], [], []
    boost_vals = []
    game_states = []

    airborne_count = 0
    supersonic_count = 0
    jumped = False
    dodged = False
    demolished = False
    near_ball_count = 0

    for frame in frames:
        car = None
        for c in frame.get("cars", []):
            if c.get("is_local", False):
                car = c
                break
        if car:
            loc = car.get("location", {})
            vel = car.get("velocity", {})
            car_pos_x.append(loc.get("x", 0))
            car_pos_y.append(loc.get("y", 0))
            car_pos_z.append(loc.get("z", 0))
            car_vel_x.append(vel.get("x", 0))
            car_vel_y.append(vel.get("y", 0))
            car_vel_z.append(vel.get("z", 0))
            boost_vals.append(car.get("boost_amount", 0))
            if not car.get("is_on_ground", True):
                airborne_count += 1
            if car.get("is_supersonic", False):
                supersonic_count += 1
            if car.get("has_jumped", False):
                jumped = True
            if car.get("is_dodging", False):
                dodged = True
            if car.get("is_demolished", False):
                demolished = True

        ball = frame.get("ball", {})
        if ball.get("location"):
            ball_pos_x.append(ball["location"].get("x", 0))
            ball_pos_y.append(ball["location"].get("y", 0))
            ball_pos_z.append(ball["location"].get("z", 0))
        if ball.get("velocity"):
            ball_vel_x.append(ball["velocity"].get("x", 0))
            ball_vel_y.append(ball["velocity"].get("y", 0))
            ball_vel_z.append(ball["velocity"].get("z", 0))
        game_states.append(frame.get("game", {}))

        if car and ball.get("location"):
            dx = car_pos_x[-1] - ball["location"].get("x", 0)
            dy = car_pos_y[-1] - ball["location"].get("y", 0)
            dz = car_pos_z[-1] - ball["location"].get("z", 0)
            if math.sqrt(dx**2 + dy**2 + dz**2) < 500:
                near_ball_count += 1

    def _mean(vals):
        return round(sum(vals) / len(vals)) if vals else 0

    if car_vel_x:
        speeds = [math.sqrt(vx**2 + vy**2 + vz**2) for vx, vy, vz in zip(car_vel_x, car_vel_y, car_vel_z)]
        avg_speed = round(sum(speeds) / len(speeds))
        max_speed = round(max(speeds))
    else:
        avg_speed = max_speed = 0

    if car_pos_x and car_pos_y:
        total_dist = sum(
            math.sqrt((car_pos_x[i] - car_pos_x[i-1])**2 + (car_pos_y[i] - car_pos_y[i-1])**2 + (car_pos_z[i] - car_pos_z[i-1])**2)
            for i in range(1, len(car_pos_x))
        )
    else:
        total_dist = 0

    near_goal_count = sum(1 for y in ball_pos_y if abs(y) > 1500)
    ball_pct = round(near_ball_count / n * 100)

    first_game = game_states[0] if game_states else {}
    last_game = game_states[-1] if game_states else {}

    return {
        "n_frames": n,
        "time_start": first_game.get("time_remaining", 0),
        "time_end": last_game.get("time_remaining", 0),
        "score_blue": last_game.get("score_blue", first_game.get("score_blue", 0)),
        "score_orange": last_game.get("score_orange", first_game.get("score_orange", 0)),
        "is_overtime": first_game.get("is_overtime", False),
        "car_total_dist": round(total_dist),
        "car_avg_speed": avg_speed,
        "car_max_speed": max_speed,
        "boost_start_pct": round(boost_vals[0] * 100) if boost_vals else 100,
        "boost_min_pct": round(min(boost_vals) * 100) if boost_vals else 0,
        "boost_used_pct": round((boost_vals[0] - min(boost_vals)) * 100) if boost_vals else 0,
        "airborne_frames": airborne_count,
        "supersonic_frames": supersonic_count,
        "has_jumped": jumped,
        "has_dodged": dodged,
        "has_demolished": demolished,
        "near_ball_frames": near_ball_count,
        "near_ball_pct": ball_pct,
        "near_goal_frames": near_goal_count,
        "ball_max_height": round(max(ball_pos_z)) if ball_pos_z else 0,
        "ball_avg_height": round(sum(ball_pos_z) / len(ball_pos_z)) if ball_pos_z else 0,
    }


def compute_action_summary(frames: list[dict]) -> dict:
    """Compute action summary identical to mira_caption_pipeline.analyze_action_summary."""
    if not frames:
        return {"n_frames": 0}

    total = len(frames)
    key_counts = Counter()
    key_sequences = []

    for frame in frames:
        keys = frame.get("keys", [])
        for k in keys:
            key_counts[k] += 1
        key_sequences.append(set(keys))

    dominant_actions = []
    for key, count in key_counts.most_common(3):
        pct = count / total * 100
        action_name = KEY_TO_ACTION.get(key, key)
        if pct > 75:
            dominant_actions.append(f"continuously {action_name}")
        elif pct > 40:
            dominant_actions.append(f"frequently {action_name}")
        elif pct > 10:
            dominant_actions.append(f"occasionally {action_name}")

    has_turning = any("A" in ks or "D" in ks for ks in key_sequences)
    has_boost = any("LShiftKey" in ks for ks in key_sequences)
    has_jump = any("Space" in ks for ks in key_sequences)
    has_airroll = any("Q" in ks or "E" in ks for ks in key_sequences)
    complex_pct = round(sum(1 for ks in key_sequences if len(ks) >= 3) / total * 100)
    idle_pct = round(sum(1 for ks in key_sequences if not ks) / total * 100)

    return {
        "n_frames": total,
        "dominant_actions": dominant_actions,
        "has_turning": has_turning,
        "has_boost": has_boost,
        "has_jump": has_jump,
        "has_airroll": has_airroll,
        "complex_maneuvers_pct": complex_pct,
        "idle_pct": idle_pct,
    }


# ---------------------------------------------------------------------------
# SQLite Setup
# ---------------------------------------------------------------------------
def init_db(db_path: Path) -> sqlite3.Connection:
    """Create the cache database with summary tables."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")
    conn.execute("PRAGMA busy_timeout=30000")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS summaries (
            match_id TEXT NOT NULL,
            chunk INTEGER NOT NULL,
            player INTEGER NOT NULL,
            actions_json TEXT,
            physics_json TEXT,
            meta_json TEXT,
            PRIMARY KEY (match_id, chunk, player)
        );

        CREATE TABLE IF NOT EXISTS chunk_meta (
            match_id TEXT NOT NULL,
            chunk INTEGER NOT NULL,
            meta_json TEXT NOT NULL,
            PRIMARY KEY (match_id, chunk)
        );

        CREATE TABLE IF NOT EXISTS extraction_progress (
            tar_path TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            entries_extracted INTEGER DEFAULT 0,
            error_message TEXT,
            started_at REAL,
            completed_at REAL
        );

        CREATE INDEX IF NOT EXISTS idx_summaries_lookup ON summaries(match_id, chunk, player);
        CREATE INDEX IF NOT EXISTS idx_chunk_meta_lookup ON chunk_meta(match_id, chunk);
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Per-Tar Extraction
# ---------------------------------------------------------------------------
def extract_tar(tar_path_str: str) -> dict:
    """Extract and pre-compute summaries from a single tar file."""
    result = {
        "tar_path": tar_path_str,
        "status": "ok",
        "summaries": [],      # (match_id, chunk, player, actions_json, physics_json)
        "metas": [],           # (match_id, chunk, meta_json)
        "summary_counts": {"summaries": 0, "metas": 0},
        "error": None,
    }

    tar_path = Path(tar_path_str)
    if not tar_path.exists():
        result["status"] = "missing"
        result["error"] = f"File not found: {tar_path_str}"
        return result

    member_re = re.compile(
        r"^(?P<match_id>.+?)_c(?P<chunk>\d{5})\."
        r"(?P<type>p(?P<player>\d)|meta)"
        r"\.(?P<ext>jsonl|json|physics\.jsonl)$"
    )

    try:
        tar = tarfile.open(tar_path, "r")
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        return result

    try:
        members = [m for m in tar.getmembers() if m.isfile()]

        # Group by (match_id, chunk) to process meta + all players together
        # Structure: chunk_data[(match_id, chunk)] = {meta: {...}, actions: {0: [...], 1: [...], ...}, physics: {0: [...], 1: [...], ...}}
        from collections import defaultdict
        chunk_data = defaultdict(lambda: {"meta": None, "actions": {}, "physics": {}})

        for member in members:
            m = member_re.match(member.name)
            if not m:
                continue

            match_id = m.group("match_id")
            chunk = int(m.group("chunk"))
            entry_type = m.group("type")
            ext = m.group("ext")
            key = (match_id, chunk)

            if entry_type == "meta" and ext == "json":
                try:
                    mf = tar.extractfile(member)
                    if mf:
                        chunk_data[key]["meta"] = json.load(mf)
                except Exception:
                    pass

            elif entry_type.startswith("p") and ext == "jsonl":
                player = int(m.group("player"))
                try:
                    mf = tar.extractfile(member)
                    if mf:
                        raw = mf.read().decode("utf-8")
                        frames = [json.loads(line) for line in raw.strip().split("\n") if line.strip()]
                        chunk_data[key]["actions"][player] = frames
                except Exception:
                    pass

            elif entry_type.startswith("p") and ext == "physics.jsonl":
                player = int(m.group("player"))
                try:
                    mf = tar.extractfile(member)
                    if mf:
                        raw = mf.read().decode("utf-8")
                        frames = [json.loads(line) for line in raw.strip().split("\n") if line.strip()]
                        chunk_data[key]["physics"][player] = frames
                except Exception:
                    pass

        # Pre-compute summaries for each (match, chunk, player)
        for (match_id, chunk), data in chunk_data.items():
            meta = data["meta"] or {}
            meta_json = json.dumps(meta)

            # Store chunk-level meta
            result["metas"].append((match_id, chunk, meta_json))
            result["summary_counts"]["metas"] += 1

            # Get all players that have either actions or physics data
            all_players = set(data["actions"].keys()) | set(data["physics"].keys())

            for player in all_players:
                action_frames = data["actions"].get(player, [])
                physics_frames = data["physics"].get(player, [])

                action_summary = compute_action_summary(action_frames)
                physics_summary = compute_physics_summary(physics_frames)

                result["summaries"].append((
                    match_id, chunk, player,
                    json.dumps(action_summary),
                    json.dumps(physics_summary),
                ))
                result["summary_counts"]["summaries"] += 1

    finally:
        tar.close()

    return result


# ---------------------------------------------------------------------------
# Batch Insert
# ---------------------------------------------------------------------------
def insert_batch(conn: sqlite3.Connection, summaries: list, metas: list):
    """Insert extracted summaries into SQLite in a single transaction."""
    if summaries:
        conn.executemany(
            "INSERT OR REPLACE INTO summaries (match_id, chunk, player, actions_json, physics_json) VALUES (?, ?, ?, ?, ?)",
            summaries,
        )
    if metas:
        conn.executemany(
            "INSERT OR REPLACE INTO chunk_meta (match_id, chunk, meta_json) VALUES (?, ?, ?)",
            metas,
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_extraction(num_workers: int = 16, resume: bool = True, limit: int = 0):
    match_to_tar: dict[str, str] = {}
    for idx_path in [TAR_INDEX_TEST, TAR_INDEX_TRAIN]:
        if idx_path.exists():
            with open(idx_path) as f:
                match_to_tar.update(json.load(f))

    all_tar_paths = sorted(set(match_to_tar.values()))
    logger.info("Loaded %d match_ids mapped to %d unique tar files",
                len(match_to_tar), len(all_tar_paths))

    if limit > 0:
        all_tar_paths = all_tar_paths[:limit]
        logger.info("Limited to %d tar files", len(all_tar_paths))

    conn = init_db(CACHE_DB)

    remaining = all_tar_paths
    if resume:
        processed = set()
        try:
            cursor = conn.execute("SELECT tar_path FROM extraction_progress WHERE status = 'ok'")
            for row in cursor:
                processed.add(row[0])
        except sqlite3.OperationalError:
            pass
        remaining = [p for p in all_tar_paths if p not in processed]
        logger.info("Resume: %d already done, %d remaining", len(processed), len(remaining))

    if not remaining:
        logger.info("All tar files already extracted!")
        conn.close()
        return

    total = len(all_tar_paths)
    done = total - len(remaining)
    logger.info("Starting extraction of %d tar files with %d workers", len(remaining), num_workers)

    total_sums = 0
    total_metas = 0
    errors = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(extract_tar, p): p for p in remaining}

        for i, future in enumerate(as_completed(futures)):
            tar_path = futures[future]
            try:
                result = future.result()
            except Exception as e:
                logger.error("Worker error for %s: %s", tar_path, e)
                errors += 1
                continue

            if result["status"] == "error":
                logger.warning("Failed %s: %s", tar_path, result.get("error"))
                conn.execute(
                    "INSERT OR REPLACE INTO extraction_progress (tar_path, status, error_message, completed_at) VALUES (?, ?, ?, ?)",
                    (tar_path, "error", result.get("error"), time.time()),
                )
                conn.commit()
                errors += 1
            else:
                insert_batch(conn, result["summaries"], result["metas"])
                total_sums += result["summary_counts"]["summaries"]
                total_metas += result["summary_counts"]["metas"]
                conn.execute(
                    "INSERT OR REPLACE INTO extraction_progress (tar_path, status, entries_extracted, completed_at) VALUES (?, ?, ?, ?)",
                    (tar_path, "ok", result["summary_counts"]["summaries"], time.time()),
                )
                conn.commit()

            processed = done + i + 1
            if processed % 50 == 0 or i == len(remaining) - 1:
                elapsed = time.time() - t_start
                rate = processed / max(elapsed, 0.01)
                eta = (total - processed) / max(rate, 0.01)
                logger.info(
                    "Progress: %d/%d (%.1f%%) | %.1f tars/s | err=%d | "
                    "summaries=%d metas=%d | ETA: %.0fs",
                    processed, total, processed / total * 100,
                    rate, errors, total_sums, total_metas, eta,
                )

    conn.close()
    elapsed = time.time() - t_start
    db_size = CACHE_DB.stat().st_size / (1024 * 1024) if CACHE_DB.exists() else 0

    logger.info("=========================================")
    logger.info("Extraction complete!")
    logger.info("  Tar files: %d/%d, errors: %d", total - errors, total, errors)
    logger.info("  Summaries: %d, Chunk metas: %d", total_sums, total_metas)
    logger.info("  Database size: %.1f MB", db_size)
    logger.info("  Time: %.0fs (%.1f tars/s)", elapsed, total / max(elapsed, 0.01))
    logger.info("  Cache: %s", CACHE_DB)
    logger.info("=========================================")

    # Verify
    conn = sqlite3.connect(str(CACHE_DB))
    for table in ["summaries", "chunk_meta"]:
        cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
        logger.info("Verify: %s = %d rows", table, cursor.fetchone()[0])
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="MIRA JSONL Cache Extraction (with summary pre-computation)")
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 8, 16),
                        help="Number of parallel workers")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume")
    parser.add_argument("--limit", type=int, default=0, help="Limit tar files (testing)")
    parser.add_argument("--drop-existing", action="store_true", help="Drop and recreate database")

    args = parser.parse_args()

    if args.drop_existing and CACHE_DB.exists():
        CACHE_DB.unlink()
        logger.info("Dropped existing cache")

    run_extraction(num_workers=args.workers, resume=not args.no_resume, limit=args.limit)


if __name__ == "__main__":
    main()
