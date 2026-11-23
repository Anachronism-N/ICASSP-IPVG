#!/usr/bin/env python3
"""
MIRA Caption Pipeline
=====================
Converts MIRA (Rocket Science) structured action/physics JSONL data into
natural language captions for Wan2.2 video generation training.

Data Flow:
  1. Build tar index: tar_path -> list of (match_id, chunk, perspectives)  (one-time)
  2. Read processed video manifest (manifest.jsonl from wan22_training/mira_processed/)
  3. Map video filename -> tar -> extract action + physics JSONL
  4. Analyze physics for game events and movement
  5. Analyze actions for keyboard input patterns
  6. Convert to natural language caption (LLM API or template-based fallback)
  7. Output Wan2.2 training manifest with captured captions

Usage:
  # Build tar index
  python scripts/mira_caption_pipeline.py --build-index

  # Generate captions (template-based, no API needed)
  python scripts/mira_caption_pipeline.py --generate \
      --limit 100

  # Generate captions with LLM
  python scripts/mira_caption_pipeline.py --generate \
      --use-llm \
      --llm-endpoint http://localhost:8000/v1/chat/completions \
      --limit 100

Environment Variables:
  MIRA_LLM_API_KEY: OpenAI-compatible API key (if not using local vLLM)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import tarfile
import time
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

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
VIDEO_PROCESSED_DIR = PROJECT_ROOT / "dataset/wan22_training/mira_processed"
OUTPUT_BASE = PROJECT_ROOT / "dataset/manifests"
TAR_INDEX_CACHE = PROJECT_ROOT / "dataset/mira_tar_index.json"
TAR_INDEX_TEST = PROJECT_ROOT / "dataset/mira_tar_index_test.json"
TAR_INDEX_TRAIN = PROJECT_ROOT / "dataset/mira_tar_index_train.json"

# Action key mapping (from MIRA: W A S D Q E Space LShiftKey LControlKey)
ACTION_KEYS = ["W", "A", "S", "D", "Q", "E", "Space", "LShiftKey", "LControlKey"]

KEY_TO_ACTION = {
    "W": "accelerating forward",
    "A": "turning left",
    "S": "braking or reversing",
    "D": "turning right",
    "Q": "air-rolling left",
    "E": "air-rolling right",
    "Space": "jumping",
    "LShiftKey": "boosting or power-sliding",
    "LControlKey": "handbraking",
}

TEAM_NAMES = {0: "Blue team", 1: "Orange team"}

ARENA_NAMES = {
    "CHN_Stadium_P": "DFH Stadium",
    "beach_P": "Salty Shores",
    "cs_p": "Champions Field",
    "eurostadium_night_p": "Mannfield (Night)",
    "eurostadium_p": "Mannfield",
    "eurostadium_rainy_p": "Mannfield (Stormy)",
    "farm_p": "Farmstead",
    "throwbackstadium_P": "Throwback Stadium",
    "trainstation_p": "Urban Central",
    "underwater_p": "Aquadome",
    "utopiastadium_p": "Utopia Coliseum",
    "wasteland_p": "Wasteland",
    "wasteland_night_p": "Wasteland (Night)",
    "park_p": "Beckwith Park",
    "park_night_p": "Beckwith Park (Night)",
    "park_rainy_p": "Beckwith Park (Stormy)",
    "neotokyo_p": "Neo Tokyo",
}


# ---------------------------------------------------------------------------
# Tar Index
# ---------------------------------------------------------------------------
def build_tar_index(split: str = "train", output_path: Optional[Path] = None) -> list[dict]:
    """Scan all tar files and build an index of every chunk.

    Uses index.json (when available) for arena/frame metadata.
    Returns a list of chunk_info dicts. Also saves match_id->tar_path mapping.
    """
    if output_path is None:
        output_path = TAR_INDEX_CACHE
    split_dir = MIRA_BASE / split
    index_path = split_dir / "index.json"

    # Load index metadata if available
    index_entries = {}
    if index_path.exists():
        with index_path.open() as f:
            for entry in json.load(f).get("entries", []):
                index_entries[entry["match_id"]] = entry

    tar_files = sorted(split_dir.rglob("*.tar"))
    logger.info("Found %d tar files in %s/%s", len(tar_files), MIRA_BASE, split)

    chunks: list[dict] = []
    match_to_tar: dict[str, str] = {}

    member_re = re.compile(
        r"^(?P<match_id>.+?)_c(?P<chunk>\d{5})\."
        r"(?:(?P<perspective>p\d)\.(?P<ext>.+)|(?P<meta>meta)\.json)$"
    )

    for tpath in tar_files:
        tar_str = str(tpath)
        try:
            with tarfile.open(tpath, "r") as tar:
                member_names = [m.name for m in tar.getmembers() if m.isfile()]

            samples: dict[tuple[str, int], set] = defaultdict(set)
            for name in member_names:
                m = member_re.match(name)
                if not m:
                    continue
                chunk_id = int(m.group("chunk"))
                perspective = m.group("perspective") or "meta"
                samples[(m.group("match_id"), chunk_id)].add(perspective)

            for (match_id, chunk_id), perspectives in sorted(samples.items()):
                entry_meta = index_entries.get(match_id, {})
                chunk_frames = entry_meta.get("chunk_frames", [])
                n_frames = chunk_frames[chunk_id] if chunk_id < len(chunk_frames) else 80
                arena = entry_meta.get("arena", "Rocket League Arena")

                real_p = {p for p in perspectives if p != "meta"}
                n_perspectives = len(real_p)

                chunks.append({
                    "match_id": match_id,
                    "chunk": chunk_id,
                    "n_players": n_perspectives,
                    "n_frames": n_frames,
                    "arena": arena,
                    "tar_path": tar_str,
                })
                if match_id not in match_to_tar:
                    match_to_tar[match_id] = tar_str

        except Exception as e:
            logger.warning("Failed to read %s: %s", tpath, e)

    logger.info("Indexed %d tar files, %d chunks, %d match_ids",
                len(set(c["tar_path"] for c in chunks)), len(chunks), len(match_to_tar))

    # Save match_id -> tar_path mapping for quick lookup
    with open(output_path, "w") as f:
        json.dump(match_to_tar, f)
    logger.info("Saved match->tar index to %s", output_path)

    return chunks


# ---------------------------------------------------------------------------
# Filename Parsing
# ---------------------------------------------------------------------------
FILENAME_RE = re.compile(
    r"^(?P<match_id>.+?)_c(?P<chunk>\d+)\.p(?P<player>\d+)_720p\.mp4$"
)


def parse_manifest_filename(filename: str) -> dict | None:
    """Parse {match_id}_c{chunk:05d}.p{player}_720p.mp4 into components."""
    m = FILENAME_RE.match(filename)
    if not m:
        return None
    return {
        "match_id": m.group("match_id"),
        "chunk": int(m.group("chunk")),
        "player": int(m.group("player")),
    }


# ---------------------------------------------------------------------------
# Data Extraction from Tar
# ---------------------------------------------------------------------------
def _read_tar_jsonl(tar: tarfile.TarFile, member_name: str) -> list[dict]:
    """Read a JSONL member from a tar file, return list of parsed dicts."""
    try:
        member = tar.getmember(member_name)
        f = tar.extractfile(member)
        if f is None:
            return []
        raw = f.read().decode("utf-8")
        return [json.loads(line) for line in raw.strip().split("\n") if line.strip()]
    except (KeyError, json.JSONDecodeError):
        return []


# ---------------------------------------------------------------------------
# Physics Analysis
# ---------------------------------------------------------------------------
def analyze_physics_summary(physics_frames: list[dict], player_idx: int) -> dict:
    """Extract comprehensive game state statistics from physics data for a single perspective.

    Returns structured stats for caption generation.
    """
    n = len(physics_frames)
    if n == 0:
        return {"n_frames": 0}

    # Collect data across all frames
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

    for frame in physics_frames:
        # Local player car
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

        # Ball
        ball = frame.get("ball", {})
        if ball.get("location"):
            ball_pos_x.append(ball["location"].get("x", 0))
            ball_pos_y.append(ball["location"].get("y", 0))
            ball_pos_z.append(ball["location"].get("z", 0))
        if ball.get("velocity"):
            ball_vel_x.append(ball["velocity"].get("x", 0))
            ball_vel_y.append(ball["velocity"].get("y", 0))
            ball_vel_z.append(ball["velocity"].get("z", 0))

        # Game state
        game_states.append(frame.get("game", {}))

        # Ball proximity to car
        if car and ball.get("location"):
            dx = car_pos_x[-1] - ball["location"].get("x", 0)
            dy = car_pos_y[-1] - ball["location"].get("y", 0)
            dz = car_pos_z[-1] - ball["location"].get("z", 0)
            if math.sqrt(dx**2 + dy**2 + dz**2) < 500:
                near_ball_count += 1

    # Derived statistics
    def _mean(vals):
        return round(sum(vals) / len(vals)) if vals else 0

    def _max(vals):
        return round(max(vals)) if vals else 0

    if car_vel_x:
        speeds = [math.sqrt(vx**2 + vy**2 + vz**2) for vx, vy, vz in zip(car_vel_x, car_vel_y, car_vel_z)]
        avg_speed = round(sum(speeds) / len(speeds))
        max_speed = round(max(speeds))
    else:
        avg_speed = max_speed = 0

    if car_pos_x and car_pos_y:
        start_x, start_y = round(car_pos_x[0]), round(car_pos_y[0])
        end_x, end_y = round(car_pos_x[-1]), round(car_pos_y[-1])
        total_dist = sum(
            math.sqrt(
                (car_pos_x[i] - car_pos_x[i-1])**2
                + (car_pos_y[i] - car_pos_y[i-1])**2
                + (car_pos_z[i] - car_pos_z[i-1])**2
            )
            for i in range(1, len(car_pos_x))
        )
    else:
        start_x = end_x = start_y = end_y = total_dist = 0

    # Goal proximity (ball within ~1500 units of a goal line)
    near_goal_count = sum(1 for y in ball_pos_y if abs(y) > 1500)

    # Ball proximity to player
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
        "car_start_pos": (start_x, start_y),
        "car_end_pos": (end_x, end_y),
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


# ---------------------------------------------------------------------------
# Action Analysis
# ---------------------------------------------------------------------------
def analyze_action_summary(action_frames: list[dict]) -> dict:
    """Summarize keyboard actions across the clip."""
    if not action_frames:
        return {"n_frames": 0}

    total = len(action_frames)
    key_counts = Counter()
    key_sequences = []

    for frame in action_frames:
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
# Template-Based Caption
# ---------------------------------------------------------------------------
def build_template_caption(
    player_id: int,
    team: int,
    n_players: int,
    arena: str,
    physics: dict,
    actions: dict,
) -> tuple[str, str]:
    """Generate concise captions from structured game data.

    Returns (caption_detailed, caption_i2v).
    Caption_detailed targets 100-250 chars. Caption_i2v targets 50-100 chars.
    """
    team_short = "Blue" if team == 0 else "Orange"

    # Determine key observable actions for the caption
    action_keywords = []
    if actions.get("has_jump", False):
        action_keywords.append("jumping")
    if actions.get("has_boost", False):
        action_keywords.append("boosting")
    if physics.get("airborne_frames", 0) > 10:
        action_keywords.append("going aerial")
    if actions.get("has_dodged", False):
        action_keywords.append("dodging")
    if actions.get("has_airroll", False):
        action_keywords.append("air-rolling")

    ball_pct = physics.get("near_ball_pct", 0)
    if ball_pct > 25:
        action_keywords.append("contesting the ball")

    if not action_keywords:
        action_keywords.append("driving and positioning")

    action_summary = ", ".join(action_keywords)

    # Build detailed caption (target: 100-250 chars)
    parts = [f"Rocket League gameplay: {action_summary}. "]

    score_b = physics.get("score_blue", 0)
    score_o = physics.get("score_orange", 0)
    parts.append(f"Score: Blue {score_b}-{score_o} Orange")

    if physics.get("is_overtime"):
        parts.append(" (overtime)")

    parts.append(f". {team_short} team perspective")

    if ball_pct > 60:
        parts.append(", dense ball contest")
    elif ball_pct > 25:
        parts.append(", near the ball")

    if physics.get("supersonic_frames", 0) > 10:
        parts.append(", supersonic speed")

    if physics.get("near_goal_frames", 0) > 10:
        parts.append(", goal area action")

    parts.append(".")

    detailed = "".join(parts)

    # Build i2v caption (one sentence, 50-100 chars)
    i2v = f"A Rocket League player {action_summary} from {team_short} team perspective."

    return detailed, i2v


# ---------------------------------------------------------------------------
# LLM Caption Generation
# ---------------------------------------------------------------------------
CAPTION_SYSTEM_PROMPT = """You are a specialized caption generator for Rocket League gameplay videos.
Given structured game state data (physics, actions, events), produce concise,
descriptive natural language captions for video generation model training.

RULES:
1. Write in natural, flowing English describing what happens in the clip.
2. Be specific about movement, actions, ball interactions, and game context.
3. Convert numeric data to qualitative descriptions. Do NOT list raw coordinates.
4. Include: scene/arena, team, what the player is doing, ball state, camera perspective.
5. Keep captions between 50-120 words. ONE sentence can cover multiple details.
6. Output just the caption text, no formatting or preamble."""


def build_llm_prompt(
    player_id: int,
    team: int,
    n_players: int,
    arena: str,
    physics: dict,
    actions: dict,
) -> str:
    """Build the LLM prompt from structured game data."""
    arena_name = ARENA_NAMES.get(arena, arena)
    team_name = TEAM_NAMES.get(team, f"Team {team}")
    match_type = "2v2" if n_players == 4 else "1v1" if n_players == 2 else f"{n_players}-player"

    sx, sy = physics.get("car_start_pos", (0, 0))
    ex, ey = physics.get("car_end_pos", (0, 0))
    t0 = physics.get("time_start", 0)
    t1 = physics.get("time_end", 0)
    duration = abs(t1 - t0) if t0 is not None and t1 is not None else 4.0
    total_f = physics.get("n_frames", 80)

    return f"""Generate a natural language caption for this Rocket League gameplay clip.

CONTEXT:
- Arena: {arena_name}
- Match type: {match_type}
- Perspective: Player {player_id} on {team_name}
- Clip duration: {duration:.1f}s at 20fps

GAME STATE:
- Time remaining: {t0:.1f}s to {t1:.1f}s
- Score: Blue {physics.get('score_blue', 0)} - Orange {physics.get('score_orange', 0)}{' (Overtime)' if physics.get('is_overtime') else ''}
- Car travel: from ({sx}, {sy}) to ({ex}, {ey})
- Distance: {physics.get('car_total_dist', 0)} units, Avg speed: {physics.get('car_avg_speed', 0)}, Max: {physics.get('car_max_speed', 0)}
- Airborne: {physics.get('airborne_frames', 0)}/{total_f} frames
- Supersonic: {physics.get('supersonic_frames', 0)}/{total_f} frames
- Boost: started {physics.get('boost_start_pct', 0)}%, used {physics.get('boost_used_pct', 0)}%
- Ball near player: {physics.get('near_ball_frames', 0)}/{total_f} frames ({physics.get('near_ball_pct', 0)}%)
- Ball near goal: {physics.get('near_goal_frames', 0)} frames
- Ball max height: {physics.get('ball_max_height', 0)} units
- Jumped: {physics.get('has_jumped')}, Dodged: {physics.get('has_dodged')}, Demolished: {physics.get('has_demolished')}

PLAYER ACTIONS:
- Dominant actions: {'; '.join(actions.get('dominant_actions', [])) or 'minimal movement'}
- Complex maneuvers: {actions.get('complex_maneuvers_pct', 0)}%
- Idle time: {actions.get('idle_pct', 0)}% of frames

Generate a concise, descriptive caption:"""


def call_llm(prompt: str, endpoint: str, model: str, api_key: str = "") -> str | None:
    """Call OpenAI-compatible LLM API for caption generation."""
    try:
        import urllib.request

        if not endpoint.endswith("/chat/completions"):
            endpoint = endpoint.rstrip("/") + "/chat/completions" if "/v1" in endpoint else endpoint + "/v1/chat/completions"

        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 256,
            "temperature": 0.7,
        }).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = urllib.request.Request(endpoint, data=payload, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning("LLM API call failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Cache-based Data Access (Fast Path)
# ---------------------------------------------------------------------------
def _read_from_cache(
    cache_db_path: str,
    match_id: str,
    chunk: int,
    player: int,
) -> dict | None:
    """Read pre-computed summaries from SQLite cache.

    Returns dict with keys: meta, n_players, actual_player_id, team, arena,
    physics_summary, action_summary.
    Returns None if not found.
    """
    import sqlite3

    conn = sqlite3.connect(cache_db_path)
    try:
        # Read summary
        cursor = conn.execute(
            "SELECT actions_json, physics_json FROM summaries WHERE match_id=? AND chunk=? AND player=?",
            (match_id, chunk, player),
        )
        sum_row = cursor.fetchone()
        if not sum_row:
            return None

        action_summary = json.loads(sum_row[0])
        physics_summary = json.loads(sum_row[1])

        # Read meta
        cursor = conn.execute(
            "SELECT meta_json FROM chunk_meta WHERE match_id=? AND chunk=?",
            (match_id, chunk),
        )
        meta_row = cursor.fetchone()
        meta = json.loads(meta_row[0]) if meta_row else {}

        # Determine player/team
        player_ids = meta.get("player_ids", [player])
        teams = meta.get("teams", [0, 1, 0, 1])
        n_players = len(player_ids)
        try:
            actual_player_id = player_ids[player]
            team = teams[player]
        except IndexError:
            actual_player_id = player
            team = player % 2

        arena = meta.get("arena", "Rocket League Arena")

    finally:
        conn.close()

    return {
        "meta": meta,
        "n_players": n_players,
        "actual_player_id": actual_player_id,
        "team": team,
        "arena": arena,
        "physics_summary": physics_summary,
        "action_summary": action_summary,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def process_single_entry(
    entry: dict,
    match_to_tar: dict[str, str],
    use_llm: bool = False,
    llm_endpoint: str = "",
    llm_model: str = "gpt-4o-mini",
    llm_api_key: str = "",
    cache_db_path: Optional[str] = None,
) -> dict | None:
    """Process one manifest entry: extract data, analyze, generate caption."""
    video_path = entry.get("video_path", "")
    filename = os.path.basename(video_path)
    parsed = parse_manifest_filename(filename)
    if not parsed:
        return None

    match_id = parsed["match_id"]
    chunk = parsed["chunk"]
    player = parsed["player"]  # perspective index (0-3)

    tar_path = match_to_tar.get(match_id, "")
    chunk_key = f"c{chunk:05d}"

    # ── Fast path: read from SQLite cache ──
    if cache_db_path and os.path.exists(cache_db_path):
        cached = _read_from_cache(cache_db_path, match_id, chunk, player)
        if cached is not None:
            meta = cached["meta"]
            n_players = cached["n_players"]
            actual_player_id = cached["actual_player_id"]
            team = cached["team"]
            arena = cached["arena"]
            physics_summary = cached["physics_summary"]
            action_summary = cached["action_summary"]

            # Check if physics data is present
            if physics_summary.get("n_frames", 0) == 0:
                return None

            # Generate caption (same as slow path)
            caption_source = "template"
            caption_i2v = None
            if use_llm and llm_endpoint:
                prompt = build_llm_prompt(actual_player_id, team, n_players, arena, physics_summary, action_summary)
                caption = call_llm(prompt, llm_endpoint, llm_model, llm_api_key)
                if caption:
                    caption_source = "llm"
                    _, caption_i2v = build_template_caption(actual_player_id, team, n_players, arena, physics_summary, action_summary)
                else:
                    caption, caption_i2v = build_template_caption(actual_player_id, team, n_players, arena, physics_summary, action_summary)
                    caption_source = "template_fallback"
            else:
                caption, caption_i2v = build_template_caption(actual_player_id, team, n_players, arena, physics_summary, action_summary)

            n_frames = physics_summary.get("n_frames", 80)
            duration = n_frames / 20.0

            return {
                "sample_id": f"mira_rocket:{match_id}_{chunk_key}.p{player}",
                "source_dataset": "mira_rocket_science",
                "source_sequence_id": match_id,
                "archive_path": tar_path,
                "video_member": filename,
                "video_path": video_path,
                "reference_image": None,
                "caption_detailed": caption,
                "caption_variants": [caption],
                "caption_i2v": caption_i2v,
                "annotation_source": f"mira_physics_{caption_source}",
                "fps": 20.0,
                "width": 1280,
                "height": 720,
                "duration_sec": round(duration, 2),
                "num_frames": n_frames,
                "split": "unassigned",
                "metadata": {
                    "match_id": match_id,
                    "chunk_id": chunk,
                    "player_id": actual_player_id,
                    "perspective_index": player,
                    "team": team,
                    "total_players": n_players,
                    "arena": arena,
                    "physics": physics_summary,
                    "actions": action_summary,
                },
            }

    # ── Slow path: extract from tar files (fallback) ──
    if not tar_path:
        return None

    try:
        with tarfile.open(tar_path, "r") as tar:
            meta_key = f"{match_id}_{chunk_key}.meta.json"
            meta = {}
            try:
                mf = tar.extractfile(tar.getmember(meta_key))
                if mf:
                    meta = json.load(mf)
            except KeyError:
                pass

            player_ids = meta.get("player_ids", [player])
            teams = meta.get("teams", [0, 1, 0, 1])
            n_players = len(player_ids)
            try:
                actual_player_id = player_ids[player]
                team = teams[player]
            except IndexError:
                actual_player_id = player
                team = player % 2

            action_member = f"{match_id}_{chunk_key}.p{player}.jsonl"
            actions = _read_tar_jsonl(tar, action_member)

            physics_member = f"{match_id}_{chunk_key}.p{player}.physics.jsonl"
            physics = _read_tar_jsonl(tar, physics_member)

    except Exception as e:
        logger.warning("Error reading %s from %s: %s", filename, os.path.basename(tar_path), e)
        return None

    if not physics:
        return None

    physics_summary = analyze_physics_summary(physics, player)
    action_summary = analyze_action_summary(actions)
    arena = meta.get("arena", "Rocket League Arena")

    # Generate caption
    caption_source = "template"
    caption_i2v = None
    if use_llm and llm_endpoint:
        prompt = build_llm_prompt(actual_player_id, team, n_players, arena, physics_summary, action_summary)
        caption = call_llm(prompt, llm_endpoint, llm_model, llm_api_key)
        if caption:
            caption_source = "llm"
            # Also generate a template-based i2v caption for LLM entries
            _, caption_i2v = build_template_caption(actual_player_id, team, n_players, arena, physics_summary, action_summary)
        else:
            caption, caption_i2v = build_template_caption(actual_player_id, team, n_players, arena, physics_summary, action_summary)
            caption_source = "template_fallback"
    else:
        caption, caption_i2v = build_template_caption(actual_player_id, team, n_players, arena, physics_summary, action_summary)

    # Duration
    n_frames = physics_summary.get("n_frames", 80)
    duration = n_frames / 20.0

    return {
        "sample_id": f"mira_rocket:{match_id}_{chunk_key}.p{player}",
        "source_dataset": "mira_rocket_science",
        "source_sequence_id": match_id,
        "archive_path": tar_path,
        "video_member": filename,
        "video_path": video_path,
        "reference_image": None,
        "caption_detailed": caption,
        "caption_variants": [caption],
        "caption_i2v": caption_i2v,
        "annotation_source": f"mira_physics_{caption_source}",
        "fps": 20.0,
        "width": 1280,
        "height": 720,
        "duration_sec": round(duration, 2),
        "num_frames": n_frames,
        "split": "unassigned",
        "metadata": {
            "match_id": match_id,
            "chunk_id": chunk,
            "player_id": actual_player_id,
            "perspective_index": player,
            "team": team,
            "total_players": n_players,
            "arena": arena,
            "physics": physics_summary,
            "actions": action_summary,
        },
    }


# Module-level wrapper for picklable multiprocessing
_process_match_to_tar: dict[str, str] = {}
_process_use_llm = False
_process_llm_endpoint = ""
_process_llm_model = "gpt-4o-mini"
_process_llm_api_key = ""
_process_cache_db_path: Optional[str] = None


def _process_single(e: dict) -> dict | None:
    """Module-level wrapper callable by ProcessPoolExecutor."""
    return process_single_entry(
        e, _process_match_to_tar, _process_use_llm,
        _process_llm_endpoint, _process_llm_model, _process_llm_api_key,
        cache_db_path=_process_cache_db_path,
    )


def process_batch(
    entries: list[dict],
    match_to_tar: dict[str, str],
    output_path: Path,
    use_llm: bool = False,
    llm_endpoint: str = "",
    llm_model: str = "gpt-4o-mini",
    llm_api_key: str = "",
    limit: int = 0,
    resume: bool = True,
    num_workers: int = 1,
    cache_db_path: Optional[str] = None,
) -> int:
    """Process entries with multiprocessing and resume support."""
    # Resume: read existing
    processed_ids = set()
    if resume and output_path.exists():
        with output_path.open() as f:
            for line in f:
                if line.strip():
                    try:
                        processed_ids.add(json.loads(line.strip()).get("sample_id", ""))
                    except json.JSONDecodeError:
                        pass
        logger.info("Resume: already processed %d entries", len(processed_ids))

    # Filter: keep only entries we haven't processed
    to_process = []
    for entry in entries:
        video_path = entry.get("video_path", "")
        filename = os.path.basename(video_path)
        parsed = parse_manifest_filename(filename)
        if not parsed:
            continue
        sid = f"mira_rocket:{parsed['match_id']}_c{parsed['chunk']:05d}.p{parsed['player']}"
        if sid in processed_ids:
            continue
        to_process.append(entry)

    if limit > 0:
        to_process = to_process[:limit]

    total = len(to_process)
    logger.info("Processing %d entries (%d already done, %d skipped as unparseable)",
                total, len(processed_ids), len(entries) - len(to_process) - len(processed_ids))

    if num_workers > 1:
        # Multiprocess each entry using batched submission to avoid OOM with 2M futures
        global _process_match_to_tar, _process_use_llm, _process_llm_endpoint, _process_llm_model, _process_llm_api_key, _process_cache_db_path
        _process_match_to_tar = match_to_tar
        _process_use_llm = use_llm
        _process_llm_endpoint = llm_endpoint
        _process_llm_model = llm_model
        _process_llm_api_key = llm_api_key
        _process_cache_db_path = cache_db_path

        BATCH_SIZE = 5000
        count = 0
        t0 = time.time()
        total_processed = 0

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            with output_path.open("a") as out_f:
                for batch_start in range(0, total, BATCH_SIZE):
                    batch_end = min(batch_start + BATCH_SIZE, total)
                    batch = to_process[batch_start:batch_end]
                    logger.info("Submitting batch %d-%d of %d...", batch_start, batch_end, total)
                    futures = [pool.submit(_process_single, e) for e in batch]

                    for i, future in enumerate(as_completed(futures)):
                        result = future.result()
                        if result:
                            out_f.write(json.dumps(result) + "\n")
                            count += 1
                        total_processed += 1

                        if total_processed % 500 == 0:
                            elapsed = time.time() - t0
                            rate = total_processed / max(elapsed, 0.01)
                            eta = (total - total_processed) / max(rate, 0.01)
                            logger.info("Progress: %d/%d (%.1f%%) | done=%d | %.1f/s | ETA: %.0fs",
                                        total_processed, total, total_processed / total * 100,
                                        count, rate, eta)
    else:
        count = 0
        with output_path.open("a") as out_f:
            for i, entry in enumerate(to_process):
                result = process_single_entry(
                    entry, match_to_tar, use_llm, llm_endpoint, llm_model, llm_api_key,
                    cache_db_path=cache_db_path,
                )
                if result:
                    out_f.write(json.dumps(result) + "\n")
                    count += 1

                if (i + 1) % 100 == 0:
                    logger.info("Progress: %d/%d | done=%d", i + 1, total, count)

    logger.info("Done. Generated %d captions", count)
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-index", action="store_true",
                        help="Build tar index and cache to disk")
    parser.add_argument("--generate", action="store_true",
                        help="Generate captions from processed video manifest")
    parser.add_argument("--manifest", type=str,
                        default=str(VIDEO_PROCESSED_DIR / "manifest.jsonl"),
                        help="Path to processed video manifest JSONL")
    parser.add_argument("--output", type=str,
                        default=str(OUTPUT_BASE / "mira_captions_manifest.jsonl"),
                        help="Output manifest path")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit entries (0 = all)")
    parser.add_argument("--use-llm", action="store_true",
                        help="Use LLM API for captions")
    parser.add_argument("--llm-endpoint", type=str,
                        default=os.environ.get("MIRA_LLM_ENDPOINT", ""),
                        help="OpenAI-compatible API endpoint")
    parser.add_argument("--llm-model", type=str,
                        default=os.environ.get("MIRA_LLM_MODEL", "gpt-4o-mini"),
                        help="LLM model name")
    parser.add_argument("--no-resume", action="store_true",
                        help="Disable resume (overwrite output)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes")
    parser.add_argument("--use-cache", type=str, default=None,
                        help="Path to SQLite cache DB (e.g., dataset/mira_jsonl_cache/cache.db)")

    args = parser.parse_args()
    llm_api_key = os.environ.get("MIRA_LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

    if args.build_index:
        build_tar_index("test", TAR_INDEX_TEST)
        build_tar_index("train", TAR_INDEX_TRAIN)
        return

    if args.generate:
        # Load merged index from test + train
        match_to_tar = {}
        for idx_path in [TAR_INDEX_TEST, TAR_INDEX_TRAIN]:
            if idx_path.exists():
                with open(idx_path) as f:
                    match_to_tar.update(json.load(f))
        if not match_to_tar:
            logger.error("Index not found. Run --build-index first.")
            sys.exit(1)
        logger.info("Loaded match->tar index: %d match_ids", len(match_to_tar))

        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            logger.error("Manifest not found: %s", manifest_path)
            sys.exit(1)

        logger.info("Loading manifest...")
        entries = []
        with manifest_path.open() as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line.strip()))
        logger.info("Loaded %d entries", len(entries))

        use_llm = args.use_llm and args.llm_endpoint
        if args.use_llm and not args.llm_endpoint:
            logger.warning("LLM requested but no endpoint. Falling back to template captions.")

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        process_batch(
            entries=entries,
            match_to_tar=match_to_tar,
            output_path=output_path,
            use_llm=use_llm,
            llm_endpoint=args.llm_endpoint,
            llm_model=args.llm_model,
            llm_api_key=llm_api_key,
            limit=args.limit,
            resume=not args.no_resume,
            num_workers=args.workers,
            cache_db_path=args.use_cache,
        )

        logger.info("Output: %s", output_path)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
