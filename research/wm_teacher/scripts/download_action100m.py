#!/usr/bin/env python3
"""
Download Action100M videos from YouTube using yt-dlp with cookies + proxy.
Supports resume by tracking downloaded/failed/unavailable videos.
Uses anti-detection measures: proxy, rate limiting, random delays.
"""

import os
import sys
import subprocess
import time
import random
from pathlib import Path

import pyarrow.parquet as pq


# Configuration
PARQUET_DIR = "/srv/wm_teacher/dataset/egocentric_interaction/action100m/data"
OUTPUT_DIR = "/srv/wm_teacher/dataset/egocentric_interaction/action100m/videos"
COOKIES = "/srv/wm_teacher/cookies_netscape.txt"
DOWNLOADED_FILE = os.path.join(OUTPUT_DIR, "downloaded.txt")
FAILED_FILE = os.path.join(OUTPUT_DIR, "failed.txt")
UNAVAILABLE_FILE = os.path.join(OUTPUT_DIR, "unavailable.txt")
BOT_DETECTED_FILE = os.path.join(OUTPUT_DIR, "bot_detected.txt")
LOG_FILE = os.path.join(OUTPUT_DIR, "download_log.txt")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Proxy configuration (same as resume_yt_downloads.sh)
PROXY = "http://star-proxy.oa.com:3128"

USER_AGENTS = [
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/PRIVATE_NODE_ADDRESS Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/PRIVATE_NODE_ADDRESS Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/PRIVATE_NODE_ADDRESS Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:133.0) Gecko/20100101 Firefox/133.0",
]

MAX_RETRIES = 3
DOWNLOAD_TIMEOUT = 1200  # 20 minutes for long videos


def log_message(msg):
    """Write timestamped message to log file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {msg}\n")


def ensure_path():
    """Add deno to PATH for JS runtime support."""
    deno_bin = os.path.expanduser("~/.deno/bin")
    if deno_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = deno_bin + os.pathsep + os.environ.get("PATH", "")


def load_progress(filename):
    """Load set of already-processed video_uids."""
    processed = set()
    if os.path.exists(filename):
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if line:
                    processed.add(line)
    return processed


def mark_processed(filename, video_uid):
    """Append video_uid to a tracking file."""
    with open(filename, "a") as f:
        f.write(f"{video_uid}\n")


def extract_video_uids(parquet_dir):
    """Extract unique video_uids from all parquet files."""
    files = sorted([
        f for f in os.listdir(parquet_dir) if f.endswith(".parquet")
    ])
    print(f"Found {len(files)} parquet files")

    all_uids = []
    for pf in files:
        path = os.path.join(parquet_dir, pf)
        table = pq.read_table(path, columns=["video_uid"])
        uids = table.column("video_uid").to_pylist()
        all_uids.extend(uids)

    seen = set()
    unique_uids = []
    for uid in all_uids:
        if uid not in seen:
            seen.add(uid)
            unique_uids.append(uid)

    print(f"Total video_uids: {len(all_uids)}, unique: {len(unique_uids)}")
    return unique_uids


def check_cookies():
    """Check if cookies file exists and is recent."""
    if not os.path.exists(COOKIES):
        print(f"WARNING: Cookies file not found: {COOKIES}")
        return False
    age = time.time() - os.path.getmtime(COOKIES)
    age_days = age / 86400
    if age_days > 7:
        print(f"WARNING: Cookies file is {age_days:.0f} days old. YouTube sessions expire quickly.")
        print("Consider updating cookies_netscape.txt if downloads fail.")
    else:
        print(f"Cookies file is {age_days:.0f} days old - looks fresh.")
    return True


def download_video(video_uid, output_dir, cookies, attempt=1):
    """Download a single YouTube video with retry logic.
    Returns (success, reason)."""
    video_path = os.path.join(output_dir, f"{video_uid}.mp4")
    video_path_temp = os.path.join(output_dir, f"{video_uid}.mp4.part")

    # Clean up partial downloads from previous attempts
    if os.path.exists(video_path_temp):
        os.remove(video_path_temp)

    if os.path.exists(video_path):
        return True, "exists"

    url = f"https://www.youtube.com/watch?v={video_uid}"
    user_agent = random.choice(USER_AGENTS)

    cmd = [
        "yt-dlp",
        "--cookies", cookies,
        "--user-agent", user_agent,
        "--proxy", PROXY,
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
        "--skip-unavailable-fragments",
        "--merge-output-format", "mp4",
        "--output", video_path,
        "--no-playlist",
        "--ignore-errors",
        "--sleep-requests", str(random.uniform(2, 5)),
        "--sleep-interval", str(random.uniform(3, 8)),
        "--max-sleep-interval", str(random.uniform(10, 30)),
        "--socket-timeout", "30",
        "--retries", "5",
        "--extractor-retries", "3",
        url,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=DOWNLOAD_TIMEOUT,
            env={
                **os.environ,
                "http_proxy": PROXY,
                "https_proxy": PROXY,
            },
        )
        stderr = (result.stderr + result.stdout).lower()
        stdout = result.stdout

        if result.returncode == 0 and os.path.exists(video_path):
            return True, "ok"

        # Classify failure
        if "video unavailable" in stderr or "private video" in stderr or "not available" in stderr:
            return False, "unavailable"
        if "removed by the uploader" in stderr or "deleted" in stderr:
            return False, "unavailable"
        if "sign in to confirm" in stderr or "not a bot" in stderr or "bot" in stderr:
            if attempt < MAX_RETRIES:
                wait = 30 + random.uniform(10, 60)
                print(f"  [BOT_DETECTED] Retry {attempt}/{MAX_RETRIES} after {wait:.0f}s...")
                time.sleep(wait)
                return download_video(video_uid, output_dir, cookies, attempt + 1)
            return False, "bot_detected"
        if "copyright" in stderr or "content claimed" in stderr:
            return False, "copyright"
        if "rate-limit" in stderr or "too many requests" in stderr:
            if attempt < MAX_RETRIES:
                wait = 60 + random.uniform(30, 120)
                print(f"  [RATE_LIMITED] Retry {attempt}/{MAX_RETRIES} after {wait:.0f}s...")
                time.sleep(wait)
                return download_video(video_uid, output_dir, cookies, attempt + 1)
        if "timed out" in stderr or "connection" in stderr:
            if attempt < MAX_RETRIES:
                wait = 15 + random.uniform(5, 30)
                print(f"  [NETWORK_ERROR] Retry {attempt}/{MAX_RETRIES} after {wait:.0f}s...")
                time.sleep(wait)
                return download_video(video_uid, output_dir, cookies, attempt + 1)

        # Log full error for debugging
        log_message(f"FAILED {video_uid} (attempt {attempt})")
        log_message(f"STDOUT: {stdout[:500]}")
        log_message(f"STDERR: {stderr[:500]}")

        return False, "failed"
    except subprocess.TimeoutExpired:
        if attempt < MAX_RETRIES:
            print(f"  [TIMEOUT] Retry {attempt}/{MAX_RETRIES}...")
            return download_video(video_uid, output_dir, cookies, attempt + 1)
        return False, "timeout"
    except Exception as e:
        log_message(f"EXCEPTION {video_uid}: {e}")
        return False, f"error: {e}"


def main():
    ensure_path()

    # Migrate old bot-detected entries from failed.txt to bot_detected.txt
    # (from old script runs that didn't track bot_detected separately)
    if os.path.exists(LOG_FILE) and os.path.exists(FAILED_FILE):
        bot_uids = set()
        with open(LOG_FILE) as log_f:
            for line in log_f:
                if "BOT_DETECTED" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        uid = parts[1].strip("[]")
                        if len(uid) == 11:
                            bot_uids.add(uid)
        if bot_uids:
            failed_entries = load_progress(FAILED_FILE)
            failed_filtered = [uid for uid in failed_entries if uid not in bot_uids]
            moved = len(failed_entries) - len(failed_filtered)
            if moved > 0:
                with open(FAILED_FILE, "w") as f:
                    for uid in failed_filtered:
                        f.write(f"{uid}\n")
                for uid in bot_uids:
                    if uid in failed_entries:
                        mark_processed(BOT_DETECTED_FILE, uid)
                print(f"=> Migrated {moved} bot-detected entries from failed.txt to bot_detected.txt\n")

    # Clear old log
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    print("=" * 60)
    print("Action100M YouTube Downloader")
    print("=" * 60)
    print(f"Output:     {OUTPUT_DIR}")
    print(f"Cookies:    {COOKIES}")
    print(f"Proxy:      {PROXY}")
    print()

    check_cookies()

    # Load progress from tracking files
    # downloaded, failed, unavailable = permanently processed (skip)
    # bot_detected = temporary, will retry on next run
    downloaded = load_progress(DOWNLOADED_FILE)
    failed = load_progress(FAILED_FILE)
    unavailable = load_progress(UNAVAILABLE_FILE)
    bot_detected = load_progress(BOT_DETECTED_FILE)
    permanently_processed = downloaded | failed | unavailable

    print(f"\nPreviously downloaded:   {len(downloaded)}")
    print(f"Previously failed:       {len(failed)}")
    print(f"Previously unavailable:  {len(unavailable)}")
    print(f"Previously bot-detected: {len(bot_detected)} (will retry)")
    print(f"Total settled (skip):    {len(permanently_processed)}")
    print()

    # Extract video_uids
    print("Reading parquet files...")
    video_uids = extract_video_uids(PARQUET_DIR)

    # Filter: skip permanently processed, but retry bot_detected
    to_download = [uid for uid in video_uids if uid not in permanently_processed]
    print(f"\nVideos to download: {len(to_download)}")
    print()

    if len(to_download) == 0:
        print("Nothing to download. All videos already processed.")
        return

    log_message(f"Starting download of {len(to_download)} videos")

    ok_count = 0
    unavailable_count = 0
    fail_count = 0
    bot_count = 0

    for i, video_uid in enumerate(to_download):
        video_path = os.path.join(OUTPUT_DIR, f"{video_uid}.mp4")
        if os.path.exists(video_path):
            mark_processed(DOWNLOADED_FILE, video_uid)
            ok_count += 1
            print(f"[{i+1}/{len(to_download)}] [SKIP] {video_uid} (file exists)")
            continue

        print(f"[{i+1}/{len(to_download)}] Downloading: {video_uid}")
        success, reason = download_video(video_uid, OUTPUT_DIR, COOKIES)

        if success:
            mark_processed(DOWNLOADED_FILE, video_uid)
            ok_count += 1
            print(f"  [OK] {video_uid}")
        elif reason == "unavailable":
            mark_processed(UNAVAILABLE_FILE, video_uid)
            unavailable_count += 1
            print(f"  [UNAVAILABLE] {video_uid}")
        elif reason == "bot_detected":
            mark_processed(BOT_DETECTED_FILE, video_uid)
            bot_count += 1
            print(f"  [BOT_DETECTED] {video_uid} (will need retry)")
        else:
            mark_processed(FAILED_FILE, video_uid)
            fail_count += 1
            print(f"  [{reason.upper()}] {video_uid}")

        # Anti-detection: random delay between downloads
        delay = random.uniform(3, 10)
        time.sleep(delay)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total unique video_uids:      {len(video_uids)}")
    print(f"Permanently processed (skip): {len(permanently_processed)}")
    print(f"Bot detected (will retry):    {len(bot_detected)}")
    print(f"Newly downloaded (OK):        {ok_count}")
    print(f"Unavailable/removed:           {unavailable_count}")
    print(f"Bot detected (needs retry):   {bot_count}")
    print(f"Failed:                        {fail_count}")
    print()

    log_message(
        f"SUMMARY: ok={ok_count} unavailable={unavailable_count} "
        f"bot={bot_count} failed={fail_count}"
    )


if __name__ == "__main__":
    main()
