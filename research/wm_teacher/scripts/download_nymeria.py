#!/usr/bin/env python3
"""Download Nymeria preview videos from download_list.txt with parallel workers."""

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOWNLOAD_LIST = REPO_ROOT / "dataset" / "egocentric_interaction" / "nymeria" / "download_list.txt"
OUTPUT_DIR = REPO_ROOT / "dataset" / "egocentric_interaction" / "nymeria"
LOG_DIR = REPO_ROOT / "dataset" / "download_logs"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = LOG_DIR / f"nymeria_{RUN_ID}.log"

PROXY = os.environ.get("WM_HTTP_PROXY", "http://star-proxy.oa.com:3128")
WORKERS = int(os.environ.get("DOWNLOAD_WORKERS", "8"))
TIMEOUT = int(os.environ.get("DOWNLOAD_TIMEOUT", "300"))

LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["http_proxy"] = PROXY
os.environ["https_proxy"] = PROXY
os.environ["HTTP_PROXY"] = PROXY
os.environ["HTTPS_PROXY"] = PROXY


class Tee:
    """Write to both stdout and a log file."""
    def __init__(self, path):
        self.file = open(path, "a")
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.file.flush()
        self.stdout.write(data)
        self.stdout.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()


sys.stdout = Tee(LOG_PATH)
sys.stderr = sys.stdout


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def load_download_list(path: Path) -> list[tuple[str, str]]:
    """Load URL, filename pairs from TSV download list."""
    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                url, filename = parts[0], parts[1]
                entries.append((url, filename))
    return entries


def download_file(url: str, filename: str) -> tuple[str, int, str]:
    """Download a single file using curl. Returns (filename, status_code, message)."""
    filepath = OUTPUT_DIR / filename

    # Skip existing non-empty files
    if filepath.exists() and filepath.stat().st_size > 0:
        return (filename, 2, f"skip {filepath.stat().st_size}")

    # Remove zero-byte files
    if filepath.exists():
        filepath.unlink()

    cmd = [
        "curl", "-fSL", "--connect-timeout", "30", "--max-time", str(TIMEOUT),
        "-o", str(filepath), url
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT + 30)
        if result.returncode == 0 and filepath.exists():
            size = filepath.stat().st_size
            if size > 0:
                return (filename, 0, f"ok {size}")
            else:
                filepath.unlink()
                return (filename, 1, "empty")
        else:
            filepath.unlink(missing_ok=True)
            return (filename, 1, f"curl_err:{result.returncode}")
    except subprocess.TimeoutExpired:
        filepath.unlink(missing_ok=True)
        return (filename, 1, "timeout")
    except Exception as e:
        filepath.unlink(missing_ok=True)
        return (filename, 1, f"error:{e}")


def main():
    log(f"Nymeria download started")
    log(f"workers={WORKERS} timeout={TIMEOUT}s proxy={PROXY}")
    log(f"download_list={DOWNLOAD_LIST} output_dir={OUTPUT_DIR}")

    if not DOWNLOAD_LIST.is_file() or DOWNLOAD_LIST.stat().st_size == 0:
        log("ERROR: Download list not found or empty")
        sys.exit(1)

    entries = load_download_list(DOWNLOAD_LIST)
    total = len(entries)
    log(f"Total files to process: {total}")

    ok_count = 0
    fail_count = 0
    skip_count = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(download_file, url, fname): fname for url, fname in entries}

        for i, future in enumerate(as_completed(futures), 1):
            filename, status, message = future.result()
            if status == 0:
                ok_count += 1
            elif status == 2:
                skip_count += 1
            else:
                fail_count += 1

            if i % 100 == 0 or i == total:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                log(f"Progress: {i}/{total} | ok={ok_count} skip={skip_count} fail={fail_count} | {rate:.1f} files/s")

    elapsed = time.time() - start_time
    log("=" * 50)
    log(f"Nymeria download completed in {elapsed/60:.1f} min")
    log(f"Total: {total} | Success: {ok_count} | Skipped: {skip_count} | Failed: {fail_count}")

    # Check for zero-byte files
    empty_files = []
    for f in OUTPUT_DIR.glob("*.mp4"):
        if f.stat().st_size == 0:
            empty_files.append(f.name)

    if empty_files:
        log(f"Empty files remaining: {len(empty_files)}")
        for fn in empty_files[:20]:
            log(f"  EMPTY: {fn}")
        if len(empty_files) > 20:
            log(f"  ... and {len(empty_files) - 20} more")
    else:
        log("No empty files remaining")

    log(f"Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
