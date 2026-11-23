#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNLOAD_LIST="$REPO_ROOT/dataset/egocentric_interaction/nymeria/download_list.txt"
OUTPUT_DIR="$REPO_ROOT/dataset/egocentric_interaction/nymeria"
LOG_DIR="$REPO_ROOT/dataset/download_logs"
RUN_ID="$(date '+%Y%m%d_%H%M%S')"
LOG_PATH="$LOG_DIR/nymeria_${RUN_ID}.log"
PROXY="${WM_HTTP_PROXY:-http://star-proxy.oa.com:3128}"
WORKERS="${DOWNLOAD_WORKERS:-8}"
TIMEOUT="${DOWNLOAD_TIMEOUT:-300}"

export http_proxy="$PROXY"
export https_proxy="$PROXY"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

exec > >(tee -a "$LOG_PATH") 2>&1

echo "[$(date)] Nymeria download started"
echo "workers=$WORKERS timeout=${TIMEOUT}s proxy=$PROXY"
echo "download_list=$DOWNLOAD_LIST output_dir=$OUTPUT_DIR"

if [[ ! -s "$DOWNLOAD_LIST" ]]; then
  echo "ERROR: Download list not found or empty: $DOWNLOAD_LIST"
  exit 1
fi

TOTAL=$(wc -l < "$DOWNLOAD_LIST")
echo "Total entries in list: $TOTAL"

# Generate curl command list, skipping already downloaded files
CMD_FILE=$(mktemp)
COUNT=0
SKIPPED=0

while IFS=$'\t' read -r url filename; do
  [[ -z "$url" || "$url" == \#* ]] && continue

  filepath="$OUTPUT_DIR/$filename"

  # Skip already downloaded non-empty files
  if [[ -s "$filepath" ]]; then
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  # Remove zero-byte leftovers
  [[ -f "$filepath" ]] && rm -f "$filepath"

  echo "curl -fSL --connect-timeout 30 --max-time $TIMEOUT -C - -o \"$filepath\" \"$url\" && echo \"OK: $filename\" || echo \"FAIL: $filename\"" >> "$CMD_FILE"
  COUNT=$((COUNT + 1))
done < "$DOWNLOAD_LIST"

echo "Files to download: $COUNT (skipped $SKIPPED already downloaded)"

if [[ $COUNT -eq 0 ]]; then
  echo "All files already downloaded!"
  rm -f "$CMD_FILE"
  exit 0
fi

echo "Starting parallel download with $WORKERS workers..."
# Run curl commands in parallel using xargs
cat "$CMD_FILE" | xargs -P "$WORKERS" -I {} bash -c '{}' 2>&1 | \
  awk -v total="$COUNT" '
    BEGIN { ok=0; fail=0; }
    /^OK:/ { ok++; }
    /^FAIL:/ { fail++; }
    {
      processed = ok + fail;
      if (processed % 20 == 0 || processed == total) {
        pct = int(processed * 100 / total);
        printf "Progress: %d/%d (%d%%) ok=%d fail=%d\n", processed, total, pct, ok, fail;
      }
    }
    END {
      printf "\n==========\n";
      printf "DONE. Total: %d | OK: %d | Failed: %d\n", total, ok, fail;
    }
  '

# Cleanup
rm -f "$CMD_FILE"

# Report on zero-byte files
EMPTY=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.mp4" -size 0 2>/dev/null | wc -l)
echo ""
echo "[$(date)] Zero-byte files remaining: $EMPTY"
echo "Log saved to: $LOG_PATH"

if [[ $EMPTY -gt 0 ]]; then
  echo "Re-run this script to retry failed/zero-byte downloads."
else
  echo "All downloads complete!"
fi

# Count total downloaded files and size
DOWNLOADED=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.mp4" -size +0 2>/dev/null | wc -l)
TOTAL_SIZE=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)
echo "Downloaded files: $DOWNLOADED, Total size: $TOTAL_SIZE"
