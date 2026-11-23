#!/bin/bash
# Resume CelebV-HQ YouTube downloads with cookies and proxy
# NOTE: set -e deliberately omitted so failures from yt-dlp don't abort the entire run

export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"
export PATH="$HOME/.deno/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CELEBV_DIR="${PROJECT_DIR}/dataset/digital_human/celebv_hq"
COOKIES="${PROJECT_DIR}/cookies_netscape.txt"
DOWNLOADED_FILE="${CELEBV_DIR}/downloaded.txt"
JSON_FILE="${CELEBV_DIR}/celebvhq_info.json"
RAW_DIR="${CELEBV_DIR}/downloaded_celebvhq/raw"

mkdir -p "$RAW_DIR"

echo "=== CelebV-HQ Resume Download ==="
echo "Proxy: $http_proxy"
echo "Cookies: $COOKIES"
echo "JSON: $JSON_FILE"
echo "Raw dir: $RAW_DIR"
echo "Downloaded log: $DOWNLOADED_FILE"
echo "Deno: $(which deno 2>/dev/null || echo 'not found')"
echo "yt-dlp: $(which yt-dlp)"
echo ""

# Read already downloaded ytb_ids from downloaded.txt
declare -A DOWNLOADED
if [ -f "$DOWNLOADED_FILE" ]; then
    while IFS= read -r line; do
        ytb_id=$(echo "$line" | awk '{print $NF}')
        [ -n "$ytb_id" ] && DOWNLOADED["$ytb_id"]=1
    done < "$DOWNLOADED_FILE"
fi

TOTAL_DOWNLOADED=${#DOWNLOADED[@]}
echo "Already downloaded: $TOTAL_DOWNLOADED videos"
echo ""

# Extract unique ytb_ids from celebvhq_info.json
echo "Extracting video IDs from JSON..."
YT_IDS=$(python3 -c "
import json
with open('$JSON_FILE') as f:
    data = json.load(f)
ids = sorted(set(v['ytb_id'] for v in data['clips'].values()))
for vid in ids:
    print(vid)
")

TOTAL=$(echo "$YT_IDS" | wc -l)
DOWNLOADING=0
FAILED=0
OK_COUNT=0
UNAVAILABLE=0

echo "Total unique ytb_ids to process: $TOTAL"
echo ""

while IFS= read -r ytb_id; do
    [ -z "$ytb_id" ] && continue

    if [ -n "${DOWNLOADED[$ytb_id]}" ]; then
        continue
    fi

    video_path="${RAW_DIR}/${ytb_id}.mp4"

    # Also check if file already exists on disk
    if [ -f "$video_path" ]; then
        echo "[SKIP] $ytb_id (file exists on disk)"
        echo "youtube $ytb_id" >> "$DOWNLOADED_FILE"
        continue
    fi

    DOWNLOADING=$((DOWNLOADING + 1))
    echo "[$DOWNLOADING/$((TOTAL - TOTAL_DOWNLOADED))] Downloading: $ytb_id"

    output=$(yt-dlp \
        --cookies "$COOKIES" \
        --remote-components ejs:github \
        --user-agent "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/PRIVATE_NODE_ADDRESS Safari/537.36" \
        -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio' \
        --sleep-requests 3 \
        --sleep-interval 5 \
        --max-sleep-interval 15 \
        --skip-unavailable-fragments \
        --merge-output-format mp4 \
        --output "$video_path" \
        --no-playlist \
        --ignore-errors \
        "https://www.youtube.com/watch?v=${ytb_id}" 2>&1)

    ret=$?

    # Check for specific error types
    if echo "$output" | grep -q "Video unavailable\|Private video\|video not available"; then
        echo "  [UNAVAILABLE] $ytb_id"
        echo "youtube $ytb_id" >> "$DOWNLOADED_FILE"
        UNAVAILABLE=$((UNAVAILABLE + 1))
    elif [ $ret -eq 0 ] && [ -f "$video_path" ]; then
        echo "youtube $ytb_id" >> "$DOWNLOADED_FILE"
        echo "  [OK] $ytb_id"
        OK_COUNT=$((OK_COUNT + 1))
    else
        FAILED=$((FAILED + 1))
        echo "  [FAILED] $ytb_id" | tee -a "${CELEBV_DIR}/failed.txt"
        # Show last line of output
        echo "  $(echo "$output" | tail -1)"
    fi
    echo ""

done <<< "$YT_IDS"

echo ""
echo "=== Summary ==="
echo "Total unique ytb_ids: $TOTAL"
echo "Already downloaded (prior): $TOTAL_DOWNLOADED"
echo "Newly downloaded (OK): $OK_COUNT"
echo "Unavailable/Private: $UNAVAILABLE"
echo "Failed: $FAILED"
echo "Done."
