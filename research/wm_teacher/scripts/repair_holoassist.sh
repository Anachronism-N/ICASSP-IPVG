#!/bin/bash
#
# repair_holoassist.sh - Repair corrupted HoloAssist tar files by resuming downloads
#
# Background:
#   3 of 4 tar files in dataset/egocentric_interaction/holoassist/data/ are corrupted
#   (truncated/Unexpected EOF) due to incomplete downloads.
#
#   Source: Azure Blob Storage (supports range requests for resume)
#   Base URL: https://hl2data.z5.web.core.windows.net/holoassist-data-release/
#
# Expected remote sizes:
#   ahat_depth.tar           601,785,251,840 bytes (~560 GB)
#   video_compress.tar       155,285,145,600 bytes (~145 GB)
#   video_pitch_shifted.tar  197,783,674,880 bytes (~184 GB)
#   eyes.tar                   2,634,506,240 bytes (~2.45 GB) - INTACT, skip
#
# Total redownload needed: ~889 GB
# Available disk space: ~197 TB

set -euo pipefail

# ---- Configuration ----
DATA_DIR="/srv/wm_teacher/dataset/egocentric_interaction/holoassist/data"
BASE_URL="https://hl2data.z5.web.core.windows.net/holoassist-data-release"
LOG_FILE="/srv/wm_teacher/scripts/repair_holoassist.log"

# Proxy settings
export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"
export no_proxy="localhost,127.0.0.1"

# Declare expected sizes (in bytes)
declare -A EXPECTED_SIZES
EXPECTED_SIZES["ahat_depth.tar"]=601785251840
EXPECTED_SIZES["video_compress.tar"]=155285145600
EXPECTED_SIZES["video_pitch_shifted.tar"]=197783674880
EXPECTED_SIZES["eyes.tar"]=2634506240

# ---- Logging ----
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log_separator() {
    echo "============================================================" | tee -a "$LOG_FILE"
}

# ---- Verification ----
verify_tar() {
    local tarfile="$1"
    local expected_size="$2"
    local local_size
    local_size=$(stat -c%s "$tarfile" 2>/dev/null || echo "0")

    # Check 1: File size
    if [ "$local_size" -ne "$expected_size" ]; then
        log "  FAIL: $tarfile size $local_size != expected $expected_size ($(( (expected_size - local_size) / 1024 / 1024 / 1024 )) GB missing)"
        return 1
    fi
    log "  OK: Size matches ($local_size bytes)"

    # Check 2: tar integrity
    if tar -tf "$tarfile" > /dev/null 2>&1; then
        log "  OK: tar integrity check passed"
        return 0
    else
        log "  FAIL: tar integrity check failed"
        return 1
    fi
}

# ---- Get remote size ----
get_remote_size() {
    local url="$1"
    local size
    size=$(curl -sI --max-time 30 "$url" 2>/dev/null | grep -i "^Content-Length:" | awk '{print $2}' | tr -d '\r')
    if [ -z "$size" ] || [ "$size" -eq 0 ] 2>/dev/null; then
        echo "0"
    else
        echo "$size"
    fi
}

# ---- Download with resume (wget) ----
download_resume() {
    local filename="$1"
    local url="${BASE_URL}/${filename}"
    local outpath="${DATA_DIR}/${filename}"

    log "  Starting resume download: $filename"
    log "  URL: $url"

    wget \
        --continue \
        --progress=dot:giga \
        --timeout=60 \
        --tries=5 \
        --retry-connrefused \
        --output-document="$outpath" \
        "$url" \
        2>&1 | tee -a "$LOG_FILE"

    return ${PIPESTATUS[0]}
}

# ---- Main repair logic ----
repair_file() {
    local filename="$1"
    local expected_size="${EXPECTED_SIZES[$filename]}"
    local filepath="${DATA_DIR}/${filename}"
    local url="${BASE_URL}/${filename}"

    log_separator
    log "Processing: $filename"

    # Check if file exists
    if [ ! -f "$filepath" ]; then
        log "  File does not exist, will download from scratch"
        local_size=0
    else
        local local_size
        local_size=$(stat -c%s "$filepath" 2>/dev/null || echo "0")
        local pct=$(( local_size * 100 / expected_size ))
        log "  Local size: $local_size bytes ($pct% of expected $expected_size)"

        if [ "$local_size" -eq "$expected_size" ]; then
            log "  File already at expected size, running integrity check..."
            if verify_tar "$filepath" "$expected_size"; then
                log "  File is intact, skipping."
                return 0
            else
                log "  File has correct size but failed integrity check. Will redownload."
                rm -f "$filepath"
            fi
        fi
    fi

    # Check if remote URL is accessible
    log "  Checking remote URL..."
    local remote_size
    remote_size=$(get_remote_size "$url")
    if [ "$remote_size" -eq 0 ]; then
        log "  ERROR: Cannot reach remote URL or got empty size response"
        log "  URL: $url"
        return 1
    fi
    log "  Remote size: $remote_size bytes"

    # Attempt resume download
    if download_resume "$filename"; then
        log "  Download completed. Verifying..."
        if verify_tar "$filepath" "$expected_size"; then
            log "  SUCCESS: $filename repaired!"
            return 0
        else
            log "  Verification failed after download. Attempting full redownload..."
            rm -f "$filepath"
            if download_resume "$filename" && verify_tar "$filepath" "$expected_size"; then
                log "  SUCCESS: $filename repaired (full redownload)!"
                return 0
            else
                log "  FAILURE: Could not repair $filename even after full redownload"
                return 1
            fi
        fi
    else
        log "  Download failed for $filename"
        return 1
    fi
}

# ---- Main ----
main() {
    log_separator
    log "HoloAssist Repair Script Started"
    log "Proxy: http_proxy=$http_proxy"
    log "Data directory: $DATA_DIR"
    log_separator

    # Verify data directory exists
    if [ ! -d "$DATA_DIR" ]; then
        log "ERROR: Data directory $DATA_DIR does not exist"
        exit 1
    fi

    # Verify base URL is reachable
    log "Verifying base URL accessibility..."
    if ! curl -sI --max-time 10 "${BASE_URL}/eyes.tar" 2>/dev/null | grep -q "200 OK"; then
        log "ERROR: Cannot reach Azure Blob Storage at $BASE_URL"
        log "Please check network/proxy settings."
        exit 1
    fi
    log "Azure Blob Storage is reachable."

    # Repair each corrupted file
    local files=("ahat_depth.tar" "video_compress.tar" "video_pitch_shifted.tar")
    local overall_status=0
    local results=()

    for f in "${files[@]}"; do
        if repair_file "$f"; then
            results+=("  [OK] $f: repaired")
        else
            results+=("  [FAIL] $f: could not repair")
            overall_status=1
        fi
    done

    # Skip eyes.tar (already intact, but verify anyway)
    log_separator
    log "Verifying eyes.tar (already intact)..."
    if verify_tar "${DATA_DIR}/eyes.tar" "${EXPECTED_SIZES[eyes.tar]}"; then
        results+=("  [OK] eyes.tar: intact (no action needed)")
        log "eyes.tar is intact."
    else
        results+=("  [FAIL] eyes.tar: integrity check failed")
        overall_status=1
    fi

    # Summary
    log_separator
    log "REPAIR SUMMARY:"
    for r in "${results[@]}"; do
        log "$r"
    done

    if [ "$overall_status" -eq 0 ]; then
        log "All files repaired successfully!"
    else
        log "Some files could not be repaired. See details above."
        log ""
        log "ALTERNATIVE DOWNLOAD SOURCES:"
        log "  1. ETH Zurich Research Collection: https://www.research-collection.ethz.ch/handle/20.500.11850/683960"
        log "  2. HoloAssist project page: https://holoassist.github.io/"
        log "  3. GitHub repo: https://github.com/Ember-HoloAssist/holoassist-release"
    fi
    log_separator

    exit "$overall_status"
}

main "$@"
