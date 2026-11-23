#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-pilot}"
case "$MODE" in
  pilot)
    QUEUE="dataset/download_queues/project_aria_previews_pilot.jsonl"
    ;;
  all)
    QUEUE="dataset/download_queues/project_aria_previews_all.jsonl"
    ;;
  *)
    echo "Usage: $0 [pilot|all]" >&2
    exit 2
    ;;
esac

PROXY="${WM_HTTP_PROXY:-http://star-proxy.oa.com:3128}"
OUTPUT_ROOT="${PROJECT_ARIA_OUTPUT_ROOT:-dataset/project_aria_downloads}"
STATUS_LOG="${PROJECT_ARIA_STATUS_LOG:-dataset/download_queues/project_aria_download_status.jsonl}"
LOG_DIR="${PROJECT_ARIA_LOG_DIR:-dataset/download_logs}"
REPORT="${PROJECT_ARIA_REPORT:-docs/23_Project_Aria下载日志.md}"
WORKERS="${DOWNLOAD_WORKERS:-4}"
RETRIES="${DOWNLOAD_RETRIES:-3}"
TIMEOUT="${DOWNLOAD_TIMEOUT:-60}"
RUN_ID="$(date '+%Y%m%d_%H%M%S')"
LOG_PATH="$LOG_DIR/project_aria_${MODE}_${RUN_ID}.log"
LOCK_DIR="$LOG_DIR/.project_aria_download.lock"

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT" "$(dirname "$REPORT")"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Another Project Aria download process appears to be running: $LOCK_DIR" >&2
  exit 3
fi

export http_proxy="$PROXY"
export https_proxy="$PROXY"
export HTTP_PROXY="$PROXY"
export HTTPS_PROXY="$PROXY"

write_report() {
  local exit_code="$1"
  local finished_at audit_output temporary_report
  finished_at="$(date '+%Y-%m-%d %H:%M:%S %z')"
  audit_output="$(python3 dataset_tools/download_queue.py audit \
    --queue "$QUEUE" \
    --output-root "$OUTPUT_ROOT" \
    --verify-hash 2>&1 || true)"
  temporary_report="${REPORT}.tmp"

  {
    printf '# Project Aria 下载日志\n\n'
    printf '> 本文由 `scripts/download_project_aria.sh` 自动更新。完整日志保存在 `%s`。\n\n' "$LOG_PATH"
    printf '## 本次执行\n\n'
    printf -- '- 完成时间：`%s`\n' "$finished_at"
    printf -- '- 模式：`%s`\n' "$MODE"
    printf -- '- 队列：`%s`\n' "$QUEUE"
    printf -- '- 输出目录：`%s`\n' "$OUTPUT_ROOT"
    printf -- '- 退出码：`%s`\n' "$exit_code"
    printf -- '- 并发/重试/超时：`%s / %s / %ss`\n\n' "$WORKERS" "$RETRIES" "$TIMEOUT"
    printf '## 完整性审计\n\n```json\n%s\n```\n\n' "$audit_output"
    printf '## 日志末尾\n\n```text\n'
    tail -n 120 "$LOG_PATH" 2>/dev/null || true
    printf '```\n'
  } > "$temporary_report"
  mv "$temporary_report" "$REPORT"
  rmdir "$LOCK_DIR" 2>/dev/null || true
}

trap 'exit_code=$?; write_report "$exit_code"' EXIT

exec > >(tee -a "$LOG_PATH") 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] Project Aria download started"
echo "mode=$MODE queue=$QUEUE output_root=$OUTPUT_ROOT"
echo "workers=$WORKERS retries=$RETRIES timeout=${TIMEOUT}s"
echo "proxy_host=$(printf '%s' "$PROXY" | sed -E 's#(https?://)[^/@]+@#\1***@#')"

if [[ ! -s "$QUEUE" ]]; then
  echo "Queue is missing or empty: $QUEUE" >&2
  exit 4
fi

echo "Checking proxy DNS (a failure is logged, then the real URL request is still attempted)..."
getent hosts star-proxy.oa.com || echo "WARNING: proxy hostname did not resolve in this environment"

echo "Gate 1/2: downloading one item and validating size/SHA1..."
if ! python3 dataset_tools/download_queue.py download \
  --queue "$QUEUE" \
  --output-root "$OUTPUT_ROOT" \
  --status-log "$STATUS_LOG" \
  --workers 1 \
  --retries "$RETRIES" \
  --timeout "$TIMEOUT" \
  --max-items 1; then
  echo "Pilot gate failed; the remaining queue was not started." >&2
  exit 10
fi

echo "Gate 2/2: first item passed; downloading the selected $MODE queue..."
python3 dataset_tools/download_queue.py download \
  --queue "$QUEUE" \
  --output-root "$OUTPUT_ROOT" \
  --status-log "$STATUS_LOG" \
  --workers "$WORKERS" \
  --retries "$RETRIES" \
  --timeout "$TIMEOUT"

echo "Final SHA1 audit..."
python3 dataset_tools/download_queue.py audit \
  --queue "$QUEUE" \
  --output-root "$OUTPUT_ROOT" \
  --verify-hash

