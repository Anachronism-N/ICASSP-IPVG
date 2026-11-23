#!/usr/bin/env bash
# ============================================================================
# 阶段二自动衔接打标脚本
# NoXi 完成后自动启动 Charades，Charades 完成后自动启动 MIRA
# 在 Node B (PRIVATE_NODE_ADDRESS) 上运行，使用单个 vLLM 端点
#
# 用法: nohup bash stage2_auto_caption.sh > /tmp/stage2_auto.log 2>&1 & disown
# ============================================================================
set -euo pipefail

PROJECT="/srv/wm_teacher"
PYTHON="/root/.venv/bin/python"
API="http://PRIVATE_NODE_ADDRESS:8000/v1/chat/completions"
RECAP_DIR="${PROJECT}/training_metadata/recaptioned"
MANIFEST_DIR="${PROJECT}/training_metadata"

run_caption() {
    local ds_name="$1"
    local manifest="$2"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === 开始 ${ds_name} 全量打标 ==="

    $PYTHON "${PROJECT}/scripts/recaption_unified.py" \
        --manifest "${manifest}" \
        --datasets "${ds_name}" \
        --output-dir "${RECAP_DIR}" \
        --api-urls "${API}" \
        --concurrent 48

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === ${ds_name} 打标完成 ==="
    wc -l "${RECAP_DIR}/${ds_name}_recaptioned.jsonl" 2>/dev/null || true
}

# 1. NoXi (36744 待打标)
run_caption "noxi" "${MANIFEST_DIR}/manifest_noxi_full.jsonl"

# 2. Charades (63603 待打标)
run_caption "charades" "${MANIFEST_DIR}/manifest_charades_full.jsonl"

# 3. MIRA (2M+ 待打标 — 需要先生成 manifest)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === 生成 MIRA manifest ==="
$PYTHON "${PROJECT}/scripts/generate_dataset_manifest.py" --dataset mira \
    --output "${MANIFEST_DIR}/manifest_mira_full.jsonl"
run_caption "mira" "${MANIFEST_DIR}/manifest_mira_full.jsonl"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === 阶段二全部完成 ==="
