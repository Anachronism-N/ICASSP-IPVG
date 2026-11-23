#!/bin/bash
# Deploy Qwen3.5-35B-A3B (native multimodal MoE, 3B active) with vLLM for 5s video captioning.
# One independent server per GPU (weights ~67GB bf16 fit a 96GB H20), ports 8001..8008.
#
# Usage: bash scripts/deploy_qwen35_vl.sh [GPU_LIST]   e.g. "0" or "0 1 2 3 4 5 6 7"
# Logs:  dataset/download_logs/vllm_qwen35_gpu{N}.log
# Does NOT touch other users' GPU processes.
set -euo pipefail
ROOT=/srv/wm_teacher
MODEL=/dockerdata/model/Qwen3.5-35B-A3B
VLLM=/opt/conda/envs/torch-base/bin/vllm
GPUS=${1:-"0 1 2 3 4 5 6 7"}

for g in $GPUS; do
  port=$((8001 + g))
  VLLM_HOST_IP=127.0.0.1 CUDA_VISIBLE_DEVICES=$g NO_PROXY=localhost,127.0.0.1 \
  setsid nohup $VLLM serve "$MODEL" \
      --served-model-name qwen35-vl \
      --host 127.0.0.1 --port $port \
      --tensor-parallel-size 1 \
      --max-model-len 16384 \
      --gpu-memory-utilization 0.90 \
      --max-num-seqs 96 \
      --max-num-batched-tokens 32768 \
      --mm-processor-cache-gb 0 \
      --allowed-local-media-path "$ROOT" \
      --limit-mm-per-prompt '{"video": 1, "image": 0}' \
      --media-io-kwargs '{"video": {"num_frames": 16}}' \
      --mm-processor-kwargs '{"size": {"longest_edge": 262144, "shortest_edge": 4096}}' \
      --trust-remote-code \
      > "/dockerdata/wm_logs/vllm_qwen35_gpu${g}.log" 2>&1 < /dev/null &
  echo "GPU $g -> port $port (pid $!)"
done
