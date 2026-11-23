#!/bin/bash
# Deploy Qwen3.5-35B-A3B for 5s video captioning on A100-40GB nodes.
# bf16 weights (~67GB) don't fit one 40GB GPU -> TP=4, two servers per node (ports 8001, 8002).
# Same sampling/video settings as the H20 deployment (deploy_qwen35_vl.sh) so captions stay consistent.
#
# Usage: bash scripts/deploy_qwen35_vl_a100.sh [N_SERVERS]   (default 2 -> GPUs 0-3 and 4-7)
# Logs:  /dockerdata/wm_logs/vllm_qwen35_tp4_{i}.log
ROOT=/srv/wm_teacher
MODEL=/dockerdata/model/Qwen3.5-35B-A3B
VLLM=/dockerdata/venv_vllm/bin/vllm
N=${1:-2}
FIRST=${FIRST_GPU:-0}   # e.g. FIRST_GPU=0 N=1 -> GPUs 0-3
MM_LIMIT=${MM_LIMIT:-'{"video": 1, "image": 0}'}  # prompt rewriting needs '{"video": 0, "image": 1}'

for i in $(seq 0 $((N - 1))); do
  port=$((8001 + i))
  gpus=$(seq -s, $((FIRST + i * 4)) $((FIRST + i * 4 + 3)))
  VLLM_HOST_IP=127.0.0.1 CUDA_VISIBLE_DEVICES=$gpus NO_PROXY=localhost,127.0.0.1 \
  setsid nohup $VLLM serve "$MODEL" \
      --served-model-name qwen35-vl \
      --host 127.0.0.1 --port $port \
      --tensor-parallel-size 4 \
      --max-model-len 16384 \
      --gpu-memory-utilization 0.90 \
      --max-num-seqs 128 \
      --max-num-batched-tokens 32768 \
      --mm-processor-cache-gb 0 \
      --allowed-local-media-path "$ROOT" \
      --limit-mm-per-prompt "$MM_LIMIT" \
      --media-io-kwargs '{"video": {"num_frames": 16}}' \
      --mm-processor-kwargs '{"size": {"longest_edge": 262144, "shortest_edge": 4096}}' \
      --trust-remote-code \
      > "/dockerdata/wm_logs/vllm_qwen35_tp4_${i}.log" 2>&1 < /dev/null &
  echo "GPUs $gpus -> port $port (pid $!)"
done
