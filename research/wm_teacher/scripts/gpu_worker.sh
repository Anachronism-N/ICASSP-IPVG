#!/bin/bash
# Per-GPU baseline worker: runs gen_baselines.py for each MODEL in turn on one GPU, taking jobs
# dynamically (lock-file claims), so GPUs can be added at any time and finish together.
# Usage (on any node): bash gpu_worker.sh GPU ID MODEL [MODEL ...]
#   waits until the GPU has < 2GB in use (lets a previous job release it), then works.
GPU=$1; ID=$2; shift 2
ROOT=/srv/wm_teacher
L=$ROOT/dataset/download_logs
cd /dockerdata/wm_tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)" -gt 2000 ]; do sleep 60; done
for M in "$@"; do
  CUDA_VISIBLE_DEVICES=$GPU /dockerdata/venv_bl/bin/python /dockerdata/wm_tmp/gen_baselines.py $M $ID 16 \
    >> $L/bl_${M}_w$ID.log 2>&1 < /dev/null
done
echo "$(date '+%F %T') worker $ID (gpu $GPU) done" >> $L/bl_workers.log
