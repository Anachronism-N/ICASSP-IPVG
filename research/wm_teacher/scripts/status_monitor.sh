#!/bin/bash
# Continuous status monitor: training step/loss, GPU, upload progress
# Appends snapshots to dataset/download_logs/status_snapshots.log every CHECK_INTERVAL sec
ROOT=/srv/wm_teacher
LOG=$ROOT/dataset/download_logs/status_snapshots.log
INTERVAL=${1:-300}

while true; do
    {
    echo "########## $(date '+%F %T') ##########"
    # Training workers
    n_workers=$(pgrep -fc "python3.10.*model_training/train.py" 2>/dev/null || echo 0)
    echo "train_workers=$n_workers"
    # Latest training progress (progress bar line with it/s) + any error
    tail -c 3000 "$ROOT/dataset/download_logs/train_round1_8gpu.log" 2>/dev/null | tr '\r' '\n' | grep -E "it/s\]|it/s," | tail -1
    tail -c 3000 "$ROOT/dataset/download_logs/train_round1_8gpu.log" 2>/dev/null | tr '\r' '\n' | grep -E "Error|Traceback|NaN" | tail -2
    # Latest checkpoints
    ls -t "$ROOT/models/train/round1_lora_r32/"*.safetensors 2>/dev/null | head -3
    # Tensorboard last loss
    /srv/PRIVATE_STORAGE - <<'EOF' 2>/dev/null
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
dirs = sorted(glob.glob('/srv/wm_teacher/models/train/round1_lora_r32/tensorboard_log/*'))
if dirs:
    ea = EventAccumulator(dirs[-1]); ea.Reload()
    if 'loss' in ea.Tags()['scalars']:
        ev = ea.Scalars('loss')
        print(f"tb_loss: step={ev[-1].step} loss={ev[-1].value:.4f} n_points={len(ev)}")
EOF
    # GPU
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | head -8 | tr '\n' ';'
    echo ""
    # Uploads
    for u in upload_mira upload_fix_continue; do
        alive=$(pgrep -f "scripts/$u.py" | wc -l)
        last=$(tail -c 1500 "$ROOT/dataset/download_logs/$u.log" 2>/dev/null | tr '\r' '\n' | grep -E "\[[0-9]+/[0-9]+\]|DONE|FAILED|✅|❌" | tail -1)
        echo "upload_$u alive=$alive | $last"
    done
    } >> "$LOG" 2>&1
    sleep "$INTERVAL"
done
