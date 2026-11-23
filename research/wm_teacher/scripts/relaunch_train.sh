#!/bin/bash
# Cleanup + relaunch round-1 training (run3)
ROOT=/srv/wm_teacher
# kill any stragglers using exact python3.10 binary match (won't match this script)
for p in $(pgrep -f "bin/python3.10.*model_training/train.py"); do kill -9 "$p" 2>/dev/null; done
sleep 2
mv "$ROOT/dataset/download_logs/train_round1_8gpu.log" "$ROOT/dataset/download_logs/train_round1_8gpu_run2.log" 2>/dev/null
cd "$ROOT"
setsid bash scripts/train_round1_8gpu.sh >> dataset/download_logs/train_round1_8gpu.log 2>&1 < /dev/null &
echo "relaunched pid $!"
