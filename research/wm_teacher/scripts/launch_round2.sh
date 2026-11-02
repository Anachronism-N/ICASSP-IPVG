#!/bin/bash
# Launch a LoRA training round on both nodes (run from the launcher node).
# Copies the training script to local disk on each node first: the shared fuse mount can serve a
# stale copy of a just-edited script.
# Usage: [TAG=round3] [INIT=ckpt] bash scripts/launch_round2.sh [TRAIN_CSV] [OUT_DIR] [extra train.py args]
# Logs: dataset/download_logs/train_${TAG}_run${RUN_ID}_rank{0,1}.log  (TAG defaults to round2)
ROOT=/srv/wm_teacher
WORKER=PRIVATE_NODE_ADDRESS
LOGDIR=$ROOT/dataset/download_logs
TAG=${TAG:-round2}
OUT=${2:-$ROOT/models/train/round2_5s_lora_r32}
RUN_ID=$(( $(ls -d $OUT/run* 2>/dev/null | wc -l) + 1 ))
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_round2_16gpu.sh}   # train_full_16gpu.sh for full fine-tuning
S=/dockerdata/wm_tmp/$TRAIN_SCRIPT
mkdir -p /dockerdata/wm_tmp && cp $ROOT/scripts/$TRAIN_SCRIPT $S
scp -q $S $WORKER:$S
ARGS=$(printf ' %q' "$@")
INIT_ENV="${INIT:+INIT=$INIT} ${PRESET_LORA:+PRESET_LORA=$PRESET_LORA} RDZV_TAG=$TAG FULL_ACT_OFFLOAD=${FULL_ACT_OFFLOAD:-0}"
# worker GPU7 (PCI d0:00) raised Xid 31 MMU faults (illegal memory access) under both vLLM and
# training -> excluded by default; override with WORKER_GPUS / LAUNCHER_GPUS.
WORKER_GPUS=${WORKER_GPUS:-0,1,2,3,4,5,6}
LAUNCHER_GPUS=${LAUNCHER_GPUS:-0,1,2,3,4,5,6,7}
ssh -f -n -o BatchMode=yes $WORKER "cd /dockerdata/wm_tmp && $INIT_ENV RUN_ID=$RUN_ID GPU_LIST=$WORKER_GPUS setsid nohup bash $S 1$ARGS > $LOGDIR/train_${TAG}_run${RUN_ID}_rank1.log 2>&1 < /dev/null &"
# subshell + setsid: rank 0 must not die with the (possibly short-lived) calling shell
(cd /dockerdata/wm_tmp && exec env $INIT_ENV RUN_ID=$RUN_ID GPU_LIST=$LAUNCHER_GPUS setsid nohup bash $S 0 "$@" \
  > $LOGDIR/train_${TAG}_run${RUN_ID}_rank0.log 2>&1 < /dev/null &)
echo "launched $TAG run$RUN_ID -> $LOGDIR/train_${TAG}_run${RUN_ID}_rank{0,1}.log"
