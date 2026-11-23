#!/bin/bash
# Run gen_baselines.py MODEL sharded over the given GPUs (default: launcher 0-7 + worker 0-6).
# Usage (on the launcher): [LAUNCHER_GPUS="0 1"] [WORKER_GPUS="0 1 2"] bash run_baseline.sh MODEL
# Logs: dataset/download_logs/bl_<MODEL>_s<k>.log
ROOT=/srv/wm_teacher
WORKER=PRIVATE_NODE_ADDRESS
M=$1
L=$ROOT/dataset/download_logs
LG=${LAUNCHER_GPUS-"0 1 2 3 4 5 6 7"}
WG=${WORKER_GPUS-"0 1 2 3 4 5 6"}  # worker GPU7 is faulty (Xid 31)
N=$(( $(echo $LG | wc -w) + $(echo $WG | wc -w) ))
cp $ROOT/scripts/gen_baselines.py /dockerdata/wm_tmp/gen_baselines.py
scp -q /dockerdata/wm_tmp/gen_baselines.py $WORKER:/dockerdata/wm_tmp/gen_baselines.py
ENV="cd /dockerdata/wm_tmp && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false"
s=0
for g in $LG; do
  bash -c "$ENV && CUDA_VISIBLE_DEVICES=$g setsid nohup /dockerdata/venv_bl/bin/python /dockerdata/wm_tmp/gen_baselines.py $M $s $N > $L/bl_${M}_s$s.log 2>&1 < /dev/null &"
  s=$((s + 1))
done
for g in $WG; do
  ssh -f -n -o BatchMode=yes $WORKER "$ENV && CUDA_VISIBLE_DEVICES=$g setsid nohup /dockerdata/venv_bl/bin/python /dockerdata/wm_tmp/gen_baselines.py $M $s $N > $L/bl_${M}_s$s.log 2>&1 < /dev/null &"
  s=$((s + 1))
done
echo "launched $N shards for baseline $M"
