#!/bin/bash
# Post-training evaluation of round 2 on one node (8 GPUs, one job per GPU):
#   val loss (eval_val_loss.py, 120 held-out 5s clips, fixed t/noise) for base / round1 / round2 ckpts
#   + 121-frame TI2V samples (inference_round2_cases.py) for base / round1 / round2-final
# Usage: bash eval_round2.sh ROUND2_FINAL_CKPT [ROUND2_MID_CKPT]
ROOT=/srv/wm_teacher
R2=$1; R2MID=${2:-}
R1=$ROOT/models/train/round1_lora_r32/step-14921.safetensors
OUT=$ROOT/models/train/round2_5s_lora_r32/eval
L=$ROOT/dataset/download_logs
mkdir -p $OUT /dockerdata/wm_tmp
cp $ROOT/scripts/eval_val_loss.py $ROOT/scripts/inference_round2_cases.py /dockerdata/wm_tmp/
cd $ROOT/DiffSynth-Studio
export PATH=/opt/conda/envs/torch-base/bin:$PATH PYTHONPATH=$ROOT/DiffSynth-Studio
export DIFFSYNTH_MODEL_BASE_PATH=/dockerdata/models DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false

run() {  # GPU NAME CMD...
  local g=$1 n=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g "$@" > $L/eval_round2_$n.log 2>&1 &
}
run 0 loss_base   python /dockerdata/wm_tmp/eval_val_loss.py --ckpt none --out $OUT/loss_base.json
run 1 loss_round1 python /dockerdata/wm_tmp/eval_val_loss.py --ckpt $R1 --out $OUT/loss_round1.json
run 2 loss_round2 python /dockerdata/wm_tmp/eval_val_loss.py --ckpt $R2 --out $OUT/loss_round2.json
[ -n "$R2MID" ] && run 3 loss_round2_mid python /dockerdata/wm_tmp/eval_val_loss.py --ckpt $R2MID --out $OUT/loss_round2_mid.json
run 4 gen_base    python /dockerdata/wm_tmp/inference_round2_cases.py NONE $OUT/gen_base
run 5 gen_round1  python /dockerdata/wm_tmp/inference_round2_cases.py $R1 $OUT/gen_round1
run 6 gen_round2  python /dockerdata/wm_tmp/inference_round2_cases.py $R2 $OUT/gen_round2
wait
echo "EVAL DONE $(date)"
