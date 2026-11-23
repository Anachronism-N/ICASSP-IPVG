#!/bin/bash
# Orchestrates the "optimize + related-work eval" phase. Run on the launcher with setsid nohup.
#  1. wait for the baseline workers (HunyuanVideo-I2V, Wan2.1-I2V-14B, dynamic jobs on all 15 GPUs)
#     and the VBench scorers still running from the first pass
#  2. Round3 = LR annealing: init from Round2 final, 30k windows (mostly unseen), lr 5e-5,
#     100-step warmup + cosine to 0, 121 frames, 15 GPUs
#  3. Round3 generations (WM-Bench + VBench-I2V) and val loss
#  4. VBench scoring of everything not scored yet (16 scorers spread over all 15 GPUs), summary
# Progress: dataset/download_logs/pipeline_round3.log
ROOT=/srv/wm_teacher
WORKER=PRIVATE_NODE_ADDRESS
L=$ROOT/dataset/download_logs
LOG=$L/pipeline_round3.log
OUT3=$ROOT/models/train/round3_anneal_lora_r32
R2=$ROOT/models/train/round2_5s_lora_r32/run2/step-1937.safetensors
log() { echo "$(date '+%F %T') $*" >> $LOG; }
py_alive() {  # host pattern -> number of python processes whose cmdline contains pattern
  ssh -o BatchMode=yes $1 "ps -eo args | awk -v p='$2' '\$1 ~ /python/ && index(\$0, p)' | wc -l"
}
wait_gone() {  # pattern: wait until no such python process on either node
  while [ $(( $(py_alive localhost "$1") + $(py_alive $WORKER "$1") )) -gt 0 ]; do sleep 120; done
}
copy() { cp $ROOT/scripts/$1 /dockerdata/wm_tmp/$1; }

log "phase 1: waiting for baseline workers + first-pass scorers"
wait_gone gen_baselines.py
wait_gone vbench_eval.py
for m in cogvideox hunyuan wan21; do
  log "  $m: $(ls $ROOT/eval/wm_bench/videos/$m 2>/dev/null | grep -c 'mp4$' ) videos, fails=$(cat $L/bl_${m}_*.log 2>/dev/null | grep -c FAIL)"
done

log "phase 2: round3 annealing"
for attempt in 1 2; do
  TAG=round3 INIT=$R2 bash $ROOT/scripts/launch_round2.sh $ROOT/training_metadata/round3/round3_train.csv $OUT3 \
    --learning_rate 5e-5 --lr_scheduler cosine --lr_warmup_steps 100 --min_lr_ratio 0.0 >> $LOG 2>&1
  sleep 600
  run=$(ls -d $OUT3/run* | sed 's/.*run//' | sort -n | tail -1)
  log "  run$run lr setup: $(grep -a '\[lr\]' $L/train_round3_run${run}_rank0.log | head -1)"
  wait_gone model_training/train.py
  if grep -q "exit=0" $L/train_round3_run${run}_rank0.log && grep -q "exit=0" $L/train_round3_run${run}_rank1.log; then
    log "round3 run$run finished OK: $(ls $OUT3/run$run | grep step | tail -1)"; break
  fi
  log "round3 run$run FAILED (attempt $attempt): $(grep -a Error $L/train_round3_run${run}_rank*.log | tail -1 | cut -c1-200)"
  for h in localhost $WORKER; do ssh -o BatchMode=yes $h 'ps -eo pid,args | awk "\$2 ~ /python/ && index(\$0,\"model_training/train.py\") {print \$1}" | xargs -r kill -9'; done
  sleep 30
done

log "phase 3: round3 generations + val loss"
copy build_eval_jobs.py
(cd /dockerdata/wm_tmp && python3 build_eval_jobs.py --suites wm,vbench --models round3 --tag _round3 >> $LOG 2>&1)
R3=$(ls -t $OUT3/run*/step-*.safetensors | head -1)
log "  round3 ckpt: $R3"
bash $ROOT/scripts/run_gen.sh $ROOT/eval/jobs_wm_round3.jsonl wm_round3 >> $LOG 2>&1
sleep 300; wait_gone gen_jobs.py
bash $ROOT/scripts/run_gen.sh $ROOT/eval/jobs_vbench_round3.jsonl vbench_round3 >> $LOG 2>&1
sleep 300; wait_gone gen_jobs.py
copy eval_val_loss.py
(cd $ROOT/DiffSynth-Studio && PATH=/opt/conda/envs/torch-base/bin:$PATH PYTHONPATH=$PWD DIFFSYNTH_MODEL_BASE_PATH=/dockerdata/models \
  DIFFSYNTH_SKIP_DOWNLOAD=true CUDA_VISIBLE_DEVICES=0 python /dockerdata/wm_tmp/eval_val_loss.py --ckpt $R3 \
  --out $ROOT/models/train/round2_5s_lora_r32/eval/loss_round3.json > $L/eval_round3_loss.log 2>&1) &
log "phase 3 done (val loss running in background on GPU0)"

log "phase 4: scoring"
copy vbench_eval.py; copy score_pool.py
(cd /dockerdata/wm_tmp && /dockerdata/venv_vbench/bin/python /dockerdata/wm_tmp/score_pool.py \
  wm:gt,base,round1,round2,round3,cogvideox,hunyuan,wan21 vbench:base,round2,round3 --conc 14 --gpus 1,2,3,4,5,6,7 \
  > $L/score_pool_round3.log 2>&1)
wait
copy summarize_eval.py
(cd /dockerdata/wm_tmp && python3 summarize_eval.py > /dev/null 2>&1)
log "PIPELINE DONE -> $ROOT/eval/summary.md"
