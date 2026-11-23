#!/bin/bash
# Full fine-tune (FT1) of the Wan2.2-TI2V-5B DiT + full evaluation (run on launcher, setsid nohup).
#  0. wait for the extra-benchmark generations (base/round4 on Physics-IQ + WorldModelBench) to finish
#  1. WorldModelBench judge for base + round4 (launcher GPU0/1), Physics-IQ round4 (CPU, background)
#  2. full FT on 15 GPUs: DeepSpeed ZeRO-2 + CPU Adam, init = base + Round4 LoRA fused,
#     data = round5_train.csv (motion+aesthetic selection, 30% short captions), lr 1e-5 cosine -> 0.
#     Crash -> resume from newest ckpt on the not-yet-seen remainder, with activation offload
#     (36 GB instead of 40.4 GB peak) and the lr continuing the cosine curve.
#  3. eval: val loss, WM-Bench, VBench raw + prompt-optimized, Physics-IQ, WorldModelBench, GT-ref,
#     VBench scoring, summary, teacher page
# Progress: dataset/download_logs/pipeline_full.log
ROOT=/srv/wm_teacher
WORKER=PRIVATE_NODE_ADDRESS
L=$ROOT/dataset/download_logs
LOG=$L/pipeline_full.log
EV=$ROOT/eval
OUT=$ROOT/models/train/full1_5b
CSV0=$ROOT/training_metadata/round5/round5_train.csv
LR=${LR:-5e-6}; WARM=100; NPROC=15  # 1e-5 run1: loss drifted up + NCCL OOM at the step-250 save
B=/dockerdata/wm_eval/bench
log() { echo "$(date '+%F %T') $*" >> $LOG; }
py_alive() { ssh -o BatchMode=yes $1 "ps -eo args | awk -v p='$2' '\$1 ~ /python/ && index(\$0, p)' | wc -l"; }
wait_gone() { while [ $(( $(py_alive localhost "$1") + $(py_alive $WORKER "$1") )) -gt 0 ]; do sleep 120; done; }
wmb() {  # wmb MODEL GPU
  [ -f $EV/wmb/results/$1.json ] && return
  mkdir -p $EV/wmb/results
  (cd $B/WorldModelBench && CUDA_VISIBLE_DEVICES=$2 PYTHONPATH=$B/VILA /dockerdata/conda_vila/bin/python evaluation.py \
     --model_name $1 --video_dir $EV/wmb/videos/$1 --judge $B/judges/vila-ewm-qwen2-1.5b --save_name wmb_$1) > $L/score_wmb_$1.log 2>&1 &&
    cp $B/WorldModelBench/wmb_$1.json $EV/wmb/results/$1.json && log "  wmb scored: $1"
}
cp $ROOT/scripts/{gen_jobs.py,build_eval_jobs.py,build_extra_bench_jobs.py,eval_val_loss.py,score_extra.sh,launch_round2.sh,train_full_16gpu.sh} /dockerdata/wm_tmp/
scp -q /dockerdata/wm_tmp/gen_jobs.py $WORKER:/dockerdata/wm_tmp/gen_jobs.py

if [ "${SKIP_PRE:-0}" != 1 ]; then
log "phase 0: waiting for extra-bench generations"
sleep 30; wait_gone gen_jobs.py
log "  physiq base=$(ls $EV/physiq/videos/base-op-run_01 | grep -c mp4$) round4=$(ls $EV/physiq/videos/round4-op-run_01 | grep -c mp4$) | wmb base=$(ls $EV/wmb/videos/base | grep -c mp4$) round4=$(ls $EV/wmb/videos/round4 | grep -c mp4$)"

log "phase 1: WorldModelBench judge base+round4 (GPU0/1), Physics-IQ round4 (CPU, bg)"
(sed '/^# ---- WorldModelBench/,$d' /dockerdata/wm_tmp/score_extra.sh > /dockerdata/wm_tmp/score_piq_only.sh
 nice -n 5 bash /dockerdata/wm_tmp/score_piq_only.sh round4 > /dockerdata/wm_logs/score_piq_round4.out 2>&1) &
wmb base 0 & p1=$!; wmb round4 1 & p2=$!; wait $p1 $p2

fi

log "phase 2: full fine-tune"
mkdir -p $OUT
TOTAL=$(( $(wc -l < $CSV0) - 1 ))
for attempt in 1 2 3; do
  last=$(ls -1t $OUT/run*/step-*.safetensors 2>/dev/null | head -1)
  # global steps done = sum over finished runs of their last checkpoint step
  done_steps=0; for d in $(ls -d $OUT/run* 2>/dev/null); do
    s=$(ls $d/step-*.safetensors 2>/dev/null | sed 's/.*step-\([0-9]*\).*/\1/' | sort -n | tail -1); done_steps=$(( done_steps + ${s:-0} )); done
  if [ $attempt = 1 ] && [ -z "$last" ]; then
    csv=$CSV0; lr=$LR; warm=$WARM; act=0
  else
    left=$(( TOTAL - done_steps * NPROC )); [ $left -lt $(( NPROC * 50 )) ] && { log "  only $left samples left, stop training"; break; }
    csv=$ROOT/training_metadata/full/full1_resume_$attempt.csv
    { head -1 $CSV0; tail -n +2 $CSV0 | shuf -n $left --random-source=<(yes $attempt); } > $csv
    lr=$(python3 -c "import math;print('%.3e'%($LR*0.5*(1+math.cos(math.pi*min(1,$done_steps*$NPROC/$TOTAL)))))")
    warm=20; act=1
  fi
  log "  attempt $attempt: csv=$(basename $csv) ($(( $(wc -l < $csv) - 1 )) rows) lr=$lr warmup=$warm act_offload=$act done_steps=$done_steps"
  (cd /dockerdata/wm_tmp && TRAIN_SCRIPT=train_full_16gpu.sh TAG=full1 FULL_ACT_OFFLOAD=$act \
     bash /dockerdata/wm_tmp/launch_round2.sh $csv $OUT --learning_rate $lr --lr_scheduler cosine \
     --lr_warmup_steps $warm --min_lr_ratio 0.0 >> $LOG 2>&1)
  sleep 900; wait_gone model_training/train.py
  run=$(ls -d $OUT/run* | sort -V | tail -1 | xargs basename)
  if grep -q "exit=0" $L/train_full1_${run}_rank0.log 2>/dev/null && grep -q "exit=0" $L/train_full1_${run}_rank1.log 2>/dev/null; then
    log "  $run finished cleanly"; break
  fi
  log "  $run did not exit cleanly: $(grep -ahoE 'OutOfMemoryError|CUDA error[^.]*|Error: [^ ]+' $L/train_full1_${run}_rank*.log | sort | uniq -c | head -3 | tr '\n' ' ')"
  for h in localhost $WORKER; do ssh -o BatchMode=yes $h "ps -eo pid,args | awk '(\$2 ~ /python|torchrun/ && index(\$0,\"model_training/train.py\")) {print \$1}' | xargs -r kill -9"; done
  sleep 30
done
F1=$(ls -t $OUT/run*/step-*.safetensors 2>/dev/null | head -1)
log "full FT ended, final ckpt=$F1"
[ -z "$F1" ] && { log "no checkpoint, abort"; exit 1; }

log "phase 3: eval"
(cd $ROOT/DiffSynth-Studio && PATH=/opt/conda/envs/torch-base/bin:$PATH PYTHONPATH=$PWD DIFFSYNTH_MODEL_BASE_PATH=/dockerdata/models \
  DIFFSYNTH_SKIP_DOWNLOAD=true PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
  python /dockerdata/wm_tmp/eval_val_loss.py --ckpt $F1 \
  --out $ROOT/models/train/round2_5s_lora_r32/eval/loss_full1.json > $L/eval_full1_loss.log 2>&1)
log "  val loss: $(grep -o '"mean_loss": [0-9.]*' $ROOT/models/train/round2_5s_lora_r32/eval/loss_full1.json | head -1)"
cd /dockerdata/wm_tmp
python3 build_eval_jobs.py --suites wm,vbench --models full1 --tag _full1 >> $LOG 2>&1
python3 build_eval_jobs.py --suites vbench --models full1 --tag _full1po --prompt-map $EV/vbench_i2v/prompts_po.json --name-suffix _po >> $LOG 2>&1
python3 build_extra_bench_jobs.py --suites physiq,wmb --models full1 --tag _full1 >> $LOG 2>&1
cat $EV/jobs_wm_full1.jsonl $EV/jobs_vbench_full1.jsonl $EV/jobs_vbench_full1po.jsonl $EV/jobs_extra_full1.jsonl 2>/dev/null > $EV/jobs_full1_all.jsonl
log "  $(wc -l < $EV/jobs_full1_all.jsonl) generation jobs"
bash $ROOT/scripts/run_gen.sh $EV/jobs_full1_all.jsonl full1_all >> $LOG 2>&1
sleep 300; wait_gone gen_jobs.py
cp $ROOT/scripts/gt_ref_metrics.py /dockerdata/wm_tmp/
(CUDA_VISIBLE_DEVICES= THREADS=64 nice -n 5 /dockerdata/venv_vbench/bin/python /dockerdata/wm_tmp/gt_ref_metrics.py full1 > $L/gtref_full1.log 2>&1) &
(sed '/^# ---- WorldModelBench/,$d' /dockerdata/wm_tmp/score_extra.sh > /dockerdata/wm_tmp/score_piq_only.sh; nice -n 5 bash /dockerdata/wm_tmp/score_piq_only.sh full1 > /dockerdata/wm_logs/score_piq_full1.out 2>&1) &
wmb full1 7 &
cp $ROOT/scripts/vbench_eval.py $ROOT/scripts/score_pool.py /dockerdata/wm_tmp/
/dockerdata/venv_vbench/bin/python /dockerdata/wm_tmp/score_pool.py wm:full1 vbench:full1,full1_po --conc 14 > $L/score_pool_full1.log 2>&1
log "  vbench scoring done, failures=$(grep -c 'rc=1' $L/score_pool_full1.log)"
wait
log "  physiq: $(grep -h 'Physics-IQ score (original)' $L/score_physiq.log | tr '\n' ' ')"
cp $ROOT/scripts/summarize_eval.py $ROOT/scripts/build_teacher_review.py $ROOT/scripts/build_demo.py /dockerdata/wm_tmp/
python3 summarize_eval.py > /dev/null 2>&1
nice -n 10 /opt/conda/envs/torch-base/bin/python build_teacher_review.py >> $L/teacher_page.log 2>&1
log "PIPELINE DONE -> $EV/summary.md"
