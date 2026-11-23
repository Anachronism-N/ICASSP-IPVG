#!/bin/bash
# Orchestrates: FT1 (running) -> dense recaption of the new data + eval-prompt refinement (vLLM) ->
# lean FT1 eval -> FT2 full fine-tune from base -> full eval of all models with prompt-optimized prompts.
# Replaces the tail of pipeline_full.sh. Run on the launcher (setsid nohup). Log: download_logs/pipeline_ft2.log
#  A. wait for FT1 training (crash -> resume on the unseen remainder, like pipeline_full)
#  B. vLLM Qwen3.5-35B-A3B: 2 TP4 servers on launcher (8001, 8002) + 1 on worker GPUs 0-3 (tunnel 8003)
#     - dense captions for dataset/wan22_5s_extra (ukdash, robomind, egodex, openvidgen)
#     - prompt refinement for Physics-IQ + WorldModelBench (VBench prompts_po.json already exists)
#  C. FT1 eval, raw protocol only (comparable to base/round4 now): val loss, WM-Bench, Physics-IQ, WMB
#  D. FT2: full FT from base, ft2_train.csv (dense captions only, camera-filtered in-domain + 40% supplementary)
#  E. eval: ft2 raw (WM, VBench, Physics-IQ, WMB, val loss, GT-ref) + prompt-optimized VBench / Physics-IQ /
#     WMB for base, round4, full1, ft2; scoring; summary; teacher page
ROOT=/srv/wm_teacher
WORKER=PRIVATE_NODE_ADDRESS
L=$ROOT/dataset/download_logs
LOG=$L/pipeline_ft2.log
EV=$ROOT/eval
B=/dockerdata/wm_eval/bench
NPROC=15
log() { echo "$(date '+%F %T') $*" >> $LOG; }
py_alive() { ssh -o BatchMode=yes $1 "ps -eo args | awk -v p='$2' '\$1 ~ /python/ && index(\$0, p)' | wc -l"; }
wait_gone() { while [ $(( $(py_alive localhost "$1") + $(py_alive $WORKER "$1") )) -gt 0 ]; do sleep 120; done; }
kill_train() { for h in localhost $WORKER; do ssh -o BatchMode=yes $h "ps -eo pid,args | awk '(\$2 ~ /python|torchrun/ && index(\$0,\"model_training/train.py\")) {print \$1}' | xargs -r kill -9"; done; }
cpw() { for f in "$@"; do cp $ROOT/scripts/$f /dockerdata/wm_tmp/$f; scp -q /dockerdata/wm_tmp/$f $WORKER:/dockerdata/wm_tmp/$f; done; }
val_loss() {  # val_loss CKPT NAME GPU
  (cd $ROOT/DiffSynth-Studio && PATH=/opt/conda/envs/torch-base/bin:$PATH PYTHONPATH=$PWD DIFFSYNTH_MODEL_BASE_PATH=/dockerdata/models \
    DIFFSYNTH_SKIP_DOWNLOAD=true PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$3 \
    python /dockerdata/wm_tmp/eval_val_loss.py --ckpt $1 --out $ROOT/models/train/round2_5s_lora_r32/eval/loss_$2.json > $L/eval_$2_loss.log 2>&1)
  log "  val loss $2: $(grep -o '"mean_loss": [0-9.]*' $ROOT/models/train/round2_5s_lora_r32/eval/loss_$2.json | head -1)"
}
wmb() {  # wmb MODEL GPU
  [ -f $EV/wmb/results/$1.json ] && return
  mkdir -p $EV/wmb/results
  (cd $B/WorldModelBench && CUDA_VISIBLE_DEVICES=$2 PYTHONPATH=$B/VILA /dockerdata/conda_vila/bin/python evaluation.py \
     --model_name $1 --video_dir $EV/wmb/videos/$1 --judge $B/judges/vila-ewm-qwen2-1.5b --save_name wmb_$1) > $L/score_wmb_$1.log 2>&1 &&
    cp $B/WorldModelBench/wmb_$1.json $EV/wmb/results/$1.json && log "  wmb scored: $1 $(grep -ao 'Total Score: [0-9.]*' $L/score_wmb_$1.log | tail -1)"
}
piq() {  # piq MODEL...  (CPU)
  sed '/^# ---- WorldModelBench/,$d' /dockerdata/wm_tmp/score_extra.sh > /dockerdata/wm_tmp/score_piq_only.sh
  nice -n 5 bash /dockerdata/wm_tmp/score_piq_only.sh "$(echo $@ | tr ' ' ,)" > /dockerdata/wm_logs/score_piq_$1.out 2>&1
  log "  physiq: $(grep -h 'Physics-IQ score (original)' $L/score_physiq.log | tr '\n' ' ')"
}
gen() {  # gen JOBS_JSONL TAG  -> all 15 GPUs, waits
  [ -s "$1" ] || { log "  no jobs in $1"; return; }
  log "  generating $(wc -l < $1) videos ($2)"
  bash $ROOT/scripts/run_gen.sh $1 $2 >> $LOG 2>&1
  sleep 300; wait_gone gen_jobs.py
}
train_full() {  # train_full OUT CSV LR TAG PRESET
  local OUT=$1 CSV0=$2 LR=$3 TAG=$4 PRESET=$5 TOTAL=$(( $(wc -l < $2) - 1 ))
  mkdir -p $OUT
  for attempt in 1 2 3; do
    local last=$(ls -1t $OUT/run*/step-*.safetensors 2>/dev/null | head -1) done_steps=0 csv lr warm act
    for d in $(ls -d $OUT/run* 2>/dev/null); do
      s=$(ls $d/step-*.safetensors 2>/dev/null | sed 's/.*step-\([0-9]*\).*/\1/' | sort -n | tail -1); done_steps=$(( done_steps + ${s:-0} )); done
    if [ -z "$last" ]; then csv=$CSV0; lr=$LR; warm=100; act=0
    else
      local left=$(( TOTAL - done_steps * NPROC )); [ $left -lt $(( NPROC * 50 )) ] && { log "  $TAG: only $left samples left, stop"; break; }
      csv=$ROOT/training_metadata/full/${TAG}_resume_$attempt.csv
      { head -1 $CSV0; tail -n +2 $CSV0 | shuf -n $left --random-source=<(yes $attempt); } > $csv
      lr=$(python3 -c "import math;print('%.3e'%($LR*0.5*(1+math.cos(math.pi*min(1,$done_steps*$NPROC/$TOTAL)))))"); warm=20; act=1
    fi
    if [ $attempt -gt 1 ] || [ -z "${ALREADY_RUNNING:-}" ]; then
      log "  $TAG attempt $attempt: $(basename $csv) ($(( $(wc -l < $csv) - 1 )) rows) lr=$lr act_offload=$act done_steps=$done_steps"
      (cd /dockerdata/wm_tmp && TRAIN_SCRIPT=train_full_16gpu.sh TAG=$TAG FULL_ACT_OFFLOAD=$act PRESET_LORA=$PRESET \
         bash /dockerdata/wm_tmp/launch_round2.sh $csv $OUT --learning_rate $lr --lr_scheduler cosine \
         --lr_warmup_steps $warm --min_lr_ratio 0.0 >> $LOG 2>&1)
      sleep 900
    fi
    ALREADY_RUNNING=
    wait_gone model_training/train.py
    local run=$(ls -d $OUT/run* | sort -V | tail -1 | xargs basename)
    if grep -q "exit=0" $L/train_${TAG}_${run}_rank0.log 2>/dev/null && grep -q "exit=0" $L/train_${TAG}_${run}_rank1.log 2>/dev/null; then
      log "  $TAG $run finished cleanly"; break
    fi
    log "  $TAG $run did not exit cleanly: $(grep -ahoE 'OutOfMemoryError|out of memory|CUDA error[^.]*|Error: [^ ]+' $L/train_${TAG}_${run}_rank*.log | sort | uniq -c | head -3 | tr '\n' ' ')"
    kill_train; sleep 30
  done
}
cpw gen_jobs.py build_eval_jobs.py build_extra_bench_jobs.py eval_val_loss.py score_extra.sh launch_round2.sh \
    train_full_16gpu.sh deploy_qwen35_vl_a100.sh caption_ctl.sh caption_5s_extra.py rewrite_prompts_vbench.py build_ft2_mixture.py

# ---------------- A
log "phase A: waiting for FT1 training"
ALREADY_RUNNING=1 train_full $ROOT/models/train/full1_5b $ROOT/training_metadata/round5/round5_train.csv 5e-6 full1 ""
F1=$(ls -t $ROOT/models/train/full1_5b/run*/step-*.safetensors 2>/dev/null | head -1)
log "FT1 final ckpt=$F1"

# ---------------- B
log "phase B: vLLM dense recaption + eval-prompt refinement"
until [ -f $ROOT/dataset/wan22_5s_extra/SLICE_DONE ]; do sleep 300; done
grep -q COPY_OK /dockerdata/wm_logs/copy_vllm.log || log "  WARNING: launcher vLLM copy missing"
MM='{"video": 1, "image": 1}'
(cd /dockerdata/wm_tmp && FIRST_GPU=0 MM_LIMIT="$MM" bash /dockerdata/wm_tmp/deploy_qwen35_vl_a100.sh 2 >> $LOG 2>&1)
ssh -o BatchMode=yes $WORKER "cd /dockerdata/wm_tmp && FIRST_GPU=0 MM_LIMIT='$MM' bash /dockerdata/wm_tmp/deploy_qwen35_vl_a100.sh 1" >> $LOG 2>&1
ssh -f -N -o BatchMode=yes -o ExitOnForwardFailure=yes -L 8003:127.0.0.1:8001 $WORKER
up=0
for i in $(seq 1 60); do
  up=0; for p in 8001 8002 8003; do curl -s --noproxy '*' -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$p/health | grep -q 200 && up=$((up+1)); done
  [ $up -eq 3 ] && break; sleep 30
done
log "  vLLM servers up: $up/3"
(cd /dockerdata/wm_tmp && /opt/conda/envs/torch-base/bin/python /dockerdata/wm_tmp/rewrite_prompts_vbench.py --port 8003 --suite physiq --conc 32 > $L/rewrite_physiq.log 2>&1;
 /opt/conda/envs/torch-base/bin/python /dockerdata/wm_tmp/rewrite_prompts_vbench.py --port 8003 --suite wmb --conc 32 > $L/rewrite_wmb.log 2>&1) &
(cd /dockerdata/wm_tmp && /opt/conda/envs/torch-base/bin/python /dockerdata/wm_tmp/caption_5s_extra.py --ports 8001-8002 --concurrency 256 > $L/caption_extra.log 2>&1)
wait
(cd /dockerdata/wm_tmp && /opt/conda/envs/torch-base/bin/python /dockerdata/wm_tmp/caption_5s_extra.py --ports 8001-8003 --concurrency 64 >> $L/caption_extra.log 2>&1)  # retry failures
log "  captions: $(wc -l < $ROOT/dataset/wan22_5s_extra/captions_5s.jsonl) / $(wc -l < $ROOT/dataset/wan22_5s_extra/manifest_5s_landscape.jsonl); $(tail -1 $L/rewrite_physiq.log); $(tail -1 $L/rewrite_wmb.log)"
pkill -f "ssh -f -N -o BatchMode=yes -o ExitOnForwardFailure=yes -L 8003" ; bash /dockerdata/wm_tmp/caption_ctl.sh stop_all >> $LOG 2>&1
ssh -o BatchMode=yes $WORKER "bash /dockerdata/wm_tmp/caption_ctl.sh stop_all" >> $LOG 2>&1
sleep 30
(cd /dockerdata/wm_tmp && python3 build_ft2_mixture.py >> $LOG 2>&1)

# ---------------- C
log "phase C: FT1 eval (raw protocol, comparable to base/round4)"
cd /dockerdata/wm_tmp
python3 build_eval_jobs.py --suites wm --models full1 --tag _full1 >> $LOG 2>&1
python3 build_extra_bench_jobs.py --suites physiq,wmb --models full1 --tag _full1 >> $LOG 2>&1
cat $EV/jobs_wm_full1.jsonl $EV/jobs_extra_full1.jsonl > $EV/jobs_full1_raw.jsonl
(val_loss $F1 full1 0) &
gen $EV/jobs_full1_raw.jsonl full1_raw
wait
cp $ROOT/scripts/gt_ref_metrics.py /dockerdata/wm_tmp/
(CUDA_VISIBLE_DEVICES= THREADS=64 nice -n 5 /dockerdata/venv_vbench/bin/python /dockerdata/wm_tmp/gt_ref_metrics.py full1 > $L/gtref_full1.log 2>&1) &
(piq full1) &
wmb full1 7 &
cp $ROOT/scripts/vbench_eval.py $ROOT/scripts/score_pool.py /dockerdata/wm_tmp/
/dockerdata/venv_vbench/bin/python /dockerdata/wm_tmp/score_pool.py wm:full1 --conc 7 > $L/score_pool_full1.log 2>&1 &
# ---------------- D (starts while CPU-side FT1 scoring finishes; WMB judge + VBench pool use launcher GPUs ~1 h)
wait
cp $ROOT/scripts/summarize_eval.py $ROOT/scripts/build_teacher_review.py $ROOT/scripts/build_demo.py /dockerdata/wm_tmp/
python3 summarize_eval.py > /dev/null 2>&1
log "phase D: FT2 full fine-tune from base"
[ $(wc -l < $ROOT/training_metadata/ft2/ft2_train.csv) -lt 20000 ] && { log "ft2 csv too small, abort"; exit 1; }
train_full $ROOT/models/train/ft2_5b $ROOT/training_metadata/ft2/ft2_train.csv 5e-6 ft2 NONE
F2=$(ls -t $ROOT/models/train/ft2_5b/run*/step-*.safetensors 2>/dev/null | head -1)
log "FT2 final ckpt=$F2"
[ -z "$F2" ] && { log "no FT2 checkpoint, abort"; exit 1; }

# ---------------- E
log "phase E: eval ft2 (raw) + prompt-optimized for base, round4, full1, ft2"
cd /dockerdata/wm_tmp
python3 build_eval_jobs.py --suites wm,vbench --models ft2 --tag _ft2 >> $LOG 2>&1
python3 build_eval_jobs.py --suites vbench --models full1,ft2 --tag _ft2po --prompt-map $EV/vbench_i2v/prompts_po.json --name-suffix _po >> $LOG 2>&1
python3 build_extra_bench_jobs.py --suites physiq,wmb --models ft2 --tag _ft2 >> $LOG 2>&1
python3 build_extra_bench_jobs.py --suites physiq,wmb --models base,round4,full1,ft2 --tag _po --po >> $LOG 2>&1
cat $EV/jobs_wm_ft2.jsonl $EV/jobs_vbench_ft2.jsonl $EV/jobs_vbench_ft2po.jsonl $EV/jobs_extra_ft2.jsonl $EV/jobs_extra_po.jsonl > $EV/jobs_ft2_all.jsonl
(val_loss $F2 ft2 0) &
gen $EV/jobs_ft2_all.jsonl ft2_all
wait
(CUDA_VISIBLE_DEVICES= THREADS=64 nice -n 5 /dockerdata/venv_vbench/bin/python /dockerdata/wm_tmp/gt_ref_metrics.py ft2 > $L/gtref_ft2.log 2>&1) &
(piq ft2 base_po round4_po full1_po ft2_po) &
(wmb ft2 0; wmb base_po 0; wmb round4_po 0) &
(wmb full1_po 1; wmb ft2_po 1) &
/dockerdata/venv_vbench/bin/python /dockerdata/wm_tmp/score_pool.py wm:ft2 vbench:ft2,full1_po,ft2_po --conc 12 > $L/score_pool_ft2.log 2>&1
log "  vbench scoring done, failures=$(grep -c 'rc=1' $L/score_pool_ft2.log)"
wait
python3 summarize_eval.py > /dev/null 2>&1
nice -n 10 /opt/conda/envs/torch-base/bin/python build_teacher_review.py >> $L/teacher_page.log 2>&1
log "PIPELINE DONE -> $EV/summary.md"
