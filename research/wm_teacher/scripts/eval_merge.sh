#!/bin/bash
# Eval of weight-merged checkpoints (models/train/merge/*.safetensors, merge_ckpt.py) queued behind
# pipeline_ft2.sh phase E. Worker GPUs start as soon as phase-E generation ends (phase-E scoring only uses
# launcher GPUs); launcher GPUs join after "PIPELINE DONE". gen_jobs.py work-stealing balances the two.
# Usage (launcher, setsid nohup): MODELS=m_full1_a05 bash eval_merge.sh   Log: download_logs/eval_merge.log
ROOT=/srv/wm_teacher
WORKER=PRIVATE_NODE_ADDRESS
L=$ROOT/dataset/download_logs; LOG=$L/eval_merge.log; EV=$ROOT/eval; B=/dockerdata/wm_eval/bench
MODELS=${MODELS:-m_full1_a05}; TAG=merge
log() { echo "$(date '+%F %T') $*" >> $LOG; }
py_alive() { ssh -o BatchMode=yes $1 "ps -eo args | awk -v p='$2' '\$1 ~ /python/ && index(\$0, p)' | wc -l"; }
for f in gen_jobs.py build_eval_jobs.py build_extra_bench_jobs.py gt_ref_metrics.py summarize_eval.py score_pool.py vbench_eval.py build_teacher_review.py eval_val_loss.py score_extra.sh; do
  cp $ROOT/scripts/$f /dockerdata/wm_tmp/$f; scp -q /dockerdata/wm_tmp/$f $WORKER:/dockerdata/wm_tmp/$f; done
cd /dockerdata/wm_tmp
J=$EV/jobs_${TAG}_all.jsonl
if [ "${RESUME_GENERATION:-0}" != 1 ]; then
  python3 build_eval_jobs.py --suites wm,vbench --models $MODELS --tag _$TAG >> $LOG 2>&1
  python3 build_extra_bench_jobs.py --suites physiq,wmb --models $MODELS --tag _$TAG >> $LOG 2>&1
  # Preserve model-major order so the first model finishes first.
  python3 -c "import json,sys;M='$MODELS'.split(',');J=[json.loads(l) for f in sys.argv[1:] for l in open(f)];J.sort(key=lambda j:M.index(j['model']));print('\n'.join(map(json.dumps,J)))" $EV/jobs_wm_$TAG.jsonl $EV/jobs_vbench_$TAG.jsonl $EV/jobs_extra_$TAG.jsonl > $J
fi
[ -s "$J" ] || { log "MERGE EVAL INCOMPLETE: job manifest missing"; exit 1; }
log "queued $(wc -l < $J) videos for $MODELS"
ENV="cd $ROOT/DiffSynth-Studio && export PATH=/opt/conda/envs/torch-base/bin:\$PATH PYTHONPATH=$ROOT/DiffSynth-Studio DIFFSYNTH_MODEL_BASE_PATH=/dockerdata/models DIFFSYNTH_SKIP_DOWNLOAD=true PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false"

# Wait until the main pipeline has actually entered scoring. Starting merge generation
# while its broad wait_gone(gen_jobs.py) is still active would make the pipelines wait on each other.
phase_e_ready() {
  [ -f "$L/score_pool_ft2.log" ] || return 1
  python3 -c "import json,os;J=[json.loads(l) for l in open('$EV/jobs_ft2_all.jsonl')];raise SystemExit(not J or any(not os.path.isfile(j['out']) or os.path.getsize(j['out'])==0 for j in J))"
}
until phase_e_ready; do sleep 300; done
if [ "${RESUME_GENERATION:-0}" != 1 ]; then
  log "phase-E generation finished -> worker shards"
  for g in 0 1 2 3 4 5 6; do
    ssh -f -n -o BatchMode=yes $WORKER "$ENV && CUDA_VISIBLE_DEVICES=$g setsid nohup python /dockerdata/wm_tmp/gen_jobs.py $J $((8 + g)) 15 > $L/gen_${TAG}_s$((8 + g)).log 2>&1 < /dev/null &"
  done
fi
until grep -q 'vbench scoring done, failures=0' "$L/pipeline_ft2.log"; do sleep 120; done
log "phase-E VBench scoring finished -> launcher GPUs 2-7"
for g in 2 3 4 5 6 7; do
  bash -c "$ENV && CUDA_VISIBLE_DEVICES=$g setsid nohup python /dockerdata/wm_tmp/gen_jobs.py $J $g 15 > $L/gen_${TAG}_s$g.log 2>&1 < /dev/null &"
done
until grep -q "PIPELINE DONE" "$L/pipeline_ft2.log"; do
  if ! ps -eo args | grep -q '^bash /dockerdata/wm_tmp/pipeline_ft2.sh$'; then
    log "MERGE EVAL INCOMPLETE: phase-E orchestrator exited without completion"
    exit 1
  fi
  sleep 300
done
python3 summarize_eval.py --validate-jobs "$EV/jobs_ft2_all.jsonl" > "$L/phase_e_before_merge_coverage.log" 2>&1 || { log "MERGE EVAL INCOMPLETE: phase-E coverage failed"; exit 1; }
log "pipeline_ft2 done -> launcher GPUs 0-1"
for g in 0 1; do
  bash -c "$ENV && CUDA_VISIBLE_DEVICES=$g setsid nohup python /dockerdata/wm_tmp/gen_jobs.py $J $g 15 > $L/gen_${TAG}_s$g.log 2>&1 < /dev/null &"
done
sleep 300
while [ $(( $(py_alive localhost "gen_jobs.py $J") + $(py_alive $WORKER "gen_jobs.py $J") )) -gt 0 ]; do sleep 120; done
log "generation done: $(python3 -c "import json,os;J=[json.loads(l) for l in open('$J')];print(sum(os.path.exists(j['out']) for j in J),'/',len(J))")"
python3 -c "import json,os;J=[json.loads(l) for l in open('$J')];raise SystemExit(not J or any(not os.path.isfile(j['out']) or os.path.getsize(j['out'])==0 for j in J))" || { log "MERGE EVAL INCOMPLETE: missing generation outputs"; exit 1; }

# The worker is now free; score fixed validation noise separately from launcher judges.
g=0
for m in ${MODELS//,/ }; do
  ckpt=$ROOT/models/train/merge/${m#m_}.safetensors
  (ssh -o BatchMode=yes "$WORKER" "$ENV && CUDA_VISIBLE_DEVICES=$g /opt/conda/envs/torch-base/bin/python /dockerdata/wm_tmp/eval_val_loss.py --ckpt $ckpt --out $ROOT/models/train/round2_5s_lora_r32/eval/loss_$m.json" > "$L/eval_${m}_loss.log" 2>&1 && log "  validation loss scored: $m") &
  g=$((g + 1))
done

(CUDA_VISIBLE_DEVICES= THREADS=64 nice -n 5 /dockerdata/venv_vbench/bin/python gt_ref_metrics.py ${MODELS//,/ } > $L/gtref_$TAG.log 2>&1;
 log "  gtref: $(grep -a 'clips:' $L/gtref_$TAG.log | tr '\n' ' ')") &
(sed '/^# ---- WorldModelBench/,$d' score_extra.sh > score_piq_only_$TAG.sh
 nice -n 5 bash score_piq_only_$TAG.sh $MODELS > /dockerdata/wm_logs/score_piq_$TAG.out 2>&1
 log "  physiq: $(grep -h 'Physics-IQ score (original)' $L/score_physiq.log | tr '\n' ' ')") &
(for m in ${MODELS//,/ }; do
  (cd $B/WorldModelBench && CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$B/VILA /dockerdata/conda_vila/bin/python evaluation.py \
     --model_name $m --video_dir $EV/wmb/videos/$m --judge $B/judges/vila-ewm-qwen2-1.5b --save_name wmb_$m > $L/score_wmb_$m.log 2>&1 &&
   cp $B/WorldModelBench/wmb_$m.json $EV/wmb/results/$m.json && log "  wmb $m: $(grep -ao 'Total Score: [0-9.]*' $L/score_wmb_$m.log | tail -1)")
 done) &
/dockerdata/venv_vbench/bin/python score_pool.py wm:$MODELS vbench:$MODELS --conc 7 --gpus 0,1,2,3,4,5,6 > $L/score_pool_$TAG.log 2>&1
log "  vbench scoring done, failures=$(grep -c 'rc=1' $L/score_pool_$TAG.log)"
wait
python3 -c "import json,sys;from pathlib import Path;R=Path('$ROOT/models/train/round2_5s_lora_r32/eval');models='$MODELS'.split(',');ok=all((R/('loss_'+m+'.json')).is_file() and json.load((R/('loss_'+m+'.json')).open()).get('n')==120 for m in models);sys.exit(not ok)" || { log "MERGE EVAL INCOMPLETE: fixed validation losses missing"; exit 1; }
python3 summarize_eval.py --validate-jobs "$EV/jobs_ft2_all.jsonl" --validate-jobs "$J" > "$L/merge_final_coverage.log" 2>&1 || { log "MERGE EVAL INCOMPLETE: final coverage check failed"; exit 1; }
nice -n 10 /opt/conda/envs/torch-base/bin/python build_teacher_review.py >> $L/teacher_page.log 2>&1 || { log "MERGE EVAL INCOMPLETE: review-page build failed"; exit 1; }
log "MERGE EVAL DONE"
