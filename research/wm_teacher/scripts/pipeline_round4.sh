#!/bin/bash
# Round4: aesthetic-filtered quality tuning, chained after pipeline_round3b (run on launcher, setsid nohup).
#  1. wait for pipeline_round3b DONE and the aesthetic scoring (worker) to finish
#  2. build round4 mixture (top-aesthetic per dataset, relative motion floor) and train
#     from Round3 final, lr 3e-5 cosine, 15 GPUs (one auto-retry from latest ckpt)
#  3. val loss + WM-Bench / VBench-I2V generations + GT-ref metrics + VBench scoring + summary
# Progress: dataset/download_logs/pipeline_round4.log
ROOT=/srv/wm_teacher
WORKER=PRIVATE_NODE_ADDRESS
L=$ROOT/dataset/download_logs
LOG=$L/pipeline_round4.log
OUT4=$ROOT/models/train/round4_aes_lora_r32
R3=$ROOT/models/train/round3_anneal_lora_r32/run1/step-2001.safetensors
log() { echo "$(date '+%F %T') $*" >> $LOG; }
py_alive() { ssh -o BatchMode=yes $1 "ps -eo args | awk -v p='$2' '\$1 ~ /python/ && index(\$0, p)' | wc -l"; }
wait_gone() { while [ $(( $(py_alive localhost "$1") + $(py_alive $WORKER "$1") )) -gt 0 ]; do sleep 120; done; }

log "phase 1: waiting for pipeline_round3b + aesthetic scoring"
until grep -q "PIPELINE DONE" $L/pipeline_round3b.log; do sleep 300; done
until grep -q "launched aesthetic scoring" $L/pipeline_round3b.log; do sleep 120; done
sleep 120
wait_gone score_aesthetic.py
log "  aesthetic scored: $(cat $ROOT/training_metadata/round4/aes_scores_*.jsonl | wc -l) windows"

log "phase 2: round4 mixture + training"
cp $ROOT/scripts/build_round4_mixture.py /dockerdata/wm_tmp/
(cd /dockerdata/wm_tmp && python3 build_round4_mixture.py --n-total 30000 >> $LOG 2>&1)
CSV=$ROOT/training_metadata/round4/round4_train.csv
[ $(wc -l < $CSV) -lt 20000 ] && { log "mixture too small, abort"; exit 1; }
cp $ROOT/scripts/launch_round2.sh /dockerdata/wm_tmp/launch_round2.sh
for attempt in 1 2; do
  init=$R3; [ $attempt -gt 1 ] && init=$(ls -t $OUT4/run*/step-*.safetensors 2>/dev/null | head -1)
  init=${init:-$R3}
  log "  launch attempt $attempt init=$init"
  (cd /dockerdata/wm_tmp && TAG=round4 INIT=$init bash /dockerdata/wm_tmp/launch_round2.sh $CSV $OUT4 \
     --learning_rate 3e-5 --lr_scheduler cosine --lr_warmup_steps 50 --min_lr_ratio 0.0 >> $LOG 2>&1)
  sleep 600
  wait_gone model_training/train.py
  run=$(ls -d $OUT4/run* | sort -V | tail -1 | xargs basename)
  if grep -q "exit=0" $L/train_round4_${run}_rank0.log 2>/dev/null && grep -q "exit=0" $L/train_round4_${run}_rank1.log 2>/dev/null; then
    break
  fi
  log "  training $run did not exit cleanly"
done
R4=$(ls -t $OUT4/run*/step-*.safetensors 2>/dev/null | head -1)
log "round4 ended, final ckpt=$R4"
[ -z "$R4" ] && { log "no round4 checkpoint, abort"; exit 1; }

log "phase 3: val loss + generations"
cp $ROOT/scripts/eval_val_loss.py /dockerdata/wm_tmp/
(cd $ROOT/DiffSynth-Studio && PATH=/opt/conda/envs/torch-base/bin:$PATH PYTHONPATH=$PWD DIFFSYNTH_MODEL_BASE_PATH=/dockerdata/models \
  DIFFSYNTH_SKIP_DOWNLOAD=true PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
  python /dockerdata/wm_tmp/eval_val_loss.py --ckpt $R4 \
  --out $ROOT/models/train/round2_5s_lora_r32/eval/loss_round4.json > $L/eval_round4_loss.log 2>&1)  # serial: shares GPU0 with gens otherwise (OOM)
cp $ROOT/scripts/build_eval_jobs.py /dockerdata/wm_tmp/
(cd /dockerdata/wm_tmp && python3 build_eval_jobs.py --suites wm,vbench --models round4 --tag _round4 >> $LOG 2>&1)
bash $ROOT/scripts/run_gen.sh $ROOT/eval/jobs_wm_round4.jsonl wm_round4 >> $LOG 2>&1
sleep 300; wait_gone gen_jobs.py
bash $ROOT/scripts/run_gen.sh $ROOT/eval/jobs_vbench_round4.jsonl vbench_round4 >> $LOG 2>&1
cp $ROOT/scripts/gt_ref_metrics.py /dockerdata/wm_tmp/
(cd /dockerdata/wm_tmp && CUDA_VISIBLE_DEVICES= THREADS=64 nice -n 5 /dockerdata/venv_vbench/bin/python /dockerdata/wm_tmp/gt_ref_metrics.py round4 \
  > $L/gtref_round4.log 2>&1) &
sleep 300; wait_gone gen_jobs.py
wait_gone eval_val_loss.py
log "  val loss: $(grep -o '"mean_loss": [0-9.]*' $ROOT/models/train/round2_5s_lora_r32/eval/loss_round4.json | head -1)"
log "  round4 videos: wm=$(ls $ROOT/eval/wm_bench/videos/round4 | grep -c 'mp4$') vbench=$(ls $ROOT/eval/vbench_i2v/videos/round4 | grep -c '\-0.mp4$')"

log "phase 4: VBench scoring"
cp $ROOT/scripts/vbench_eval.py $ROOT/scripts/score_pool.py /dockerdata/wm_tmp/
(cd /dockerdata/wm_tmp && /dockerdata/venv_vbench/bin/python /dockerdata/wm_tmp/score_pool.py \
  wm:round4 vbench:round4 --conc 16 > $L/score_pool_round4.log 2>&1)
log "  scoring done, failures=$(grep -c 'rc=1' $L/score_pool_round4.log)"
wait
cp $ROOT/scripts/summarize_eval.py /dockerdata/wm_tmp/
(cd /dockerdata/wm_tmp && python3 summarize_eval.py > /dev/null 2>&1)
log "PIPELINE DONE -> $ROOT/eval/summary.md"
