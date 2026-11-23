#!/bin/bash
# Follow-up after Round3 annealing (run on the launcher with setsid nohup):
#  1. wait for Round3 training (run1) to end; record final ckpt
#  2. Round3 generations on WM-Bench + VBench-I2V (15 GPUs)
#  3. Wan2.1-I2V-14B baseline with fp8 layerwise casting (dynamic workers, 15 GPUs) +
#     Round3 val loss on launcher GPU0 is started first
#  4. VBench scoring of everything missing + GT-referenced metrics (CPU) + summary
# Progress: dataset/download_logs/pipeline_round3b.log
ROOT=/srv/wm_teacher
WORKER=PRIVATE_NODE_ADDRESS
L=$ROOT/dataset/download_logs
LOG=$L/pipeline_round3b.log
OUT3=$ROOT/models/train/round3_anneal_lora_r32
log() { echo "$(date '+%F %T') $*" >> $LOG; }
py_alive() { ssh -o BatchMode=yes $1 "ps -eo args | awk -v p='$2' '\$1 ~ /python/ && index(\$0, p)' | wc -l"; }
wait_gone() { while [ $(( $(py_alive localhost "$1") + $(py_alive $WORKER "$1") )) -gt 0 ]; do sleep 120; done; }
copy() { cp $ROOT/scripts/$1 /dockerdata/wm_tmp/$1; scp -q /dockerdata/wm_tmp/$1 $WORKER:/dockerdata/wm_tmp/$1; }

log "phase 1: waiting for round3 training"
sleep 300
wait_gone model_training/train.py
ok0=$(grep -c "exit=0" $L/train_round3_run1_rank0.log); ok1=$(grep -c "exit=0" $L/train_round3_run1_rank1.log)
R3=$(ls -t $OUT3/run*/step-*.safetensors 2>/dev/null | head -1)
log "round3 ended: exit_ok rank0=$ok0 rank1=$ok1, final ckpt=$R3"
[ -z "$R3" ] && { log "no round3 checkpoint, abort"; exit 1; }

log "phase 2: round3 val loss + generations"
cp $ROOT/scripts/eval_val_loss.py /dockerdata/wm_tmp/
(cd $ROOT/DiffSynth-Studio && PATH=/opt/conda/envs/torch-base/bin:$PATH PYTHONPATH=$PWD DIFFSYNTH_MODEL_BASE_PATH=/dockerdata/models \
  DIFFSYNTH_SKIP_DOWNLOAD=true PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
  python /dockerdata/wm_tmp/eval_val_loss.py --ckpt $R3 \
  --out $ROOT/models/train/round2_5s_lora_r32/eval/loss_round3.json > $L/eval_round3_loss.log 2>&1)
log "  val loss: $(tail -c 2000 $L/eval_round3_loss.log | grep -o '"mean_loss": [0-9.]*' | head -1)"
cp $ROOT/scripts/build_eval_jobs.py /dockerdata/wm_tmp/
(cd /dockerdata/wm_tmp && python3 build_eval_jobs.py --suites wm,vbench --models round3 --tag _round3 >> $LOG 2>&1)
bash $ROOT/scripts/run_gen.sh $ROOT/eval/jobs_wm_round3.jsonl wm_round3 >> $LOG 2>&1
sleep 300; wait_gone gen_jobs.py
bash $ROOT/scripts/run_gen.sh $ROOT/eval/jobs_vbench_round3.jsonl vbench_round3 >> $LOG 2>&1
sleep 300; wait_gone gen_jobs.py
log "  round3 videos: wm=$(ls $ROOT/eval/wm_bench/videos/round3 | grep -c 'mp4$') vbench=$(ls $ROOT/eval/vbench_i2v/videos/round3 | grep -c '\-0.mp4$')"

log "phase 3: wan21 (fp8 layerwise) baseline on all GPUs + scoring of finished suites on launcher"
copy gen_baselines.py; copy gpu_worker.sh
for g in 0 1 2 3 4 5 6 7; do (cd /dockerdata/wm_tmp && setsid nohup bash /dockerdata/wm_tmp/gpu_worker.sh $g $g wan21 > /dev/null 2>&1 < /dev/null &); done
for g in 0 1 2 3 4 5 6; do ssh -f -n -o BatchMode=yes $WORKER "cd /dockerdata/wm_tmp && setsid nohup bash /dockerdata/wm_tmp/gpu_worker.sh $g $((8+g)) wan21 > /dev/null 2>&1 < /dev/null &"; done
cp $ROOT/scripts/gt_ref_metrics.py /dockerdata/wm_tmp/
(cd /dockerdata/wm_tmp && CUDA_VISIBLE_DEVICES= THREADS=64 nice -n 5 /dockerdata/venv_vbench/bin/python /dockerdata/wm_tmp/gt_ref_metrics.py round3 \
  > $L/gtref_round3.log 2>&1) &
sleep 600
log "  wan21 speed: $(grep -h 's/video' $L/bl_wan21_w*.log | tail -1) | OOM switches: $(grep -h OOM $L/bl_wan21_w*.log | wc -l)"
wait_gone gen_baselines.py
log "  wan21: $(ls $ROOT/eval/wm_bench/videos/wan21 | grep -c 'mp4$') videos, fails=$(cat $L/bl_wan21_w*.log | grep -c FAIL)"
(cd /dockerdata/wm_tmp && CUDA_VISIBLE_DEVICES= THREADS=64 nice -n 5 /dockerdata/venv_vbench/bin/python /dockerdata/wm_tmp/gt_ref_metrics.py wan21 \
  > $L/gtref_wan21.log 2>&1) &

log "phase 4: VBench scoring"
cp $ROOT/scripts/vbench_eval.py $ROOT/scripts/score_pool.py /dockerdata/wm_tmp/
(cd /dockerdata/wm_tmp && /dockerdata/venv_vbench/bin/python /dockerdata/wm_tmp/score_pool.py \
  wm:gt,base,round1,round2,round3,cogvideox,hunyuan,wan21 vbench:base,round2,round3 --conc 16 \
  > $L/score_pool_round3.log 2>&1)
log "  scoring done, failures=$(grep -c 'rc=1' $L/score_pool_round3.log)"
wait
cp $ROOT/scripts/summarize_eval.py /dockerdata/wm_tmp/
(cd /dockerdata/wm_tmp && python3 summarize_eval.py > /dev/null 2>&1)
log "PIPELINE DONE -> $ROOT/eval/summary.md"
