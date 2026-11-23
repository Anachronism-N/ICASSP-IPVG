#!/bin/bash
# Round5 + prompt-optimized VBench protocol, chained after pipeline_round4 (run on launcher, setsid nohup).
# Goal: close the VBench-I2V gap to DreamX-World-1.0 (90.79) on Wan2.2-TI2V-5B.
#  1. wait for pipeline_round4 DONE
#  2. vLLM Qwen3.5 (worker GPUs 0-3): rewrite the 355 VBench prompts (image + short prompt -> training-style
#     caption) and make short captions for the Round5 pool; then stop vLLM
#  3. prompt-optimized VBench gens for base + current best LoRA (15 GPUs)
#  4. Round5 training from the best LoRA: motion+aesthetic selection, 30% short captions, lr 3e-5 cosine
#  5. Round5 eval: val loss, WM-Bench, VBench raw + prompt-optimized, GT-ref, VBench scoring, summary, teacher page
# Progress: dataset/download_logs/pipeline_round5.log
ROOT=/srv/wm_teacher
WORKER=PRIVATE_NODE_ADDRESS
L=$ROOT/dataset/download_logs
LOG=$L/pipeline_round5.log
EV=$ROOT/eval
OUT5=$ROOT/models/train/round5_dyn_lora_r32
log() { echo "$(date '+%F %T') $*" >> $LOG; }
py_alive() { ssh -o BatchMode=yes $1 "ps -eo args | awk -v p='$2' '\$1 ~ /python/ && index(\$0, p)' | wc -l"; }
wait_gone() { while [ $(( $(py_alive localhost "$1") + $(py_alive $WORKER "$1") )) -gt 0 ]; do sleep 120; done; }
copy() { cp $ROOT/scripts/$1 /dockerdata/wm_tmp/$1; scp -q /dockerdata/wm_tmp/$1 $WORKER:/dockerdata/wm_tmp/$1; }
on_worker() { ssh -o BatchMode=yes $WORKER "$1"; }

log "phase 1: waiting for pipeline_round4"
until grep -qE "PIPELINE DONE|abort" $L/pipeline_round4.log; do sleep 300; done
sleep 60

# best LoRA so far = lowest WM-Bench DreamSim among round3/round4
BEST=$(python3 -c "
import json; g=json.load(open('$EV/summary.json')).get('wm_gtref',{})
c=[m for m in ('round4','round3') if m in g]; print(min(c,key=lambda m:g[m]['dreamsim']) if c else 'round3')")
BEST_CKPT=$(cd /dockerdata/wm_tmp && cp $ROOT/scripts/build_eval_jobs.py . && python3 -c "import build_eval_jobs as b; print(b.MODELS['$BEST'])")
log "best LoRA so far: $BEST ($BEST_CKPT)"

log "phase 2: vLLM prompt rewriting + short captions (worker GPUs 0-3)"
copy deploy_qwen35_vl_a100.sh; copy rewrite_prompts_vbench.py; copy shorten_captions.py; copy caption_ctl.sh
on_worker "cd /dockerdata/wm_tmp && FIRST_GPU=0 MM_LIMIT='{\"video\": 0, \"image\": 1}' bash /dockerdata/wm_tmp/deploy_qwen35_vl_a100.sh 1" >> $LOG 2>&1
for i in $(seq 1 60); do
  on_worker "curl -s --noproxy '*' -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8001/health" | grep -q 200 && break
  sleep 30
done
on_worker "cd /dockerdata/wm_tmp && /opt/conda/envs/torch-base/bin/python rewrite_prompts_vbench.py --port 8001 > $L/rewrite_prompts.log 2>&1"
log "  $(tail -1 $L/rewrite_prompts.log)"
on_worker "cd /dockerdata/wm_tmp && /opt/conda/envs/torch-base/bin/python shorten_captions.py --ports 8001 > $L/shorten_captions.log 2>&1"
log "  $(tail -1 $L/shorten_captions.log)"
on_worker "bash /dockerdata/wm_tmp/caption_ctl.sh stop_server 8001" >> $LOG 2>&1
sleep 30
cp $ROOT/scripts/build_round5_mixture.py /dockerdata/wm_tmp/
(cd /dockerdata/wm_tmp && python3 build_round5_mixture.py finalize 0.3 >> $LOG 2>&1)

NPO=$(python3 -c "import json;print(len(json.load(open('$EV/vbench_i2v/prompts_po.json'))))" 2>/dev/null || echo 0)
log "  rewritten VBench prompts: $NPO / 355"
if [ "$NPO" -ge 340 ]; then
log "phase 3: prompt-optimized VBench gens (base_po, ${BEST}_po)"
(cd /dockerdata/wm_tmp && python3 build_eval_jobs.py --suites vbench --models base,$BEST --tag _po \
   --prompt-map $EV/vbench_i2v/prompts_po.json --name-suffix _po >> $LOG 2>&1)
bash $ROOT/scripts/run_gen.sh $EV/jobs_vbench_po.jsonl vbench_po >> $LOG 2>&1
sleep 300; wait_gone gen_jobs.py
log "  po videos: base=$(ls $EV/vbench_i2v/videos/base_po | grep -c '\-0.mp4$') $BEST=$(ls $EV/vbench_i2v/videos/${BEST}_po | grep -c '\-0.mp4$')"
# score the po videos on the launcher while Round5 trains? No: training uses all GPUs. Score now (~1h).
cp $ROOT/scripts/vbench_eval.py $ROOT/scripts/score_pool.py /dockerdata/wm_tmp/
(cd /dockerdata/wm_tmp && /dockerdata/venv_vbench/bin/python /dockerdata/wm_tmp/score_pool.py \
  vbench:base_po,${BEST}_po --conc 16 > $L/score_pool_po.log 2>&1)
cp $ROOT/scripts/summarize_eval.py /dockerdata/wm_tmp/
(cd /dockerdata/wm_tmp && python3 summarize_eval.py > /dev/null 2>&1)
log "  po scoring done, failures=$(grep -c 'rc=1' $L/score_pool_po.log); see eval/summary.md"
else
  log "phase 3 skipped: prompt rewriting incomplete"
fi

log "phase 3b: extra public benchmarks (Physics-IQ original + WorldModelBench) for base, $BEST"
cp $ROOT/scripts/build_extra_bench_jobs.py $ROOT/scripts/build_eval_jobs.py /dockerdata/wm_tmp/
if (cd /dockerdata/wm_tmp && python3 build_extra_bench_jobs.py --suites physiq,wmb --models base,$BEST >> $LOG 2>&1); then
  bash $ROOT/scripts/run_gen.sh $EV/jobs_extra.jsonl extra >> $LOG 2>&1
  sleep 300; wait_gone gen_jobs.py
  log "  extra videos: physiq base=$(ls $EV/physiq/videos/base-op-run_01 | grep -c mp4$) $BEST=$(ls $EV/physiq/videos/$BEST-op-run_01 | grep -c mp4$) | wmb base=$(ls $EV/wmb/videos/base | grep -c mp4$) $BEST=$(ls $EV/wmb/videos/$BEST | grep -c mp4$)"
  (WMB_GPU=0 bash $ROOT/scripts/score_extra.sh base,$BEST >> $LOG 2>&1)
else
  log "  extra bench job build failed (data missing?), skipped"
fi

log "phase 4: Round5 training (init $BEST)"
CSV=$ROOT/training_metadata/round5/round5_train.csv
[ $(wc -l < $CSV) -lt 20000 ] && { log "round5 csv too small, abort"; exit 1; }
cp $ROOT/scripts/launch_round2.sh /dockerdata/wm_tmp/launch_round2.sh
for attempt in 1 2; do
  init=$BEST_CKPT; [ $attempt -gt 1 ] && init=$(ls -t $OUT5/run*/step-*.safetensors 2>/dev/null | head -1)
  init=${init:-$BEST_CKPT}
  log "  launch attempt $attempt init=$init"
  (cd /dockerdata/wm_tmp && TAG=round5 INIT=$init bash /dockerdata/wm_tmp/launch_round2.sh $CSV $OUT5 \
     --learning_rate 3e-5 --lr_scheduler cosine --lr_warmup_steps 50 --min_lr_ratio 0.0 >> $LOG 2>&1)
  sleep 600
  wait_gone model_training/train.py
  run=$(ls -d $OUT5/run* | sort -V | tail -1 | xargs basename)
  grep -q "exit=0" $L/train_round5_${run}_rank0.log 2>/dev/null && grep -q "exit=0" $L/train_round5_${run}_rank1.log 2>/dev/null && break
  log "  training $run did not exit cleanly"
done
R5=$(ls -t $OUT5/run*/step-*.safetensors 2>/dev/null | head -1)
log "round5 ended, final ckpt=$R5"
[ -z "$R5" ] && { log "no round5 checkpoint, abort"; exit 1; }

log "phase 5: Round5 eval"
cp $ROOT/scripts/eval_val_loss.py /dockerdata/wm_tmp/
(cd $ROOT/DiffSynth-Studio && PATH=/opt/conda/envs/torch-base/bin:$PATH PYTHONPATH=$PWD DIFFSYNTH_MODEL_BASE_PATH=/dockerdata/models \
  DIFFSYNTH_SKIP_DOWNLOAD=true PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
  python /dockerdata/wm_tmp/eval_val_loss.py --ckpt $R5 \
  --out $ROOT/models/train/round2_5s_lora_r32/eval/loss_round5.json > $L/eval_round5_loss.log 2>&1)  # serial: shares GPU0 with gens otherwise (OOM)
cp $ROOT/scripts/build_eval_jobs.py /dockerdata/wm_tmp/
(cd /dockerdata/wm_tmp && python3 build_eval_jobs.py --suites wm,vbench --models round5 --tag _round5 >> $LOG 2>&1)
[ "$NPO" -ge 340 ] && (cd /dockerdata/wm_tmp && python3 build_eval_jobs.py --suites vbench --models round5 --tag _round5po \
   --prompt-map $EV/vbench_i2v/prompts_po.json --name-suffix _po >> $LOG 2>&1)
(cd /dockerdata/wm_tmp && cp $ROOT/scripts/build_extra_bench_jobs.py . && python3 build_extra_bench_jobs.py --suites physiq,wmb --models round5 --tag _round5 >> $LOG 2>&1)
cat $EV/jobs_wm_round5.jsonl $EV/jobs_vbench_round5.jsonl $EV/jobs_vbench_round5po.jsonl $EV/jobs_extra_round5.jsonl 2>/dev/null > $EV/jobs_round5_all.jsonl
bash $ROOT/scripts/run_gen.sh $EV/jobs_round5_all.jsonl round5_all >> $LOG 2>&1
sleep 300; wait_gone gen_jobs.py
cp $ROOT/scripts/gt_ref_metrics.py /dockerdata/wm_tmp/
(cd /dockerdata/wm_tmp && CUDA_VISIBLE_DEVICES= THREADS=64 nice -n 5 /dockerdata/venv_vbench/bin/python /dockerdata/wm_tmp/gt_ref_metrics.py round5 \
  > $L/gtref_round5.log 2>&1) &
wait_gone eval_val_loss.py
log "  val loss: $(grep -o '"mean_loss": [0-9.]*' $ROOT/models/train/round2_5s_lora_r32/eval/loss_round5.json | head -1)"
(cd /dockerdata/wm_tmp && /dockerdata/venv_vbench/bin/python /dockerdata/wm_tmp/score_pool.py \
  wm:round5 vbench:round5,round5_po --conc 16 > $L/score_pool_round5.log 2>&1)
log "  scoring done, failures=$(grep -c 'rc=1' $L/score_pool_round5.log)"
(WMB_GPU=0 bash $ROOT/scripts/score_extra.sh round5 >> $LOG 2>&1)
wait
(cd /dockerdata/wm_tmp && python3 summarize_eval.py > /dev/null 2>&1)
cp $ROOT/scripts/{build_teacher_review.py,build_demo.py} /dockerdata/wm_tmp/
(cd /dockerdata/wm_tmp && nice -n 10 /opt/conda/envs/torch-base/bin/python build_teacher_review.py >> $L/teacher_page.log 2>&1)
log "PIPELINE DONE -> $EV/summary.md"
