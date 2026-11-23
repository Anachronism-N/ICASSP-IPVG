#!/bin/bash
# Watchdog for round-2 training (2 nodes) + the wan22-5s HF upload. Run on the launcher node.
# Every INTERVAL s:
#   - training: if the torchrun job is gone on either node and the latest run didn't finish
#     cleanly, kill leftovers on both nodes and relaunch (launch_round2.sh -> new runN, initialised
#     from the newest checkpoint) on a random subset of round2_train.csv sized to the samples not
#     yet trained (DiffSynth can't skip already-seen samples on resume).
#   - upload: restart upload_5s.py if it died before finishing (it resumes from its state file);
#     gives up after 3 consecutive runs that uploaded nothing (e.g. HF storage quota 403).
#   - appends a status line to dataset/download_logs/status_round2.log
ROOT=/srv/wm_teacher
WORKER=PRIVATE_NODE_ADDRESS
OUT=$ROOT/models/train/round2_5s_lora_r32
LOGS=$ROOT/dataset/download_logs
L=/dockerdata/wm_logs
TOTAL_SAMPLES=$(( $(wc -l < $ROOT/training_metadata/round2/round2_train.csv) - 1 ))
WORLD=15
INTERVAL=${1:-300}
export HF_TOKEN=${HF_TOKEN:?set HF_TOKEN}

train_alive() {  # $1 = host
  ssh -o BatchMode=yes -o ConnectTimeout=20 $1 \
    'ps -eo args | awk "\$1 ~ /python/ && index(\$0,\"model_training/train.py\") && index(\$0,\"round2_5s_lora_r32\")" | wc -l' 2>/dev/null
}
kill_train() {
  for h in localhost $WORKER; do
    ssh -o BatchMode=yes $h 'ps -eo pid,args | awk "index(\$0,\"model_training/train.py\") || index(\$0,\"train_round2_16gpu.sh\")" | grep -v awk | awk "{print \$1}" | xargs -r kill -9' 2>/dev/null
  done
}
latest_run() { ls -d $OUT/run* 2>/dev/null | sed 's/.*run//' | sort -n | tail -1; }
steps_done() {  # samples already trained = sum over runs of (last step of the run) * WORLD
  local s=0 r last
  for r in $OUT/run*; do
    last=$(ls $r/step-*.safetensors 2>/dev/null | sed 's/.*step-\([0-9]*\).*/\1/' | sort -n | tail -1)
    s=$(( s + ${last:-0} ))
  done
  echo $s
}

upload_fail_streak=0
while true; do
  ts=$(date '+%F %T')
  run=$(latest_run)
  r0=$(train_alive localhost); r1=$(train_alive $WORKER)
  finished=$(grep -l "exit=0" $LOGS/train_round2_run${run}_rank0.log 2>/dev/null)
  step=$(grep -a "it\]" $LOGS/train_round2_run${run}_rank0.log 2>/dev/null | tr '\r' '\n' | grep -E "^ *[0-9]+%" | tail -1 | grep -oE "[0-9]+/[0-9]+ \[[^]]*\]")
  tstate="run$run procs=${r0:-?}+${r1:-?} $step"
  # hang: processes alive but tqdm hasn't written for 20 min (a step is ~20s, a save ~1 min)
  log0=$LOGS/train_round2_run${run}_rank0.log
  if [ -z "$finished" ] && grep -qE "it\]" $log0 2>/dev/null && \
     [ $(( $(date +%s) - $(stat -c %Y $log0) )) -gt 1200 ]; then
    r0=0; tstate="$tstate HUNG"
  fi
  if [ -f $L/ROUND2_TRAIN_DONE ]; then
    tstate="done"
  elif [ -n "$finished" ] && [ "${r0:-0}" = 0 ]; then
    touch $L/ROUND2_TRAIN_DONE; tstate="FINISHED run$run"
  elif [ "${r0:-0}" -lt 8 ] || [ "${r1:-0}" -lt 7 ]; then
    if [[ "$tstate" != *HUNG* ]]; then
      sleep 60  # transient (startup / checkpoint save)? re-check once before acting
      r0=$(train_alive localhost); r1=$(train_alive $WORKER)
    fi
    if [ "${r0:-0}" -lt 8 ] || [ "${r1:-0}" -lt 7 ]; then
      if ! grep -qE "it\]" $LOGS/train_round2_run${run}_rank0.log 2>/dev/null && \
         [ $(( $(date +%s) - $(stat -c %Y $LOGS/train_round2_run${run}_rank0.log) )) -lt 900 ]; then
        tstate="$tstate starting"
      else
        kill_train; sleep 20
        done_s=$(( $(steps_done) * WORLD ))
        left=$(( TOTAL_SAMPLES - done_s ))
        if [ $left -le $(( WORLD * 50 )) ]; then
          touch $L/ROUND2_TRAIN_DONE; tstate="DONE (crash near end, $done_s samples trained)"
        else
          csv=$ROOT/training_metadata/round2/round2_train_resume_$(date +%m%d%H%M).csv
          { head -1 $ROOT/training_metadata/round2/round2_train.csv
            tail -n +2 $ROOT/training_metadata/round2/round2_train.csv | shuf -n $left; } > $csv
          bash $ROOT/scripts/launch_round2.sh $csv >> $LOGS/status_round2.log 2>&1
          tstate="RELAUNCHED (trained=$done_s left=$left csv=$(basename $csv))"
        fi
      fi
    fi
  fi

  # post-training eval (launcher node, 8 healthy GPUs): final ckpt + mid-training ckpt
  if [ -f $L/ROUND2_TRAIN_DONE ] && [ ! -f $L/ROUND2_EVAL_STARTED ]; then
    final=$(ls -1t $OUT/run*/step-*.safetensors | head -1)
    mid=$(ls $OUT/run1/step-3000.safetensors 2>/dev/null)
    touch $L/ROUND2_EVAL_STARTED
    cp $ROOT/scripts/eval_round2.sh /dockerdata/wm_tmp/eval_round2.sh
    (cd /dockerdata/wm_tmp && setsid nohup bash /dockerdata/wm_tmp/eval_round2.sh $final $mid \
      > $LOGS/eval_round2.log 2>&1 < /dev/null &)
    tstate="$tstate | EVAL STARTED final=$(basename $(dirname $final))/$(basename $final)"
  elif [ -f $L/ROUND2_EVAL_STARTED ]; then
    grep -q "EVAL DONE" $LOGS/eval_round2.log 2>/dev/null && tstate="$tstate | eval done" || tstate="$tstate | eval running"
  fi

  # upload
  ustate=$(grep "upload_5s" $L/upload_5s.log 2>/dev/null | tail -1 | cut -c12-140)
  # match only the python process itself (other shells' command lines may mention the script)
  if ! ps -eo args | awk '$1 ~ /python/ && index($0, "wm_tmp/upload_5s.py") {f=1} END {exit !f}'; then
    if grep -q "windows_remaining=0 slicing_done=True" $L/upload_5s.log 2>/dev/null; then
      ustate="UPLOAD DONE"
    elif [ $upload_fail_streak -ge 3 ]; then
      ustate="UPLOAD STOPPED (3 runs without progress; last: $(grep -E 'FAIL|403' $L/upload_5s.log | tail -1 | cut -c1-120))"
    else
      ok=$(grep -c "shard ok" $L/upload_5s.log 2>/dev/null)
      [ "${ok:-0}" = 0 ] && upload_fail_streak=$((upload_fail_streak + 1)) || upload_fail_streak=0
      mv $L/upload_5s.log $L/upload_5s_$(date +%m%d%H%M).log
      (cd /dockerdata/wm_tmp && setsid nohup /opt/conda/envs/torch-base/bin/python /dockerdata/wm_tmp/upload_5s.py \
        > $L/upload_5s.log 2>&1 < /dev/null &)
      ustate="RESTARTED upload (streak=$upload_fail_streak)"
    fi
  fi
  nup=$(grep -h "shard ok" $L/upload_5s*.log 2>/dev/null | wc -l)
  echo "$ts | train: $tstate | upload: new_shards=$nup $ustate" >> $LOGS/status_round2.log
  sleep $INTERVAL
done
