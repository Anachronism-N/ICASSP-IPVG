#!/bin/bash
# Run GPU stages one after another across both nodes. A stage starts when no GPU job
# (gen_jobs.py / gen_baselines.py / vbench_eval.py / train.py) is alive on either node.
# Usage: bash gpu_queue.sh "stage cmd 1" "stage cmd 2" ...      (log: dataset/download_logs/gpu_queue.log)
WORKER=PRIVATE_NODE_ADDRESS
LOG=/srv/wm_teacher/dataset/download_logs/gpu_queue.log
busy() {
  for h in localhost $WORKER; do
    n=$(ssh -o BatchMode=yes $h "ps -eo args | awk '\$1 ~ /python/ && (index(\$0,\"gen_jobs.py\") || index(\$0,\"gen_baselines.py\") || index(\$0,\"vbench_eval.py\") || index(\$0,\"model_training/train.py\"))' | wc -l")
    [ "${n:-1}" -gt 0 ] && return 0
  done
  return 1
}
for stage in "$@"; do
  while busy; do sleep 120; done
  echo "$(date '+%F %T') START: $stage" >> $LOG
  bash -c "$stage" >> $LOG 2>&1
  sleep 180  # let the stage's processes come up before polling again
done
echo "$(date '+%F %T') QUEUE DONE" >> $LOG
