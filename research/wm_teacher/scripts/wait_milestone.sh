#!/bin/bash
# Block until a round-2 milestone is reached (or the training needs attention), then print status.
#   wait_milestone.sh step N        -> until any run has step-N.safetensors (or later)
#   wait_milestone.sh train_done    -> until ROUND2_TRAIN_DONE
#   wait_milestone.sh eval_done     -> until eval_round2.log has EVAL DONE
#   wait_milestone.sh upload_done   -> until the upload reports done / stopped
# Also returns early if status_round2.log reports RELAUNCHED / HUNG / DONE / STOPPED.
ROOT=/srv/wm_teacher
OUT=$ROOT/models/train/round2_5s_lora_r32
S=$ROOT/dataset/download_logs/status_round2.log
L=/dockerdata/wm_logs
start_lines=$(wc -l < $S)
reached() {
  case $1 in
    step) ls $OUT/run*/step-*.safetensors 2>/dev/null | sed 's/.*step-\([0-9]*\).*/\1/' |
            awk -v n=$2 '$1 >= n {f=1} END {exit !f}' ;;
    train_done) [ -f $L/ROUND2_TRAIN_DONE ] ;;
    eval_done) grep -q "EVAL DONE" $ROOT/dataset/download_logs/eval_round2.log 2>/dev/null ;;
    upload_done) tail -1 $S | grep -qE "UPLOAD (DONE|STOPPED)" ;;
  esac
}
while ! reached "$@"; do
  if tail -n +$((start_lines + 1)) $S | grep -qE "RELAUNCHED|HUNG|UPLOAD STOPPED"; then
    echo "ATTENTION:"; tail -n +$((start_lines + 1)) $S | grep -E "RELAUNCHED|HUNG|UPLOAD STOPPED" | tail -3
    break
  fi
  sleep 120
done
echo "== milestone '$*' $(date '+%F %T')"
tail -1 $S | cut -c1-220
ls -1 $OUT/run*/step-*.safetensors 2>/dev/null | tail -3
