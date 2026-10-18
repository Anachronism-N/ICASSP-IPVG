#!/bin/bash
# Run one foreground, supervised generator per GPU on this node.
# Invoke separately on each node; NODE_RANK=0..3 NNODES=4 yields shards 0..31.
# Usage: NODE_RANK=0 NNODES=4 bash run_gen.sh JOBS.jsonl TAG [--models a,b]
set -euo pipefail
ROOT=/srv/wm_teacher
: "${NODE_RANK:?Set NODE_RANK explicitly for this node}"
: "${NNODES:?Set NNODES explicitly for the cluster}"
if [[ ! $NODE_RANK =~ ^[0-9]+$ || ! $NNODES =~ ^[1-9][0-9]*$ ]]; then
  printf 'Invalid NODE_RANK or NNODES\n' >&2
  exit 2
fi
NODE_RANK=$((10#$NODE_RANK))
NNODES=$((10#$NNODES))
if (( NODE_RANK >= NNODES || $# < 2 )); then
  printf 'Usage: NODE_RANK=<rank> NNODES=<count> bash run_gen.sh JOBS TAG [--models a,b]\n' >&2
  exit 2
fi
JOBS=$1; TAG=$2; shift 2
if [[ ! $TAG =~ ^[a-zA-Z0-9_-]+$ ]]; then
  printf 'TAG must contain only letters, digits, underscores or hyphens\n' >&2
  exit 2
fi
[[ -s $JOBS && -d $ROOT/DiffSynth-Studio ]]
PYTHON=/opt/conda/envs/torch-base/bin/python
export PATH=/opt/conda/envs/torch-base/bin:$PATH
export PYTHONPATH=$ROOT/DiffSynth-Studio
export DIFFSYNTH_MODEL_BASE_PATH=/dockerdata/models
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export WM_CHECKPOINT_CACHE_DIR=/dockerdata/models/wm_checkpoints
cd "$ROOT/DiffSynth-Studio"
"$PYTHON" -c 'import av, imageio_ffmpeg, torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 8'
[[ -d $DIFFSYNTH_MODEL_BASE_PATH/Wan-AI && -d $DIFFSYNTH_MODEL_BASE_PATH/DiffSynth-Studio ]]
# Do not let 32 generators fall back to mmap of shared checkpoint files.
"$PYTHON" - "$JOBS" <<'PY'
import hashlib, os, sys, json
checkpoints = {json.loads(line)["ckpt"] for line in open(sys.argv[1])}
for source in checkpoints - {"NONE"}:
    key = hashlib.sha256(os.path.abspath(source).encode()).hexdigest() + ".safetensors"
    local = os.path.join(os.environ["WM_CHECKPOINT_CACHE_DIR"], key)
    assert os.path.isfile(local) and os.path.getsize(local) > 0, f"Missing local checkpoint: {local}"
    assert os.path.getsize(local) == os.path.getsize(source), f"Incomplete local checkpoint: {local}"
PY
L=$ROOT/dataset/download_logs
mkdir -p "$L" /dockerdata/wm_tmp
# Unique local copy: never overwrite a script being read by a running process.
LOCAL_DIR=$(mktemp -d "/dockerdata/wm_tmp/gen_${TAG}_r${NODE_RANK}_XXXXXX")
cp "$ROOT/scripts/gen_jobs.py" "$LOCAL_DIR/gen_jobs.py"
NSHARDS=$((NNODES * 8))
pids=()
stop_workers() {
  trap - EXIT INT TERM
  if (( ${#pids[@]} )); then
    kill -TERM "${pids[@]}" 2>/dev/null || true
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  fi
}
trap stop_workers EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for g in 0 1 2 3 4 5 6 7; do
  s=$((NODE_RANK * 8 + g))
  log="$L/gen_${TAG}_s${s}_${LOCAL_DIR##*/}.log"
  CUDA_VISIBLE_DEVICES=$g "$PYTHON" "$LOCAL_DIR/gen_jobs.py" "$JOBS" "$s" "$NSHARDS" "$@" > "$log" 2>&1 &
  pids+=("$!")
  printf '[started] rank=%s gpu=%s shard=%s/%s pid=%s log=%s\n' "$NODE_RANK" "$g" "$s" "$NSHARDS" "$!" "$log"
done
status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[i]}"; then status=1; fi
  unset 'pids[i]'
done
# Exit only means workers stopped; full manifest/score validation is separate.
exit "$status"
