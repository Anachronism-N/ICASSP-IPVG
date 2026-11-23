#!/bin/bash
# Watchdog for the 5s reslice + caption pipeline. Every INTERVAL s:
#   - restarts any unhealthy vLLM server (one per GPU, ports 8001-8008)
#   - restarts caption_5s.py if it died before finishing
#   - restarts reslice_5s.py if it died before printing [done] (it resumes from manifests)
#   - appends a one-line status to /dockerdata/wm_logs/watchdog_5s.log
# Writes /dockerdata/wm_logs/PIPELINE_DONE when slicing and captioning have both finished.
ROOT=/srv/wm_teacher
L=/dockerdata/wm_logs
INTERVAL=${1:-300}
DATASETS=noxi,charades,egoexo4d,direct,vfhq,matrix,easycom,h2o,seamless,openvidhd,hoigen1m,celebvhq,mira
cd "$ROOT"

alive() { ps -eo args | awk -v pat="$1" 'index($0, pat) == 1 {n++} END {exit !(n > 0)}'; }

# Kill the API server AND its EngineCore for one GPU. nvidia-smi shows host PIDs (useless inside
# the container), so the EngineCore pid is taken from the server log, which deploy overwrites.
kill_gpu() {
  local g=$1 p=$((8001 + $1)) ep
  ep=$(grep -ao "EngineCore pid=[0-9]*" $L/vllm_qwen35_gpu$g.log 2>/dev/null | head -1 | cut -d= -f2)
  # match only the real server binary; the awk/xargs command lines contain the same words
  ps -eo pid,args | awk -v p="--port $p " '$2 ~ /python3\.13$/ && index($0, "vllm serve") && index($0, p) {print $1}' | xargs -r kill -9
  [ -n "$ep" ] && ps -o args= -p "$ep" 2>/dev/null | grep -q VLLM && kill -9 "$ep"
  for i in $(seq 1 12); do  # wait for the GPU memory to be released
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)" -lt 2000 ] && return 0
    sleep 5
  done
}

restart_gpu() {
  kill_gpu "$1"
  bash scripts/deploy_qwen35_vl.sh "$1" >> $L/watchdog_5s.log 2>&1
}

while true; do
  ts=$(date '+%F %T')
  # vLLM health
  bad=""
  for g in 0 1 2 3 4 5 6 7; do
    p=$((8001 + g))
    # /health stays 200 even when the engine is wedged (mm-cache AssertionError), so probe with
    # a real 1-token completion instead.
    code=$(curl -s --noproxy '*' -m 60 http://127.0.0.1:$p/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"qwen35-vl","messages":[{"role":"user","content":"OK"}],"max_tokens":1,"chat_template_kwargs":{"enable_thinking":false}}' \
      -o /dev/null -w '%{http_code}')
    if [ "$code" != "200" ]; then
      # give a freshly (re)started server 10 min before touching it again
      st=$L/vllm_restart_gpu$g.ts
      if [ ! -f $st ] || [ $(( $(date +%s) - $(cat $st) )) -gt 600 ]; then
        restart_gpu "$g"
        bash scripts/deploy_qwen35_vl.sh "$g" >> $L/watchdog_5s.log 2>&1
        date +%s > $st
        bad="$bad restart_gpu$g"
      else
        bad="$bad starting_gpu$g"
      fi
    fi
  done
  rs_done=$(grep -c '^\[done\]' $L/reslice_5s.log 2>/dev/null)
  cap_done=$(grep -c '^\[done\]' $L/caption_5s.log 2>/dev/null)
  if [ "${rs_done:-0}" = "0" ] && ! alive "python3 scripts/reslice_5s.py"; then
    mv $L/reslice_5s.log $L/reslice_5s_$(date +%m%d%H%M).log 2>/dev/null
    setsid nice -n 10 python3 scripts/reslice_5s.py --all-windows --datasets $DATASETS \
      --out-root dataset/wan22_5s --workers 96 --chunk 12 > $L/reslice_5s.log 2>&1 < /dev/null &
    bad="$bad restart_reslice"
  fi
  if [ "${cap_done:-0}" = "0" ] && ! alive "python3 scripts/caption_5s.py"; then
    mv $L/caption_5s.log $L/caption_5s_$(date +%m%d%H%M).log 2>/dev/null
    setsid python3 scripts/caption_5s.py --ports 8001-8008 --concurrency 768 \
      > $L/caption_5s.log 2>&1 < /dev/null &
    bad="$bad restart_caption"
  fi
  man=$(cat dataset/wan22_5s/manifest_5s_landscape.jsonl dataset/wan22_5s/manifest_5s_portrait.jsonl 2>/dev/null | wc -l)
  cap=$(wc -l < dataset/wan22_5s/captions_5s.jsonl 2>/dev/null)
  capf=$(wc -l < dataset/wan22_5s/captions_5s_failed.jsonl 2>/dev/null)
  rsl=$(grep '^\[progress\]' $L/reslice_5s.log | tail -1 | cut -c12-120)
  cpl=$(grep '^\[progress\]' $L/caption_5s.log | tail -1 | cut -c12-160)
  echo "$ts sliced=$man captioned=$cap cap_failed=$capf | reslice: $rsl | caption: $cpl |$bad" >> $L/watchdog_5s.log
  if [ "${rs_done:-0}" != "0" ] && [ "${cap_done:-0}" != "0" ]; then
    echo "$ts sliced=$man captioned=$cap failed=$capf" > $L/PIPELINE_DONE
    echo "$ts PIPELINE DONE" >> $L/watchdog_5s.log
    exit 0
  fi
  sleep "$INTERVAL"
done
