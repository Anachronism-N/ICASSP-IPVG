#!/bin/bash
# Control the A100 caption stack on this node (vLLM servers + caption_5s.py).
#   caption_ctl.sh stop_server PORT     kill the vLLM server on PORT and its engine/worker procs
#   caption_ctl.sh stop_caption         kill caption_5s.py
#   caption_ctl.sh start_caption PORTS CONCURRENCY
#   caption_ctl.sh stop_all
ROOT=/srv/wm_teacher
L=/dockerdata/wm_logs
self=$$

pids_matching() {  # pids whose cmdline contains all given words, excluding this script's tree
  ps -eo pid,ppid,args | awk -v self=$self -v a="$1" -v b="${2:-}" \
    '$1 != self && $2 != self && index($0, a) && index($0, b) && !index($0, "caption_ctl") {print $1}'
}

stop_server() {  # API server by port; EngineCore + TP workers by the pids its log recorded
  local port=$1 log=$L/vllm_qwen35_tp4_$(( $1 - 8001 )).log
  pids_matching "vllm serve" "--port $port " | xargs -r kill -9
  grep -aoE "(EngineCore|Worker_TP[0-9]+) pid=[0-9]+" $log 2>/dev/null | sort -u | cut -d= -f2 |
    while read p; do grep -qai vllm /proc/$p/cmdline 2>/dev/null && kill -9 $p; done
  echo "stopped server on port $port"
}

case $1 in
  stop_server) stop_server $2 ;;
  stop_caption) pids_matching "python" "caption_5s.py" | xargs -r kill; echo "caption stopped" ;;
  start_caption)
    cp $ROOT/scripts/caption_5s.py /dockerdata/wm_tmp/caption_5s.py
    cd /dockerdata/wm_tmp
    [ -f $L/caption_5s.log ] && mv $L/caption_5s.log $L/caption_5s_$(date +%m%d%H%M%S).log
    setsid nohup /opt/conda/envs/torch-base/bin/python /dockerdata/wm_tmp/caption_5s.py \
      --ports $2 --concurrency ${3:-128} > $L/caption_5s.log 2>&1 < /dev/null &
    echo "caption started (ports $2)" ;;
  stop_all)
    pids_matching "python" "caption_5s.py" | xargs -r kill
    for p in 8001 8002; do stop_server $p; done ;;
esac
