#!/bin/bash
# Rolling restart of selected vLLM servers (one at a time) so captioning never loses capacity.
# Usage: bash scripts/rolling_restart_vllm.sh "0 1 2 6 7"
ROOT=/srv/wm_teacher
L=/dockerdata/wm_logs
cd "$ROOT"
eval "$(sed -n '/^alive()/,/^}$/p; /^kill_gpu()/,/^}$/p; /^restart_gpu()/,/^}$/p' scripts/watchdog_5s.sh)"

probe() {
  curl -s --noproxy '*' -m 30 http://127.0.0.1:$1/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"qwen35-vl","messages":[{"role":"user","content":"OK"}],"max_tokens":1,"chat_template_kwargs":{"enable_thinking":false}}' \
    -o /dev/null -w '%{http_code}'
}

for g in $1; do
  p=$((8001 + g))
  echo "$(date '+%T') restarting gpu$g" >> $L/watchdog_5s.log
  restart_gpu "$g"
  date +%s > $L/vllm_restart_gpu$g.ts
  for i in $(seq 1 40); do
    [ "$(probe $p)" = "200" ] && break
    sleep 15
  done
  echo "$(date '+%T') gpu$g probe=$(probe $p)" >> $L/watchdog_5s.log
done
