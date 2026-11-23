#!/bin/bash
# Install vLLM 0.31.0 into /dockerdata/venv_vllm on an A100 node.
# llguidance>=1.7 has no x86_64 wheel and its sdist needs a Rust toolchain download, so install
# vllm's deps without it (llguidance is only used for guided decoding, which captioning doesn't use).
set -e
export http_proxy=${http_proxy:-http://star-proxy.oa.com:3128} https_proxy=${https_proxy:-http://star-proxy.oa.com:3128}
V=/dockerdata/venv_vllm
L=/dockerdata/wm_logs
ps -eo pid,args | awk '$2 ~ /venv_vllm\/bin\/python/ && /pip install/ {print $1}' | xargs -r kill
curl -s -m 30 https://pypi.org/pypi/vllm/0.31.0/json | /opt/conda/envs/torch-base/bin/python -c "
import json, sys
d = json.load(sys.stdin)
reqs = [x.split(';')[0].strip() for x in d['info']['requires_dist']
        if 'extra ==' not in x and not x.startswith('llguidance')]
open('$L/vllm_reqs.txt', 'w').write('\n'.join(reqs) + '\n')"
$V/bin/pip install --prefer-binary -i https://pypi.org/simple -r $L/vllm_reqs.txt
$V/bin/pip install --no-deps -i https://pypi.org/simple vllm==0.31.0
echo VLLM_INSTALL_DONE
