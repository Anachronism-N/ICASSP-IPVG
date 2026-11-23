#!/bin/bash
# eval_candidate.sh NAME LORA GEN_NODE SCORE_GPUS(comma, node1)  -- dev gen (188) on GEN_NODE, scoring on node1, paired summary
set -uo pipefail
NAME=$1; LORA=$2; GN=$3; SG=$4
R=/srv/wm_teacher; E=/srv/wm_teacher/eval/teacher_progress_campaign_20261104; T=/srv/wm_teacher/training_metadata/train_opt_20261123; DEV=/srv/wm_teacher/eval/teacher_T1_paired_dev_20261122; PY=/dockerdata/venv_qwen32b_20261103/bin/python3.12; VPY=/dockerdata/venv_vbench_20261121/bin/python
[ -f $LORA ] || { echo "no LoRA $LORA"; exit 2; }
cd $R && [ -f $DEV/jobs_$NAME.jsonl ] || CUDA_VISIBLE_DEVICES= python3 scripts/build_teacher_paired_dev_jobs.py --root $DEV --model $NAME=$LORA --seeds 0,1 --append || exit 3
mk(){ python3 - "$@" <<'EOF'
import json,sys
name,phase,cmd,gpus,cap,purpose,out=sys.argv[1],sys.argv[2],json.loads(sys.argv[3]),json.loads(sys.argv[4]),int(sys.argv[5]),sys.argv[6],sys.argv[7]
env={"PYTHONPATH":"/dockerdata/wm_a800_fa2_runtime_20261103/site","OMP_NUM_THREADS":"4","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"}
if 'score' in name: env={"OMP_NUM_THREADS":"4","WM_EVAL_HOME":"/dockerdata/wm_eval_20261121","VBENCH_CACHE_DIR":"/dockerdata/wm_eval_20261121/vbench_cache","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"}
json.dump(dict(name=name,phase=phase,reserve_in_ledger=True,purpose=purpose,command=cmd,env=env,exclusive_process=False,gpu_indices=gpus,cap_seconds=cap,
  sources=["scripts/teacher_dev_generate.py","scripts/teacher_paired_dev_summary.py","scripts/score_e1.py","scripts/teacher_per_gpu_spawn.py","scripts/teacher_guarded_launch.py","scripts/teacher_gpu_exclusive.py"],success_files=[]),open(out,'w'),indent=1)
EOF
}
runnode(){ n=$1; name=$2; h=$(awk -v k=$n 'NR==k{print $1}' /etc/taiji/hostfile); P=$(ssh -o BatchMode=yes $h "cd $R && mkdir -p $E/${name}_logs && CUDA_VISIBLE_DEVICES= $PY scripts/teacher_guarded_launch.py prepare --spec $T/specs/${name}.json --stage-dir $E/${name}" 2>&1 | tail -1); SHA=$(echo "$P" | python3 -c "import json,sys;print(json.loads(sys.stdin.read())['plan']['sha256'])") || { echo "${name} prepare failed: $P"; return 1; }; echo "${name} start $(date)"; ssh -o BatchMode=yes $h "cd $R && exec $PY $R/scripts/teacher_guarded_launch.py run --plan $E/${name}/plan.json --sha $SHA --gpu-authorized > $E/${name}/parent.log 2>&1"; echo "${name} end $(date) $(cat $E/${name}/parent.log)"; }
mk ${NAME}gen training_optimization "[\"$PY\",\"$R/scripts/teacher_dev_generate.py\",\"spawn\",\"--jobs\",\"$DEV/jobs_$NAME.jsonl\",\"--log-dir\",\"$E/${NAME}gen_logs\"]" "[0,1,2,3,4,5,6,7]" 9000 "$NAME dev videos 94x2 seeds, same sampler/inputs as base" $T/specs/${NAME}gen.json
runnode $GN ${NAME}gen
python3 -c "import json,sys;t=json.load(open('$E/${NAME}gen/run01/terminal.json'));sys.exit(0 if t['success'] else 3)" || { echo "gen failed"; exit 4; }
( THREADS=24 CUDA_VISIBLE_DEVICES= WM_EVAL_HOME=/dockerdata/wm_eval_20261121 $VPY scripts/teacher_paired_dev_summary.py score gtref --root $DEV ${NAME}_s0 ${NAME}_s1 > $DEV/gtref_${NAME}.log 2>&1 ) &
mk ${NAME}score training_optimization "[\"$PY\",\"$R/scripts/teacher_per_gpu_spawn.py\",\"$E/${NAME}score_logs\",\"$VPY\",\"$R/scripts/teacher_paired_dev_summary.py\",\"score-shard\",\"vbench\",\"--root\",\"$DEV\",\"--configs\",\"${NAME}_s0,${NAME}_s1\",\"--shard\",\"{shard}\"]" "[$SG]" 3600 "$NAME paired dev VBench 8-dim scoring" $T/specs/${NAME}score.json
runnode 1 ${NAME}score
wait
cd $R && python3 -c "" && $VPY scripts/teacher_paired_dev_summary.py summary --root $DEV --candidate $NAME --out $DEV/summary_${NAME}_vs_base.json
echo EVAL_DONE_$NAME
