#!/bin/bash
# Physics-IQ Verified official scoring for one model (4 runs): official ffmpeg trim (-t 5 -r 24), official evaluator, official aggregation.
set -euo pipefail
MODEL=${1:?model tag e.g. wan22ti2v5b_base}; NPROC=${2:-8}
O=/srv/wm_teacher/eval/official_reeval_20261122/physiq_verified
D=/dockerdata/wm_eval_20261121/official/physics-IQ-benchmark
PY=/dockerdata/venv_physiq_20261122/bin/python
FF=/usr/local/bin/ffmpeg
for k in 01 02 03 04; do
  src=$O/videos_raw/$MODEL-bpp-run_$k; dst=$O/generated_videos_5s/$MODEL-bpp-run_$k
  n=$(ls $src/*.mp4 | wc -l); [ "$n" = 198 ] || { echo "run_$k has $n/198 videos"; exit 2; }
  mkdir -p $dst
  for v in $src/*.mp4; do o=$dst/$(basename $v); [ -s $o ] || $FF -loglevel error -y -i "$v" -t 5 -r 24 "$o"; done
done
cd $D
$PY physiq/run_physics_iq.py --input_folders $(for k in 01 02 03 04; do echo $O/generated_videos_5s/$MODEL-bpp-run_$k; done) \
  --output_folder $O/scores --descriptions_file $D/descriptions/best_practice/descriptions_base.csv \
  --benchmark_base_folder /srv/wm_teacher/eval/physiq_verified --n_process $NPROC
$PY physiq/aggregate_runs_from_csvs.py $O/scores/physics-IQ-benchmark-verified/results/$MODEL-bpp-run_0{1,2,3,4}.csv --score-type verified --model-name $MODEL --save-csv $O/scores/$MODEL.verified_aggregate.csv
$PY physiq/aggregate_runs_from_csvs.py $O/scores/physics-IQ-benchmark-verified/results/$MODEL-bpp-run_0{1,2,3,4}.csv --score-type original --model-name $MODEL --save-csv $O/scores/$MODEL.original_metric_aggregate.csv
echo SCORING_DONE
