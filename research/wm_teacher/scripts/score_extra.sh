#!/bin/bash
# Score extra public benchmarks for the given models (run on the launcher).
#   Physics-IQ (original, CPU): trim each <model>-op-run_NN to exactly 5 s (120 frames @24fps) -> run_physics_iq.py
#   WorldModelBench (GPU $WMB_GPU): official evaluation.py with the vila-ewm-qwen2-1.5b judge
# Usage: bash score_extra.sh base,round3   -> eval/physiq/results/, eval/wmb/results/<model>.json
ROOT=/srv/wm_teacher
EV=$ROOT/eval; B=/dockerdata/wm_eval/bench; L=$ROOT/dataset/download_logs
FF=${FFMPEG:-$(/opt/conda/envs/torch-base/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')}
[ -x "$FF" ] || { printf 'ffmpeg unavailable: %s\n' "$FF" >&2; exit 1; }
WMB_GPU=${WMB_GPU:-0}
MODELS=${1//,/ }

# ---- Physics-IQ
dirs=()
for m in $MODELS; do
  for src in $EV/physiq/videos/$m-op-run_*; do
    [ -d "$src" ] || continue
    dst=$EV/physiq/videos_5s/$(basename $src); mkdir -p $dst
    for f in $src/*.mp4; do
      o=$dst/$(basename $f); [ -s "$o" ] && continue
      # our 1280x704 = 16:9 input with 8 px cropped top/bottom -> pad back (edge smear) so pixels align with the 16:9 GT
      $FF -nostdin -loglevel error -y -i "$f" -frames:v 120 -an \
        -vf "pad=1280:720:0:8,fillborders=top=8:bottom=8:mode=smear" -c:v libx264 -crf 12 -pix_fmt yuv420p "$o"
    done
    n=$(ls $dst | grep -c 'mp4$'); [ "$n" -eq 198 ] && dirs+=($dst) || echo "physiq: $dst has $n/198 videos, skipped"
  done
done
if [ ${#dirs[@]} -gt 0 ]; then
  (cd $B/physics-IQ-benchmark && /dockerdata/venv_piq/bin/python physiq/run_physics_iq.py \
     --input_folders "${dirs[@]}" --output_folder $EV/physiq/results \
     --descriptions_file $B/physics-IQ-benchmark/descriptions/descriptions_original.csv \
     --original_physics_iq --benchmark_base_folder $B/physics-IQ-benchmark-data --n_process 48) >> $L/score_physiq.log 2>&1
  echo "physiq scored: ${dirs[*]##*/} (log score_physiq.log)"
fi

# ---- WorldModelBench
if [ -x /dockerdata/conda_vila/bin/python ]; then
  for m in $MODELS; do
    [ -f $EV/wmb/results/$m.json ] && continue
    n=$(ls $EV/wmb/videos/$m 2>/dev/null | grep -c 'mp4$'); [ "$n" -ne 350 ] && { echo "wmb: $m has $n/350 videos, skipped"; continue; }
    mkdir -p $EV/wmb/results
    (cd $B/WorldModelBench && CUDA_VISIBLE_DEVICES=$WMB_GPU PYTHONPATH=$B/VILA /dockerdata/conda_vila/bin/python evaluation.py \
       --model_name $m --video_dir $EV/wmb/videos/$m --judge $B/judges/vila-ewm-qwen2-1.5b --save_name wmb_$m) > $L/score_wmb_$m.log 2>&1 &&
      cp $B/WorldModelBench/wmb_$m.json $EV/wmb/results/$m.json && echo "wmb scored: $m"
  done
else
  echo "wmb: VILA env missing, skipped"
fi
