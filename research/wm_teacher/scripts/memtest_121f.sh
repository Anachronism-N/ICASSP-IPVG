#!/bin/bash
# Peak-memory probe for Wan2.2-TI2V-5B LoRA at 121 frames on one A100-40GB.
# Usage: bash scripts/memtest_121f.sh GPU TAG [extra train.py args...]
ROOT=/srv/wm_teacher
GPU=$1; TAG=$2; shift 2
cd $ROOT/DiffSynth-Studio
# local env: the shared-fs conda env imports far too slowly through fuse
export PATH=/opt/conda/envs/torch-base/bin:$PATH PYTHONPATH=$ROOT/DiffSynth-Studio
export DIFFSYNTH_MODEL_BASE_PATH=/dockerdata/models DIFFSYNTH_SKIP_DOWNLOAD=true
export CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M="diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"
( while sleep 5; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU; done ) > /dockerdata/wm_logs/memtest_$TAG.mem &
MON=$!
accelerate launch --config_file $ROOT/training_metadata/round2/accel_1gpu.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path $ROOT --dataset_metadata_path $ROOT/training_metadata/round2/memtest8.csv --data_file_keys video \
  --height 704 --width 1280 --num_frames 121 --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:$M" --learning_rate 1e-4 --num_epochs 1 \
  --remove_prefix_in_ckpt pipe.dit. --output_path /dockerdata/wm_tmp/memtest_$TAG \
  --lora_base_model dit --lora_target_modules q,k,v,o,ffn.0,ffn.2 --lora_rank 32 \
  --extra_inputs input_image --use_gradient_checkpointing --task sft --save_steps 100000 "$@"
echo "EXIT $?"
kill $MON
echo "PEAK_MiB $(sort -n /dockerdata/wm_logs/memtest_$TAG.mem | tail -1)"
