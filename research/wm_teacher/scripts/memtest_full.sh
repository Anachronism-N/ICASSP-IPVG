#!/bin/bash
# Peak-memory probe: FULL fine-tune of Wan2.2-TI2V-5B DiT (121f 1280x704) on one A100-40GB,
# DeepSpeed ZeRO-2 + CPU Adam (optimizer offload). Usage: bash memtest_full.sh GPU TAG [extra train.py args]
ROOT=/srv/wm_teacher
GPU=$1; TAG=$2; shift 2
cd $ROOT/DiffSynth-Studio
export PATH=/opt/conda/envs/torch-base/bin:$PATH PYTHONPATH=$ROOT/DiffSynth-Studio
export DIFFSYNTH_MODEL_BASE_PATH=/dockerdata/models DIFFSYNTH_SKIP_DOWNLOAD=true
export CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
( while sleep 3; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU; done ) > /dockerdata/wm_logs/memtest_full_$TAG.mem &
MON=$!
M="diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"
accelerate launch --config_file $ROOT/training_metadata/full/accel_ds_1gpu.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path $ROOT --dataset_metadata_path $ROOT/training_metadata/round2/memtest8.csv --data_file_keys video \
  --height 704 --width 1280 --num_frames 121 --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:$M" --learning_rate 1e-5 --num_epochs 1 \
  --remove_prefix_in_ckpt pipe.dit. --output_path /dockerdata/wm_tmp/memtest_full_$TAG \
  --trainable_models dit --customized_optimizer deepspeed.ops.adam.DeepSpeedCPUAdam \
  --preset_lora_path $ROOT/models/train/round4_aes_lora_r32/run1/step-1644.safetensors --preset_lora_model dit \
  --extra_inputs input_image --use_gradient_checkpointing --task sft --save_steps 4 "$@"
echo "EXIT $?"
kill $MON
echo "PEAK_MiB $(sort -n /dockerdata/wm_logs/memtest_full_$TAG.mem | tail -1)"
