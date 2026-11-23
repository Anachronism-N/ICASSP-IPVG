#!/bin/bash
# Round-1 8xH20 full LoRA training on the 59,682-sample mixture
# Mixture: training_metadata/round1_train.csv (built by scripts/build_round1_mixture.py)
# Base:    Wan2.2-TI2V-5B, LoRA rank 32, 1280x704, 49 frames
set -euo pipefail

PROJECT_ROOT="/srv/wm_teacher"
DIFFSYNTH_DIR="${PROJECT_ROOT}/DiffSynth-Studio"
cd "$DIFFSYNTH_DIR"

# Training env (torch 2.6.0+cu124 / accelerate 1.14.0)
export PATH="/srv/PRIVATE_STORAGE"
# Shadow the editable-installed diffsynth (OmniDance repo) with OUR checkout
export PYTHONPATH="${DIFFSYNTH_DIR}"

TRAIN_CSV="${PROJECT_ROOT}/training_metadata/round1_train.csv"
ACCELERATE_CONFIG="${PROJECT_ROOT}/training_metadata/accelerate_config_8gpu.yaml"
MODEL_FILES="diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"
OUTPUT_DIR="${PROJECT_ROOT}/models/train/round1_lora_r32"

echo "============================================"
echo "Round-1 Training: 8x H20, 59,682 samples"
echo "Output: $OUTPUT_DIR"
echo "Start:  $(date)"
echo "============================================"

accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    examples/wanvideo/model_training/train.py \
    --dataset_base_path "$PROJECT_ROOT" \
    --dataset_metadata_path "$TRAIN_CSV" \
    --data_file_keys "video" \
    --height 704 --width 1280 --num_frames 49 \
    --dataset_repeat 2 \
    --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:${MODEL_FILES}" \
    --learning_rate 1e-4 \
    --num_epochs 1 \
    --resume_from_checkpoint "${OUTPUT_DIR}/step-1000.safetensors" \
    --remove_prefix_in_ckpt "pipe.dit." \
    --output_path "$OUTPUT_DIR" \
    --lora_base_model "dit" \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --lora_rank 32 \
    --extra_inputs "input_image" \
    --use_gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --task "sft" \
    --save_steps 1000 \
    --find_unused_parameters \
    --enable_tensorboard_log

echo "============================================"
echo "Training complete! Check: $OUTPUT_DIR"
echo "End:    $(date)"
echo "============================================"
