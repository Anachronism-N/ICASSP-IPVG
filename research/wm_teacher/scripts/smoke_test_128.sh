#!/bin/bash
# Smoke training: 128 samples, LoRA rank 16, 200 steps
set -euo pipefail

PROJECT_ROOT="/srv/wm_teacher"
DIFFSYNTH_DIR="${PROJECT_ROOT}/DiffSynth-Studio"
cd "$DIFFSYNTH_DIR"

# Training env (torch 2.6.0+cu124 / accelerate 1.14.0, 8x H20 verified)
export PATH="/srv/PRIVATE_STORAGE"
# Shadow the editable-installed diffsynth (OmniDance repo) with OUR checkout
export PYTHONPATH="${DIFFSYNTH_DIR}"

# Config
HEIGHT="704"
WIDTH="1280"
NUM_FRAMES="49"
DATASET_REPEAT="3"
LEARNING_RATE="1e-4"
NUM_EPOCHS="1"
LORA_RANK="16"
SAVE_STEPS="100"
OUTPUT_DIR="${PROJECT_ROOT}/models/train/smoke_test_128"
MODEL_CACHE="${PROJECT_ROOT}/models/Wan2.2-TI2V-5B"
SMOKE_CSV="${PROJECT_ROOT}/training_metadata/smoke/smoke128.csv"
ACCELERATE_CONFIG="${PROJECT_ROOT}/training_metadata/accelerate_config.yaml"
MODEL_FILES="diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"

echo "============================================"
echo "Wan2.2 Smoke Test (128 samples, 200 steps)"
echo "============================================"
echo "Model:      $MODEL_CACHE"
echo "Metadata:   $SMOKE_CSV"
echo "Resolution: ${WIDTH}x${HEIGHT}"
echo "Frames:     $NUM_FRAMES"
echo "LoRA rank:  $LORA_RANK"
echo "Output:     $OUTPUT_DIR"
echo "============================================"

# Generate accelerate config if not exists
if [ ! -f "$ACCELERATE_CONFIG" ]; then
    accelerate config default --mixed_precision bf16 --num_processes 1 --num_machines 1 --machine_rank 0
fi

# Launch training
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    examples/wanvideo/model_training/train.py \
    --dataset_base_path "$PROJECT_ROOT" \
    --dataset_metadata_path "$SMOKE_CSV" \
    --data_file_keys "video" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --num_frames "$NUM_FRAMES" \
    --dataset_repeat "$DATASET_REPEAT" \
    --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:${MODEL_FILES}" \
    --learning_rate "$LEARNING_RATE" \
    --num_epochs "$NUM_EPOCHS" \
    --remove_prefix_in_ckpt "pipe.dit." \
    --output_path "$OUTPUT_DIR" \
    --lora_base_model "dit" \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --lora_rank "$LORA_RANK" \
    --extra_inputs "input_image" \
    --use_gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --task "sft" \
    --save_steps "$SAVE_STEPS" \
    --enable_tensorboard_log

echo ""
echo "============================================"
echo "Smoke test complete! Check: $OUTPUT_DIR"
echo "============================================"
