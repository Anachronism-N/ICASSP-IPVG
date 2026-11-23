#!/bin/bash
# ==============================================================================
# Wan2.2 Smoke Test Training Script
# Trains on a small subset to verify the training pipeline works correctly.
#
# Data: 5,000 HOIGen-1M + 476 Seamless Interaction = 5,476 samples
# Model: Wan2.2-TI2V-5B (LoRA, rank=16 for faster smoke test)
#
# Usage:
#   bash scripts/smoke_test_wan22.sh
# ==============================================================================

set -euo pipefail

PROJECT_ROOT="/srv/wm_teacher"
DIFFSYNTH_DIR="${PROJECT_ROOT}/DiffSynth-Studio"
cd "$DIFFSYNTH_DIR"

# ─── Smoke Test Parameters ───────────────────────────────────────────────────
HEIGHT="704"
WIDTH="1280"
NUM_FRAMES="49"
DATASET_REPEAT="10"          # Fewer repeats for smoke test
LEARNING_RATE="1e-4"
NUM_EPOCHS="1"               # Single epoch for validation
LORA_RANK="16"               # Smaller rank for faster testing
SAVE_STEPS="500"             # Save checkpoints more frequently
OUTPUT_DIR="${PROJECT_ROOT}/models/train/smoke_test_lora"

MODEL_CACHE="${PROJECT_ROOT}/models/Wan2.2-TI2V-5B"

echo "============================================"
echo "Wan2.2 Smoke Test Training"
echo "============================================"

# ─── Step 1: Prepare smoke test metadata ─────────────────────────────────────
echo ""
echo "Step 1: Preparing smoke test metadata CSV..."
python3 "${PROJECT_ROOT}/scripts/prepare_smoke_test.py"

SMOKE_CSV="${PROJECT_ROOT}/training_metadata/smoke_combined_metadata.csv"
if [ ! -f "$SMOKE_CSV" ]; then
    echo "ERROR: Failed to generate smoke test metadata."
    exit 1
fi

NUM_SAMPLES=$(wc -l < "$SMOKE_CSV")
NUM_SAMPLES=$((NUM_SAMPLES - 1))
echo "Smoke test samples: $NUM_SAMPLES"

# ─── Step 2: Check model ─────────────────────────────────────────────────────
echo ""
echo "Step 2: Checking model availability..."
if [ ! -d "$MODEL_CACHE" ]; then
    echo "Model not found. Downloading..."
    mkdir -p "$MODEL_CACHE"
    if command -v modelscope &> /dev/null; then
        modelscope download --model "Wan-AI/Wan2.2-TI2V-5B" --local_dir "$MODEL_CACHE"
    else
        echo "ERROR: modelscope not installed. Run: pip install modelscope"
        exit 1
    fi
fi
echo "Model: $MODEL_CACHE"

# ─── Step 3: Run smoke test training ─────────────────────────────────────────
echo ""
echo "Step 3: Starting smoke test training..."
echo "  Resolution: ${WIDTH}x${HEIGHT}"
echo "  Frames:     ${NUM_FRAMES}"
echo "  LoRA rank:  ${LORA_RANK}"
echo "  LR:         ${LEARNING_RATE}"
echo "  Epochs:     ${NUM_EPOCHS}"
echo "  Repeat:     ${DATASET_REPEAT}x"
echo "  Samples:    ${NUM_SAMPLES}"
echo "============================================"

# Use PROJECT_ROOT as base path since metadata has relative paths
accelerate launch \
    examples/wanvideo/model_training/train.py \
    --dataset_base_path "$PROJECT_ROOT" \
    --dataset_metadata_path "$SMOKE_CSV" \
    --data_file_keys "video,prompt" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --num_frames "$NUM_FRAMES" \
    --dataset_repeat "$DATASET_REPEAT" \
    --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
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
