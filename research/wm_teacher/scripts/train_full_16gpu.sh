#!/bin/bash
# FULL fine-tune of the Wan2.2-TI2V-5B DiT (121f 1280x704), 2 nodes (launcher 8 + worker 7 GPUs),
# DeepSpeed ZeRO-2 with optimizer state + fp32 master weights on CPU (DeepSpeedCPUAdam).
# Memtest (1x A100-40GB): 26 s/step, peak 40.4GB (too tight: NCCL OOM at the first ckpt save on 15 GPUs
#   with overlap_comm + 2e8 buckets -> now overlap_comm off, 5e7 buckets); with FULL_ACT_OFFLOAD=1 44 s/step, peak 36.1GB.
# First run initialises from base + PRESET_LORA fused; later runs resume from the newest full ckpt.
# Usage (each node): RUN_ID=N bash train_full_16gpu.sh MACHINE_RANK TRAIN_CSV OUT_DIR [extra train.py args]
set -uo pipefail
RANK=$1; TRAIN_CSV=$2; OUT=$3; shift 3
ROOT=/srv/wm_teacher
PRESET_LORA=${PRESET_LORA:-$ROOT/models/train/round4_aes_lora_r32/run1/step-1644.safetensors}
MASTER=PRIVATE_NODE_ADDRESS
cd $ROOT/DiffSynth-Studio
export PATH=/opt/conda/envs/torch-base/bin:$PATH PYTHONPATH=$ROOT/DiffSynth-Studio
export DIFFSYNTH_MODEL_BASE_PATH=/dockerdata/models DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export NCCL_SOCKET_IFNAME=eth1 GLOO_SOCKET_IFNAME=eth1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN
# accelerate picks DeepSpeed up from the environment when launched via torchrun
export ACCELERATE_USE_DEEPSPEED=true ACCELERATE_DEEPSPEED_CONFIG_FILE=$ROOT/training_metadata/full/ds_zero2_offload.json
export ACCELERATE_MIXED_PRECISION=bf16
RUN_ID=${RUN_ID:?set RUN_ID}
LAST=$(ls -1t $OUT/run*/step-*.safetensors 2>/dev/null | head -1)
if [ -n "$LAST" ]; then INIT_ARGS=(--resume_from_checkpoint "$LAST")
elif [ "$PRESET_LORA" = NONE ]; then INIT_ARGS=()   # start from the Wan2.2-TI2V-5B base weights
else INIT_ARGS=(--preset_lora_path "$PRESET_LORA" --preset_lora_model dit); fi
ACT=(); [ "${FULL_ACT_OFFLOAD:-0}" = 1 ] && ACT=(--use_gradient_checkpointing_offload)
OUT=$OUT/run$RUN_ID
export CUDA_VISIBLE_DEVICES=${GPU_LIST:-0,1,2,3,4,5,6,7}
NPROC=$(echo $CUDA_VISIBLE_DEVICES | tr , '\n' | wc -l)
echo "[full] rank=$RANK csv=$TRAIN_CSV out=$OUT init=${INIT_ARGS[*]:-base} act_offload=${FULL_ACT_OFFLOAD:-0} gpus=$CUDA_VISIBLE_DEVICES start=$(date)"
M="diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"
LOCAL_ADDR=$([ "$RANK" = 0 ] && echo $MASTER || echo PRIVATE_NODE_ADDRESS)
torchrun --nnodes 2 --nproc_per_node $NPROC --max_restarts 0 --local_addr $LOCAL_ADDR \
  --rdzv_backend c10d --rdzv_endpoint $MASTER:29513 --rdzv_id full_${RDZV_TAG:-x}_run$RUN_ID \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path $ROOT --dataset_metadata_path $TRAIN_CSV --data_file_keys video \
  --height 704 --width 1280 --num_frames 121 --dataset_repeat 1 --dataset_num_workers 4 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:$M" --num_epochs 1 \
  "${INIT_ARGS[@]}" --remove_prefix_in_ckpt pipe.dit. --output_path $OUT \
  --trainable_models dit --customized_optimizer deepspeed.ops.adam.DeepSpeedCPUAdam \
  --extra_inputs input_image --use_gradient_checkpointing "${ACT[@]}" \
  --task sft --save_steps 250 --enable_tensorboard_log "$@"
echo "[full] rank=$RANK exit=$? end=$(date)"
