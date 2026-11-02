#!/bin/bash
# Round-2: Wan2.2-TI2V-5B LoRA r32 on 5s windows (121 frames, 1280x704, Qwen3.5 dense captions),
# 2 nodes x 8 A100-40GB, initialised from the round-1 LoRA (49 frames) -> 2s->5s curriculum.
# Measured on one A100-40GB: peak 37.6GB, ~20 s/step (bs=1, gradient checkpointing).
#
# Usage (run on EACH node, same args):  bash train_round2_16gpu.sh MACHINE_RANK [TRAIN_CSV] [OUT_DIR] [extra train.py args]
#   rank 0 = launcher PRIVATE_NODE_ADDRESS, rank 1 = worker PRIVATE_NODE_ADDRESS
set -uo pipefail
RANK=$1
ROOT=/srv/wm_teacher
TRAIN_CSV=${2:-$ROOT/training_metadata/round2/round2_train.csv}
OUT=${3:-$ROOT/models/train/round2_5s_lora_r32}
shift $(( $# < 3 ? $# : 3 ))
INIT=${INIT:-$ROOT/models/train/round1_lora_r32/step-14921.safetensors}
MASTER=PRIVATE_NODE_ADDRESS
cd $ROOT/DiffSynth-Studio

# local conda env + local weights (shared-fs env/weights are far too slow through fuse)
export PATH=/opt/conda/envs/torch-base/bin:$PATH PYTHONPATH=$ROOT/DiffSynth-Studio
export DIFFSYNTH_MODEL_BASE_PATH=/dockerdata/models DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# no InfiniBand on these nodes: plain TCP over eth1 (LoRA grads are only ~160MB/step)
export NCCL_SOCKET_IFNAME=eth1 GLOO_SOCKET_IFNAME=eth1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN

# DiffSynth restarts its step counter at 0 on resume, so every (re)launch writes to its own runN/
# dir and initialises from the newest checkpoint of any earlier run (round-1 LoRA for run1).
# RUN_ID must be identical on both nodes -> passed in by the launcher (launch_round2.sh).
RUN_ID=${RUN_ID:?set RUN_ID}
LAST=$(ls -1t $OUT/run*/step-*.safetensors 2>/dev/null | head -1)
RESUME=${LAST:-$INIT}
OUT=$OUT/run$RUN_ID
echo "[round2] rank=$RANK csv=$TRAIN_CSV out=$OUT init=$RESUME start=$(date)"

# GPU_LIST lets a node run with fewer GPUs (worker GPU7 throws Xid 31 MMU faults -> excluded).
# torchrun's c10d rendezvous accepts a different nproc per node; accelerate launch does not.
export CUDA_VISIBLE_DEVICES=${GPU_LIST:-0,1,2,3,4,5,6,7} ACCELERATE_MIXED_PRECISION=bf16
NPROC=$(echo $CUDA_VISIBLE_DEVICES | tr , '\n' | wc -l)
echo "[round2] gpus=$CUDA_VISIBLE_DEVICES nproc=$NPROC"

M="diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"
# --local_addr: otherwise rank 0 advertises its FQDN as MASTER_ADDR, which the worker can't resolve
LOCAL_ADDR=$([ "$RANK" = 0 ] && echo $MASTER || echo PRIVATE_NODE_ADDRESS)
torchrun --nnodes 2 --nproc_per_node $NPROC --max_restarts 0 --local_addr $LOCAL_ADDR \
  --rdzv_backend c10d --rdzv_endpoint $MASTER:29511 --rdzv_id round2_run$RUN_ID \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path $ROOT --dataset_metadata_path $TRAIN_CSV --data_file_keys video \
  --height 704 --width 1280 --num_frames 121 --dataset_repeat 1 --dataset_num_workers 4 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:$M" \
  --learning_rate 1e-4 --num_epochs 1 \
  --resume_from_checkpoint "$RESUME" \
  --remove_prefix_in_ckpt pipe.dit. --output_path $OUT \
  --lora_base_model dit --lora_target_modules q,k,v,o,ffn.0,ffn.2 --lora_rank 32 \
  --extra_inputs input_image --use_gradient_checkpointing \
  --task sft --save_steps 500 --find_unused_parameters --enable_tensorboard_log "$@"
echo "[round2] rank=$RANK exit=$? end=$(date)"
