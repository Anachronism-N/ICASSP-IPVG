#!/usr/bin/env bash
#SBATCH --job-name=ipvg-multinode-profile
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -euo pipefail

# Profiling skeleton only: no universal upstream training CLI is assumed.
# Validate NCCL/network, shared paths and the actual sharded trainer before use.
IPVG_ROOT="${IPVG_ROOT:-${SLURM_SUBMIT_DIR:?Submit this job with sbatch}}"
: "${IPVG_UPSTREAM_LAUNCHER:?Set an executable deployment-owned upstream launch script}"
test -x "$IPVG_UPSTREAM_LAUNCHER"
cd "$IPVG_ROOT"
export MASTER_ADDR
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${IPVG_MASTER_PORT:-29500}"
export IPVG_NNODES="$SLURM_NNODES"
export IPVG_GPUS_PER_NODE=8
# The launcher reads SLURM_NODEID for node rank and launches 8 local workers once.
# It must use verified upstream args/config and write its own resource/checkpoint log.
# Do not wrap an already-spawning launcher in another 8-task-per-node srun.
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 "$IPVG_UPSTREAM_LAUNCHER"
