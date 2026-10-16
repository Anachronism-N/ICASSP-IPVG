#!/usr/bin/env bash
#SBATCH --job-name=ipvg-campaign
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -euo pipefail

IPVG_ROOT="${IPVG_ROOT:-${SLURM_SUBMIT_DIR:?Submit this job with sbatch}}"
IPVG_PYTHON="${IPVG_PYTHON:-$IPVG_ROOT/.venv/bin/python}"
: "${IPVG_CAMPAIGN:?Set a finite real-backend campaign JSON path}"
cd "$IPVG_ROOT"
test -x "$IPVG_PYTHON"
test -f "$IPVG_CAMPAIGN"
# This runs one controller, not eight copies. A configured backend owns its GPU launch.
# Match each experiment's resources.gpus and timeout to its actual backend allocation.
srun --ntasks=1 "$IPVG_PYTHON" -m ipvg campaign run "$IPVG_CAMPAIGN" --root "$IPVG_ROOT"
