#!/usr/bin/env bash
#SBATCH --job-name=ipvg-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -euo pipefail

# Submit from the project root after mkdir -p logs/slurm.
IPVG_ROOT="${IPVG_ROOT:-${SLURM_SUBMIT_DIR:?Submit this job with sbatch}}"
IPVG_PYTHON="${IPVG_PYTHON:-$IPVG_ROOT/.venv/bin/python}"
IPVG_CONFIG="${IPVG_CONFIG:-configs/experiments/synthetic_smoke.json}"
cd "$IPVG_ROOT"
test -x "$IPVG_PYTHON"
test -f "$IPVG_CONFIG"
"$IPVG_PYTHON" -m ipvg doctor --output "runs/doctor-slurm-${SLURM_JOB_ID}.json"
"$IPVG_PYTHON" -m ipvg experiment plan "$IPVG_CONFIG" --root "$IPVG_ROOT"
srun --ntasks=1 "$IPVG_PYTHON" -m ipvg experiment run "$IPVG_CONFIG" --root "$IPVG_ROOT"
