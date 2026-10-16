#!/usr/bin/env bash
#SBATCH --job-name=ipvg-inference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err
set -euo pipefail

IPVG_ROOT="${IPVG_ROOT:-${SLURM_SUBMIT_DIR:?Submit this job with sbatch}}"
IPVG_PYTHON="${IPVG_PYTHON:-$IPVG_ROOT/.venv/bin/python}"
: "${IPVG_CONFIG_LIST:?Set newline-separated experiment config paths, without blank lines}"
cd "$IPVG_ROOT"
test -x "$IPVG_PYTHON"
test -f "$IPVG_CONFIG_LIST"
mapfile -t IPVG_ARRAY_CONFIGS < "$IPVG_CONFIG_LIST"
IPVG_INDEX="${SLURM_ARRAY_TASK_ID:?An array task ID is required}"
if (( IPVG_INDEX < 0 || IPVG_INDEX >= ${#IPVG_ARRAY_CONFIGS[@]} )); then
  echo "Array index exceeds config-list length" >&2
  exit 2
fi
IPVG_CONFIG="${IPVG_ARRAY_CONFIGS[$IPVG_INDEX]}"
test -n "$IPVG_CONFIG"
test -f "$IPVG_CONFIG"
# Each config must have a unique run_id and a disjoint manifest shard.
"$IPVG_PYTHON" -m ipvg experiment plan "$IPVG_CONFIG" --root "$IPVG_ROOT"
srun --ntasks=1 "$IPVG_PYTHON" -m ipvg experiment run "$IPVG_CONFIG" --root "$IPVG_ROOT"
