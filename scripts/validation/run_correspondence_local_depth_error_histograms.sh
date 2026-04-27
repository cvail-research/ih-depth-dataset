#!/bin/bash
#SBATCH --job-name=local_depth_errors
#SBATCH --output=logs/out/%j_local_depth_errors.out
#SBATCH --error=logs/err/%j_local_depth_errors.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=prod

set -euo pipefail

mkdir -p logs/out logs/err

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1

OUT_DIR="${1:-analysis/qc_review/correspondence_distance_errors}"
SAMPLE_RADIUS_PX="${2:-5}"
BINS="${3:-auto-tertiles}"

rm -rf "${OUT_DIR}"

srun uv run python -m ihd.datasets.summarize_correspondence_distance_errors \
  --out-dir "${OUT_DIR}" \
  --sample-radius-px "${SAMPLE_RADIUS_PX}" \
  --bins "${BINS}" \
  --preprocess-suffix platform_sphere_r2p5
