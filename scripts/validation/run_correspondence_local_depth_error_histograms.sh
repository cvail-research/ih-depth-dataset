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
BINS="${3:-0,10,100,inf}"
DEPTH_LABEL_ROOT="${4:-analysis/depth_labels/platform_sphere_r2p5}"
SAMPLE_MODE="${5:-closest-range}"
OCCLUSION_FILTER_RADIUS_PX="${6:-0}"
OCCLUSION_MIN_DEPTH_GAP_M="${7:-1.0}"
OCCLUSION_MIN_DEPTH_GAP_RATIO="${8:-0.05}"

rm -rf "${OUT_DIR}"
mkdir -p "${DEPTH_LABEL_ROOT}"

srun uv run python -m ihd.datasets.summarize_correspondence_distance_errors \
  --out-dir "${OUT_DIR}" \
  --sample-radius-px "${SAMPLE_RADIUS_PX}" \
  --sample-mode "${SAMPLE_MODE}" \
  --bins "${BINS}" \
  --preprocess-suffix platform_sphere_r2p5 \
  --save-depth-labels \
  --depth-label-root "${DEPTH_LABEL_ROOT}" \
  --occlusion-filter-radius-px "${OCCLUSION_FILTER_RADIUS_PX}" \
  --occlusion-min-depth-gap-m "${OCCLUSION_MIN_DEPTH_GAP_M}" \
  --occlusion-min-depth-gap-ratio "${OCCLUSION_MIN_DEPTH_GAP_RATIO}"
