#!/bin/bash
#SBATCH --job-name=stage_platform_overlays
#SBATCH --output=logs/out/%j_stage_platform_overlays.out
#SBATCH --error=logs/err/%j_stage_platform_overlays.err
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=prod

set -euo pipefail

mkdir -p logs/out logs/err

OVERWRITE="${1:---overwrite}"
TITLE_MODE="${2:-none}"
OCCLUSION_FILTER_RADIUS_PX="${3:-0}"
OCCLUSION_MIN_DEPTH_GAP_M="${4:-1.0}"
OCCLUSION_MIN_DEPTH_GAP_RATIO="${5:-0.05}"

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1

CMD=(
  uv run python -m ihd.qc_review.stage_rendered_overlays_to_disk
  --results-root analysis/lidar_labeling
  --data-root /disk
  --preprocess-suffix platform_sphere_r2p5
  --out-root analysis/overlay_checks/platform_sphere_r2p5
  --title-mode "${TITLE_MODE}"
  --manifest-out analysis/qc_review/staged_platform_sphere_overlays_manifest.csv
  --occlusion-filter-radius-px "${OCCLUSION_FILTER_RADIUS_PX}"
  --occlusion-min-depth-gap-m "${OCCLUSION_MIN_DEPTH_GAP_M}"
  --occlusion-min-depth-gap-ratio "${OCCLUSION_MIN_DEPTH_GAP_RATIO}"
)

if [ "${OVERWRITE}" = "--overwrite" ]; then
  CMD+=(--overwrite)
fi

srun "${CMD[@]}"
