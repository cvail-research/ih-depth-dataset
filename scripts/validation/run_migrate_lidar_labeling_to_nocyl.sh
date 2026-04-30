#!/bin/bash
#SBATCH --job-name=migrate_nocyl
#SBATCH --output=logs/out/%j_migrate_nocyl.out
#SBATCH --error=logs/err/%j_migrate_nocyl.err
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

OVERWRITE="${1:-}"
EXTRA_ARGS=()
if [ "${OVERWRITE}" = "--overwrite" ]; then
  EXTRA_ARGS+=(--overwrite)
fi

srun uv run python -m ihd.datasets.migrate_lidar_labeling_to_nocyl \
  --results-root analysis/lidar_labeling \
  --workspace-root analysis/annotation_workspace_nocyl \
  --preprocess-suffix platform_sphere_r2p5 \
  --manifest-out analysis/qc_review/migrated_lidar_labeling_to_nocyl_manifest.csv \
  "${EXTRA_ARGS[@]}"
