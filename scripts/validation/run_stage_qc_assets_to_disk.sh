#!/bin/bash
#SBATCH --job-name=ih_stage_qc_assets
#SBATCH --output=logs/out/%j_ih_stage_qc_assets.out
#SBATCH --error=logs/err/%j_ih_stage_qc_assets.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=prod

set -euo pipefail

mkdir -p logs/out logs/err

if [ "$#" -lt 0 ] || [ "$#" -gt 3 ]; then
  echo "Usage: sbatch $0 [<results_root>] [--overwrite] [<manifest_out>]" >&2
  exit 1
fi

RESULTS_ROOT="analysis/lidar_labeling"
OVERWRITE=""
MANIFEST_OUT="analysis/qc_review/staged_to_disk_manifest.csv"

if [ "$#" -ge 1 ]; then
  RESULTS_ROOT="$1"
fi
if [ "$#" -ge 2 ]; then
  OVERWRITE="$2"
fi
if [ "$#" -ge 3 ]; then
  MANIFEST_OUT="$3"
fi

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1

CMD=(
  uv run python -m ihd.qc_review.stage_to_disk
  --results-root "${RESULTS_ROOT}"
  --data-root /disk
  --manifest-out "${MANIFEST_OUT}"
)

if [ "${OVERWRITE}" = "--overwrite" ]; then
  CMD+=(--overwrite)
fi

echo "Staging QC assets into /disk scene folders"
echo "Results root: ${RESULTS_ROOT}"
echo "Manifest out: ${MANIFEST_OUT}"
echo "Overwrite: ${OVERWRITE:-false}"

srun "${CMD[@]}"
