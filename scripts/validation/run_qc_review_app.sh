#!/bin/bash
#SBATCH --job-name=ih_qc_review
#SBATCH --output=logs/out/%j_ih_qc_review.out
#SBATCH --error=logs/err/%j_ih_qc_review.err
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --partition=prod

set -euo pipefail

mkdir -p logs/out logs/err

if [ "$#" -lt 1 ] || [ "$#" -gt 5 ]; then
  echo "Usage: sbatch $0 <reviewer_id> [<results_root>] [<port>] [<scene_list_csv>] [<review_mode>]" >&2
  exit 1
fi

REVIEWER_ID="$1"
RESULTS_ROOT="${2:-analysis/lidar_labeling}"
PORT="${3:-8765}"
SCENE_LIST_CSV="${4:-}"
REVIEW_MODE="${5:-label_quality}"
REPO_ROOT="${SLURM_SUBMIT_DIR}"

cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1

echo "IH-Depth QC review starting on host $(hostname -s) port ${PORT}"
echo "Reviewer: ${REVIEWER_ID}"
echo "Results root: ${RESULTS_ROOT}"
echo "Scene list CSV: ${SCENE_LIST_CSV:-<all discovered scenes>}"
echo "Review mode: ${REVIEW_MODE}"
echo "Session outputs: ${REPO_ROOT}/analysis/qc_review/sessions/${REVIEWER_ID}"
echo "Local forwarding command:"
echo "ssh -N -L ${PORT}:\$(ssh YOUR_WORKSTATION_HOST \"squeue -n ih_qc_review -h -o %N | head -n 1\"):${PORT} YOUR_WORKSTATION_HOST"

ARGS=(
  --reviewer-id "${REVIEWER_ID}"
  --results-root "${RESULTS_ROOT}"
  --data-root /disk
  --host 0.0.0.0
  --port "${PORT}"
  --review-mode "${REVIEW_MODE}"
)

if [ -n "${SCENE_LIST_CSV}" ]; then
  ARGS+=(--scene-list-csv "${SCENE_LIST_CSV}")
fi

srun uv run python -m ihd.qc_review.app "${ARGS[@]}"
