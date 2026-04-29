#!/bin/bash
#SBATCH --job-name=ih_failure_inspect
#SBATCH --output=logs/out/%j_ih_failure_inspect.out
#SBATCH --error=logs/err/%j_ih_failure_inspect.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=prod

set -euo pipefail

mkdir -p logs/out logs/err

if [ "$#" -gt 4 ]; then
  echo "Usage: sbatch $0 [<failure_csv>] [<point_csv>] [<port>] [<threshold_percent>]" >&2
  exit 1
fi

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1

FAILURE_CSV="${1:-analysis/qc_review/reproducible_qc_report/scenes_failing_distance_5pct.csv}"
POINT_CSV="${2:-analysis/qc_review/correspondence_distance_errors/per_correspondence_local_depth_errors.csv}"
PORT="${3:-8766}"
THRESHOLD_PERCENT="${4:-5}"

echo "Failure inspection app"
echo "Failure CSV: ${FAILURE_CSV}"
echo "Point CSV: ${POINT_CSV}"
echo "Port: ${PORT}"
echo "Node: ${SLURMD_NODENAME:-unknown}"

srun uv run python -m ihd.qc_review.failure_inspection_app \
  --failure-csv "${FAILURE_CSV}" \
  --point-csv "${POINT_CSV}" \
  --threshold-percent "${THRESHOLD_PERCENT}" \
  --host 0.0.0.0 \
  --port "${PORT}"
