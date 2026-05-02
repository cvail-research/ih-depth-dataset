#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCENE_LIST_CSV="${1:-analysis/qc_review/occlusion_review/occlusion_review_release_scenes.csv}"
REVIEWER_ID="${2:-occlusion_release}"
RESULTS_ROOT="${3:-analysis/lidar_labeling}"
PORT="${4:-8765}"

cd "${REPO_ROOT}"

echo "Submitting occlusion QC review"
echo "Reviewer: ${REVIEWER_ID}"
echo "Results root: ${RESULTS_ROOT}"
echo "Port: ${PORT}"
echo "Scene list: ${SCENE_LIST_CSV}"

sbatch scripts/validation/run_qc_review_app.sh "${REVIEWER_ID}" "${RESULTS_ROOT}" "${PORT}" "${SCENE_LIST_CSV}" occlusion
