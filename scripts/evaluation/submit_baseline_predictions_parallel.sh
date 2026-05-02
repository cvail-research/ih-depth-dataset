#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SCENE_MANIFEST="${1:-manifests/07_frozen_manifest_v0.csv}"
OUT_ROOT="${2:-analysis/evaluation/baseline_predictions_full}"

mkdir -p logs/out logs/err "${OUT_ROOT}"

echo "Submitting baseline prediction batch jobs from ${SCENE_MANIFEST}"

sbatch --job-name=ih_pred_depthpro \
  --output=logs/out/%j_ih_pred_depthpro.out \
  --error=logs/err/%j_ih_pred_depthpro.err \
  --time=24:00:00 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem=14G \
  --partition=gpu \
  scripts/evaluation/run_one_baseline_model_predictions.sh depthpro "${SCENE_MANIFEST}" cuda "${OUT_ROOT}"

sbatch --job-name=ih_pred_unidepthv2 \
  --output=logs/out/%j_ih_pred_unidepthv2.out \
  --error=logs/err/%j_ih_pred_unidepthv2.err \
  --time=24:00:00 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=12G \
  --partition=prod \
  scripts/evaluation/run_one_baseline_model_predictions.sh unidepthv2 "${SCENE_MANIFEST}" cpu "${OUT_ROOT}"

sbatch --job-name=ih_pred_depthanythingv2 \
  --output=logs/out/%j_ih_pred_depthanythingv2.out \
  --error=logs/err/%j_ih_pred_depthanythingv2.err \
  --time=24:00:00 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=12G \
  --partition=prod \
  scripts/evaluation/run_one_baseline_model_predictions.sh depthanythingv2 "${SCENE_MANIFEST}" cpu "${OUT_ROOT}"

sbatch --job-name=ih_pred_unik3d \
  --output=logs/out/%j_ih_pred_unik3d.out \
  --error=logs/err/%j_ih_pred_unik3d.err \
  --time=24:00:00 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=12G \
  --partition=prod \
  scripts/evaluation/run_one_baseline_model_predictions.sh unik3d "${SCENE_MANIFEST}" cpu "${OUT_ROOT}"
