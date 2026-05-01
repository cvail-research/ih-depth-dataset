#!/bin/bash
#SBATCH --job-name=ih_hsi_dav2
#SBATCH --output=logs/out/%j_ih_hsi_dav2.out
#SBATCH --error=logs/err/%j_ih_hsi_dav2.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs/out logs/err

MANIFEST="${1:-analysis/evaluation/baseline_smoke_predictions/prediction_inputs.csv}"
OUT_ROOT="${2:-analysis/evaluation/depthanythingv2_hsi_smoke_predictions}"

scripts/hsi/run_predict_depthanythingv2_hsi.sh \
  --manifest "${MANIFEST}" \
  --out-dir "${OUT_ROOT}" \
  --device cuda \
  --no-vis

uv run python -m ihd.evaluation.evaluate_depth_prediction \
  --manifest "${OUT_ROOT}/depthanythingv2_hsi_patch/prediction_manifest.csv" \
  --out-json "${OUT_ROOT}/depthanythingv2_hsi_patch/metrics_summary.json" \
  --out-csv "${OUT_ROOT}/depthanythingv2_hsi_patch/metrics_per_scene.csv"
