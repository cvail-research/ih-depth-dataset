#!/bin/bash
#SBATCH --job-name=ih_hsi_udv2
#SBATCH --output=logs/out/%j_ih_hsi_udv2.out
#SBATCH --error=logs/err/%j_ih_hsi_udv2.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --partition=gpu

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs/out logs/err

MANIFEST="${1:?Usage: sbatch $0 <prediction_inputs.csv> [out_root]}"
OUT_ROOT="${2:-analysis/evaluation/hsi_benchmark_predictions_split07_test/unidepthv2_hsi}"
DEVICE="${DEVICE:-cuda}"
MODEL_NAME="${MODEL_NAME:-lpiccinelli/unidepth-v2-vitl14}"

scripts/evaluation/learning_hsi/run_predict_unidepthv2_hsi.sh \
  --manifest "${MANIFEST}" \
  --out-dir "${OUT_ROOT}" \
  --device "${DEVICE}" \
  --model-name "${MODEL_NAME}" \
  --no-vis

uv run python -m ihd.evaluation.evaluate_depth_prediction \
  --manifest "${OUT_ROOT}/unidepthv2_hsi_patch/prediction_manifest.csv" \
  --out-json "${OUT_ROOT}/unidepthv2_hsi_patch/metrics_summary.json" \
  --out-csv "${OUT_ROOT}/unidepthv2_hsi_patch/metrics_per_scene.csv"
