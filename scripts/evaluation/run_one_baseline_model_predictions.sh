#!/bin/bash
#SBATCH --job-name=ih_one_baseline
#SBATCH --output=logs/out/%j_ih_one_baseline.out
#SBATCH --error=logs/err/%j_ih_one_baseline.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=prod

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p logs/out logs/err

MODEL="${1:?Usage: sbatch $0 <model> [scene_manifest] [device] [out_root]}"
SCENE_MANIFEST="${2:-manifests/06_frozen_manifest_v0.csv}"
DEVICE="${3:-cuda}"
OUT_ROOT="${4:-analysis/evaluation/baseline_predictions_full}"
mkdir -p "${OUT_ROOT}"

case "${MODEL}" in
  unik3d)
    if [[ -n "${UNIK3D_PYTHON:-}" ]]; then
      PYTHON_BIN="${UNIK3D_PYTHON}" scripts/evaluation/learning_pseudogrey/run_predict_unik3d.sh --scene-manifest "${SCENE_MANIFEST}" --out-dir "${OUT_ROOT}" --device "${DEVICE}" --no-vis
    else
      scripts/evaluation/learning_pseudogrey/run_predict_unik3d.sh --scene-manifest "${SCENE_MANIFEST}" --out-dir "${OUT_ROOT}" --device "${DEVICE}" --no-vis
    fi
    ;;
  unidepthv2)
    if [[ -n "${UNIDEPTHV2_PYTHON:-}" ]]; then
      PYTHON_BIN="${UNIDEPTHV2_PYTHON}" scripts/evaluation/learning_pseudogrey/run_predict_unidepthv2.sh --scene-manifest "${SCENE_MANIFEST}" --out-dir "${OUT_ROOT}" --device "${DEVICE}" --no-vis
    else
      scripts/evaluation/learning_pseudogrey/run_predict_unidepthv2.sh --scene-manifest "${SCENE_MANIFEST}" --out-dir "${OUT_ROOT}" --device "${DEVICE}" --no-vis
    fi
    ;;
  depthanythingv2)
    if [[ -n "${DEPTHANYTHINGV2_PYTHON:-}" ]]; then
      PYTHON_BIN="${DEPTHANYTHINGV2_PYTHON}" scripts/evaluation/learning_pseudogrey/run_predict_depthanythingv2.sh --scene-manifest "${SCENE_MANIFEST}" --out-dir "${OUT_ROOT}" --device "${DEVICE}" --no-vis
    else
      scripts/evaluation/learning_pseudogrey/run_predict_depthanythingv2.sh --scene-manifest "${SCENE_MANIFEST}" --out-dir "${OUT_ROOT}" --device "${DEVICE}" --no-vis
    fi
    ;;
  depthpro)
    if [[ -n "${DEPTHPRO_PYTHON:-}" ]]; then
      PYTHON_BIN="${DEPTHPRO_PYTHON}" scripts/evaluation/learning_pseudogrey/run_predict_depthpro.sh --scene-manifest "${SCENE_MANIFEST}" --out-dir "${OUT_ROOT}" --device "${DEVICE}" --no-vis
    else
      scripts/evaluation/learning_pseudogrey/run_predict_depthpro.sh --scene-manifest "${SCENE_MANIFEST}" --out-dir "${OUT_ROOT}" --device "${DEVICE}" --no-vis
    fi
    ;;
  bispectral)
    scripts/evaluation/physics_based/run_predict_bispectral.sh --scene-manifest "${SCENE_MANIFEST}" --out-dir "${OUT_ROOT}" --no-vis
    ;;
  *)
    echo "Unknown model: ${MODEL}" >&2
    exit 2
    ;;
esac

PRED_MANIFEST="${OUT_ROOT}/${MODEL}/prediction_manifest.csv"
uv run python -m ihd.evaluation.evaluate_depth_prediction \
  --manifest "${PRED_MANIFEST}" \
  --out-json "${OUT_ROOT}/${MODEL}/metrics_summary.json" \
  --out-csv "${OUT_ROOT}/${MODEL}/metrics_per_scene.csv"

echo "[$(date --iso-8601=seconds)] Finished ${MODEL}"
