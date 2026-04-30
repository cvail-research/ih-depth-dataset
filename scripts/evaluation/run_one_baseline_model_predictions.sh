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

MODEL="${1:?Usage: sbatch $0 <model> [scene_manifest] [limit] [out_root]}"
SCENE_MANIFEST="${2:-analysis/qc_review/reproducible_qc_report/scenes_accepted_by_rmse5px_distance_5pct_with_drop_rule.csv}"
LIMIT="${3:-3}"
OUT_ROOT="${4:-analysis/evaluation/baseline_smoke_predictions}"

INPUT_MANIFEST="${OUT_ROOT}/prediction_inputs.csv"
mkdir -p "${OUT_ROOT}"

uv run python -m ihd.evaluation.build_prediction_input_manifest \
  --scene-manifest "${SCENE_MANIFEST}" \
  --limit "${LIMIT}" \
  --out-csv "${INPUT_MANIFEST}"

case "${MODEL}" in
  unik3d)
    if [[ -n "${UNIK3D_PYTHON:-}" ]]; then
      PYTHON_BIN="${UNIK3D_PYTHON}" scripts/evaluation/run_predict_unik3d.sh --manifest "${INPUT_MANIFEST}" --out-dir "${OUT_ROOT}" --device cpu --no-vis
    else
      scripts/evaluation/run_predict_unik3d.sh --manifest "${INPUT_MANIFEST}" --out-dir "${OUT_ROOT}" --device cpu --no-vis
    fi
    ;;
  unidepthv2)
    PYTHON_BIN="${UNIDEPTHV2_PYTHON:-python}" \
      scripts/evaluation/run_predict_unidepthv2.sh --manifest "${INPUT_MANIFEST}" --out-dir "${OUT_ROOT}" --no-vis
    ;;
  depthanythingv2)
    PYTHON_BIN="${DEPTHANYTHINGV2_PYTHON:-python}" \
      scripts/evaluation/run_predict_depthanythingv2.sh --manifest "${INPUT_MANIFEST}" --out-dir "${OUT_ROOT}" --no-vis
    ;;
  depthpro)
    PYTHON_BIN="${DEPTHPRO_PYTHON:-python}" \
      scripts/evaluation/run_predict_depthpro.sh --manifest "${INPUT_MANIFEST}" --out-dir "${OUT_ROOT}" --no-vis
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
