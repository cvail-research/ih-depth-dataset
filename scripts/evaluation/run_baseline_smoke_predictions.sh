#!/bin/bash
#SBATCH --job-name=ih_baseline_smoke
#SBATCH --output=logs/out/%j_ih_baseline_smoke.out
#SBATCH --error=logs/err/%j_ih_baseline_smoke.err
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

SCENE_MANIFEST="${1:-analysis/qc_review/reproducible_qc_report/scenes_accepted_by_rmse5px_distance_5pct_with_drop_rule.csv}"
LIMIT="${2:-3}"
OUT_ROOT="${3:-analysis/evaluation/baseline_smoke_predictions}"
MODELS_CSV="${4:-unik3d,unidepthv2,depthanythingv2,depthpro,bispectral}"

INPUT_MANIFEST="${OUT_ROOT}/prediction_inputs.csv"
mkdir -p "${OUT_ROOT}"

uv run python -m ihd.evaluation.build_prediction_input_manifest \
  --scene-manifest "${SCENE_MANIFEST}" \
  --limit "${LIMIT}" \
  --out-csv "${INPUT_MANIFEST}"

IFS=',' read -r -a MODELS <<< "${MODELS_CSV}"
for MODEL in "${MODELS[@]}"; do
  MODEL="$(echo "${MODEL}" | xargs)"
  echo "[$(date --iso-8601=seconds)] Running ${MODEL}"
  case "${MODEL}" in
    unik3d)
      if [[ -n "${UNIK3D_PYTHON:-}" ]]; then
        PYTHON_BIN="${UNIK3D_PYTHON}" scripts/evaluation/learning_pseudogrey/run_predict_unik3d.sh --manifest "${INPUT_MANIFEST}" --out-dir "${OUT_ROOT}" --device cpu --no-vis
      else
        scripts/evaluation/learning_pseudogrey/run_predict_unik3d.sh --manifest "${INPUT_MANIFEST}" --out-dir "${OUT_ROOT}" --device cpu --no-vis
      fi
      ;;
    unidepthv2)
      if [[ -n "${UNIDEPTHV2_PYTHON:-}" ]]; then
        PYTHON_BIN="${UNIDEPTHV2_PYTHON}" scripts/evaluation/learning_pseudogrey/run_predict_unidepthv2.sh --manifest "${INPUT_MANIFEST}" --out-dir "${OUT_ROOT}" --no-vis
      else
        scripts/evaluation/learning_pseudogrey/run_predict_unidepthv2.sh --manifest "${INPUT_MANIFEST}" --out-dir "${OUT_ROOT}" --no-vis
      fi
      ;;
    depthanythingv2)
      if [[ -n "${DEPTHANYTHINGV2_PYTHON:-}" ]]; then
        PYTHON_BIN="${DEPTHANYTHINGV2_PYTHON}" scripts/evaluation/learning_pseudogrey/run_predict_depthanythingv2.sh --manifest "${INPUT_MANIFEST}" --out-dir "${OUT_ROOT}" --no-vis
      else
        scripts/evaluation/learning_pseudogrey/run_predict_depthanythingv2.sh --manifest "${INPUT_MANIFEST}" --out-dir "${OUT_ROOT}" --no-vis
      fi
      ;;
    depthpro)
      if [[ -n "${DEPTHPRO_PYTHON:-}" ]]; then
        PYTHON_BIN="${DEPTHPRO_PYTHON}" scripts/evaluation/learning_pseudogrey/run_predict_depthpro.sh --manifest "${INPUT_MANIFEST}" --out-dir "${OUT_ROOT}" --no-vis
      else
        scripts/evaluation/learning_pseudogrey/run_predict_depthpro.sh --manifest "${INPUT_MANIFEST}" --out-dir "${OUT_ROOT}" --no-vis
      fi
      ;;
    bispectral)
      scripts/evaluation/physics_based/run_predict_bispectral.sh --manifest "${INPUT_MANIFEST}" --out-dir "${OUT_ROOT}" --no-vis
      ;;
    *)
      echo "Unknown model: ${MODEL}" >&2
      exit 2
      ;;
  esac

  PRED_MANIFEST="${OUT_ROOT}/${MODEL}/prediction_manifest.csv"
  if [[ -s "${PRED_MANIFEST}" ]]; then
    uv run python -m ihd.evaluation.evaluate_depth_prediction \
      --manifest "${PRED_MANIFEST}" \
      --out-json "${OUT_ROOT}/${MODEL}/metrics_summary.json" \
      --out-csv "${OUT_ROOT}/${MODEL}/metrics_per_scene.csv"
  fi
done

echo "[$(date --iso-8601=seconds)] Baseline smoke predictions finished"
