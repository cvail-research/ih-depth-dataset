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
    ATTENUATION_PROFILE="${ATTENUATION_PROFILE:-auto}"
    scripts/evaluation/physics_based/run_predict_bispectral.sh --scene-manifest "${SCENE_MANIFEST}" --out-dir "${OUT_ROOT}" --attenuation-profile "${ATTENUATION_PROFILE}" --no-vis
    ;;
  quadspectral)
    ATTENUATION_PROFILE="${ATTENUATION_PROFILE:-auto}"
    COR_COEFF="${COR_COEFF:-1.0}"
    scripts/evaluation/physics_based/run_predict_quadspectral.sh \
      --scene-manifest "${SCENE_MANIFEST}" \
      --out-dir "${OUT_ROOT}" \
      --attenuation-profile "${ATTENUATION_PROFILE}" \
      --cor-coeff "${COR_COEFF}" \
      --no-vis
    ;;
  hyperspectral)
    ATTENUATION_PROFILE="${ATTENUATION_PROFILE:-auto}"
    DOWNWELLING="${DOWNWELLING:-true}"
    CHUNK_SIZE="${CHUNK_SIZE:-128}"
    EMISS_REG="${EMISS_REG:-1e7}"
    TV_REG="${TV_REG:-1e-4}"
    HYPER_ARGS=(
      --scene-manifest "${SCENE_MANIFEST}"
      --out-dir "${OUT_ROOT}"
      --attenuation-profile "${ATTENUATION_PROFILE}"
      --chunk-size "${CHUNK_SIZE}"
      --emiss-reg "${EMISS_REG}"
      --tv-reg "${TV_REG}"
      --no-vis
    )
    if [[ -n "${HYPER_LR:-}" ]]; then
      HYPER_ARGS+=(--lr "${HYPER_LR}")
    fi
    if [[ -n "${HYPER_NUM_ITERATIONS:-}" ]]; then
      HYPER_ARGS+=(--num-iterations "${HYPER_NUM_ITERATIONS}")
    fi
    if [[ "${DOWNWELLING}" == "true" ]]; then
      HYPER_ARGS+=(--downwelling)
    else
      HYPER_ARGS+=(--no-downwelling)
    fi
    scripts/evaluation/physics_based/run_predict_hyperspectral.sh "${HYPER_ARGS[@]}"
    ;;
  newcrf_thr_ms2|bts_thr_ms2|adabins_thr_ms2|dpt_large_thr_ms2)
    case "${MODEL}" in
      newcrf_thr_ms2) SUPDEPTH_VARIANT="newcrf" ;;
      bts_thr_ms2) SUPDEPTH_VARIANT="bts" ;;
      adabins_thr_ms2) SUPDEPTH_VARIANT="adabins" ;;
      dpt_large_thr_ms2) SUPDEPTH_VARIANT="dpt_large" ;;
    esac
    SUPDEPTH_THIRD_PARTY_ROOT="${SUPDEPTH_THIRD_PARTY_ROOT:-third_party/SupDepth4Thermal}"
    SUPDEPTH_CHECKPOINTS_DIR="${SUPDEPTH_CHECKPOINTS_DIR:-${SUPDEPTH_THIRD_PARTY_ROOT}/checkpoints}"
    case "${MODEL}" in
      newcrf_thr_ms2) CKPT_NAME="MS2_MD_NeWCRF_THR_ckpt.ckpt" ;;
      bts_thr_ms2) CKPT_NAME="MS2_MD_BTS_THR_ckpt.ckpt" ;;
      adabins_thr_ms2) CKPT_NAME="MS2_MD_AdaBins_THR_ckpt.ckpt" ;;
      dpt_large_thr_ms2) CKPT_NAME="MS2_MD_DPT_Large_THR_ckpt.ckpt" ;;
    esac
    SUPDEPTH_CHECKPOINT_PATH="${SUPDEPTH_CHECKPOINT_PATH:-${SUPDEPTH_CHECKPOINTS_DIR}/${CKPT_NAME}}"
    SUPDEPTH_CALIBRATION_SCALE="${SUPDEPTH_CALIBRATION_SCALE:-}"
    SUPDEPTH_CALIBRATION_JSON="${SUPDEPTH_CALIBRATION_JSON:-}"
    THERMAL_ARGS=(
      --model-variant "${SUPDEPTH_VARIANT}"
      --scene-manifest "${SCENE_MANIFEST}"
      --out-dir "${OUT_ROOT}"
      --device "${DEVICE}"
      --third-party-root "${SUPDEPTH_THIRD_PARTY_ROOT}"
      --checkpoint-path "${SUPDEPTH_CHECKPOINT_PATH}"
      --no-vis
    )
    if [[ -n "${SUPDEPTH_CALIBRATION_SCALE}" ]]; then
      THERMAL_ARGS+=(--calibration-scale "${SUPDEPTH_CALIBRATION_SCALE}")
    fi
    if [[ -n "${SUPDEPTH_CALIBRATION_JSON}" ]]; then
      THERMAL_ARGS+=(--calibration-json "${SUPDEPTH_CALIBRATION_JSON}")
    fi
    scripts/evaluation/learning_pseudogrey/run_predict_supdepth4thermal.sh "${THERMAL_ARGS[@]}"
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
