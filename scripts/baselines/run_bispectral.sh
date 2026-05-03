#!/bin/bash
#SBATCH --job-name=ihd_bispectral
#SBATCH --output=logs/out/%j_ihd_bispectral.out
#SBATCH --error=logs/err/%j_ihd_bispectral.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=prod

set -euo pipefail

# If you want to submit with no CLI args, paste your .hdr path here.
# Use single quotes to preserve literal '$' characters in the path.
DEFAULT_SCENE_HDR='/disk/IHTest_202009_DistStA/Path3_DistStA/Path3_Step1_DistStA/IHTest_202009_Path3_Step1_LWHSI1__DistStA.hdr'

# Optional: hardcode the repo root so you can run `sbatch` from any directory.
# If empty, the script uses SLURM_SUBMIT_DIR (preferred) or falls back to a relative path.
DEFAULT_REPO_ROOT='/home/malurool/ih-depth-dataset'

# Optional: lidar ground truth for metrics.
# IMPORTANT: this must match the HSI spatial size (same H,W) for the given scene.
# Set this to the lidar.mat registered to your HSI scene.
DEFAULT_LIDAR_MAT=''

# Optional: sparse label .npz used by ihd.evaluation.evaluate_depth_prediction.
DEFAULT_LABEL_NPZ=''

# If 1, fail the job if label is missing. Set to 0 to run prediction-only.
REQUIRE_LABEL_METRICS=1

# If 1, save a PNG visualization to ihd/inference/physics_based/outputs.
SAVE_FIG=1

if [ "$#" -eq 1 ]; then
  HDR_IN="$1"
elif [ -n "${SCENE_HDR:-}" ]; then
  HDR_IN="$SCENE_HDR"
elif [ -n "${DEFAULT_SCENE_HDR:-}" ]; then
  HDR_IN="$DEFAULT_SCENE_HDR"
else
  echo "Usage: $0 <scene.hdr>" >&2
  echo "Option A (recommended): sbatch $0 '/path/with/$/scene.hdr'" >&2
  echo "Option B: sbatch --export=ALL,SCENE_HDR='/path/with/$/scene.hdr' $0" >&2
  echo "Option C: set DEFAULT_SCENE_HDR inside this script, then sbatch $0" >&2
  echo "Note: if your path contains a literal '$' character, use single quotes or escape it as '\$'." >&2
  exit 1
fi

if [ ! -f "$HDR_IN" ]; then
  echo "HDR not found: $HDR_IN" >&2
  echo "Host: $(hostname)" >&2
  echo "If this path exists on compute nodes but not on the login node, this check will fail when running locally." >&2
  exit 1
fi

HDR="$(readlink -f "$HDR_IN")"

REPO_ROOT="${IHD_REPO_ROOT:-${SLURM_SUBMIT_DIR:-${DEFAULT_REPO_ROOT:-""}}}"
if [ -z "$REPO_ROOT" ]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$REPO_ROOT"

if [ ! -f "ihd/inference/physics_based/precompute_attenuation.py" ]; then
  echo "Repo root does not look correct: $REPO_ROOT" >&2
  echo "Missing: ihd/inference/physics_based/precompute_attenuation.py" >&2
  echo "Fix: run sbatch from the repo root, or set IHD_REPO_ROOT=/path/to/ih-depth-dataset" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1

DATA_DIR="ihd/inference/physics_based/data"
PRECOMP_DIR="$DATA_DIR/precomputed"
OUT_ROOT="${OUT_ROOT:-analysis/evaluation/physics_based_predictions}"
MODEL_SLUG="bispectral"
SCENE_NAME="$(basename "${HDR}" .hdr)"
PRED_DIR="${OUT_ROOT}/${MODEL_SLUG}/${SCENE_NAME}"
PRED_PATH="${PRED_DIR}/depth_prediction.npz"

if [ ! -f "$PRECOMP_DIR/attenuation_LWHSI1.npy" ] || [ ! -f "$PRECOMP_DIR/attenuation_LWHSI2.npy" ]; then
  python ihd/inference/physics_based/precompute_attenuation.py --data-dir "$DATA_DIR"
fi

LABEL_NPZ_PATH="${LABEL_NPZ:-${DEFAULT_LABEL_NPZ:-}}"
# Retained for backward compatibility (optional manual diagnostics).
LIDAR_MAT_PATH="${LIDAR_MAT:-${DEFAULT_LIDAR_MAT:-}}"

FIG_ARGS=()
if [ "${SAVE_FIG}" -eq 1 ]; then
  FIG_ARGS=()
else
  FIG_ARGS=(--no-vis)
fi

scripts/evaluation/run_predict_bispectral.sh \
  --hdr "$HDR" \
  --data-dir "$DATA_DIR" \
  --out-dir "$PRED_DIR" \
  "${FIG_ARGS[@]}"

if [ "${REQUIRE_LABEL_METRICS}" -eq 1 ]; then
  if [ -z "${LABEL_NPZ_PATH}" ]; then
    echo "Metrics requested but no label .npz was configured." >&2
    echo "Fix: set DEFAULT_LABEL_NPZ in this script, or export LABEL_NPZ='/path/to/projected_lidar_depth_label.npz'." >&2
    echo "To run without metrics, set REQUIRE_LABEL_METRICS=0." >&2
    exit 3
  fi
  if [ ! -f "${LABEL_NPZ_PATH}" ]; then
    echo "Label file not found (required for metrics): ${LABEL_NPZ_PATH}" >&2
    exit 3
  fi
  scripts/evaluation/run_evaluate_depth_prediction.sh \
    --prediction "$PRED_PATH" \
    --label "$LABEL_NPZ_PATH" \
    --out-json "${PRED_DIR}/metrics_summary.json" \
    --out-csv "${PRED_DIR}/metrics_per_scene.csv"
  echo "Metrics saved to: ${PRED_DIR}/metrics_summary.json and ${PRED_DIR}/metrics_per_scene.csv"
else
  if [ -n "${LABEL_NPZ_PATH}" ] && [ -f "${LABEL_NPZ_PATH}" ]; then
    scripts/evaluation/run_evaluate_depth_prediction.sh \
      --prediction "$PRED_PATH" \
      --label "$LABEL_NPZ_PATH" \
      --out-json "${PRED_DIR}/metrics_summary.json" \
      --out-csv "${PRED_DIR}/metrics_per_scene.csv"
    echo "Metrics saved to: ${PRED_DIR}/metrics_summary.json and ${PRED_DIR}/metrics_per_scene.csv"
  else
    echo "Prediction generated at ${PRED_PATH} (evaluation skipped)."
  fi
fi
