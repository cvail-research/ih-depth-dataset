#!/bin/bash
#SBATCH --job-name=ihd_quadspectral
#SBATCH --output=logs/out/%j_ihd_quadspectral.out
#SBATCH --error=logs/err/%j_ihd_quadspectral.err
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

# If 1, fail the job if lidar is missing. Set to 0 to run without metrics.
REQUIRE_LIDAR_METRICS=0

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

if [ ! -f "$PRECOMP_DIR/attenuation_LWHSI1.npy" ] || [ ! -f "$PRECOMP_DIR/attenuation_LWHSI2.npy" ]; then
  python ihd/inference/physics_based/precompute_attenuation.py --data-dir "$DATA_DIR"
fi

LIDAR_MAT_PATH="${LIDAR_MAT:-${DEFAULT_LIDAR_MAT:-}}"
LIDAR_ARGS=()

if [ "${REQUIRE_LIDAR_METRICS}" -eq 1 ]; then
  if [ -z "${LIDAR_MAT_PATH}" ]; then
    echo "Metrics requested but no lidar file was configured." >&2
    echo "Fix: set DEFAULT_LIDAR_MAT inside this script, or export LIDAR_MAT='/path/to/lidar.mat'." >&2
    echo "To run without metrics, set REQUIRE_LIDAR_METRICS=0." >&2
    exit 3
  fi
  if [ ! -f "${LIDAR_MAT_PATH}" ]; then
    echo "Lidar file not found (required for metrics): ${LIDAR_MAT_PATH}" >&2
    exit 3
  fi
  LIDAR_ARGS=(--lidar-mat "${LIDAR_MAT_PATH}")
else
  if [ -n "${LIDAR_MAT_PATH}" ]; then
    if [ -f "${LIDAR_MAT_PATH}" ]; then
      LIDAR_ARGS=(--lidar-mat "${LIDAR_MAT_PATH}")
    else
      echo "Lidar file not found (metrics disabled): ${LIDAR_MAT_PATH}" >&2
    fi
  fi
fi

FIG_ARGS=()
if [ "${SAVE_FIG}" -eq 1 ]; then
  FIG_ARGS=(--save-fig)
fi

python ihd/inference/physics_based/run_quadspectral.py --hsi-hdr "$HDR" --data-dir "$DATA_DIR" --save-npy "${FIG_ARGS[@]}" "${LIDAR_ARGS[@]}"
