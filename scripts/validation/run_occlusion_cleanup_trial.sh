#!/bin/bash
#SBATCH --job-name=occlusion_cleanup_trial
#SBATCH --output=logs/out/%j_occlusion_cleanup_trial.out
#SBATCH --error=logs/err/%j_occlusion_cleanup_trial.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=prod

set -euo pipefail

mkdir -p logs/out logs/err

if [ "$#" -lt 3 ] || [ "$#" -gt 7 ]; then
  echo "Usage: sbatch $0 <workspace_dir> <raw_las> <out_dir> [<sphere_specs>] [<box_specs>] [<gap_m>] [<gap_ratio>]" >&2
  echo "  sphere_specs format: x,y,z,radius;x,y,z,radius;..." >&2
  echo "  box_specs format: x_min,x_max,y_min,y_max,z_min,z_max;..." >&2
  exit 1
fi

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1

WORKSPACE_DIR="$1"
RAW_LAS="$2"
OUT_DIR="$3"
SPHERE_SPECS="${4:-}"
BOX_SPECS="${5:-}"
GAP_M="${6:-1.0}"
GAP_RATIO="${7:-0.05}"

PREPROCESS_DIR="${OUT_DIR}/preprocess"
RAW_OVERLAY="${OUT_DIR}/raw_overlay.png"
CLEAN_OVERLAY="${OUT_DIR}/clean_overlay.png"
MANIFEST="${OUT_DIR}/trial_manifest.csv"

mkdir -p "${OUT_DIR}"

if [ -n "${SPHERE_SPECS}" ]; then
  IFS=';' read -r -a sphere_specs <<< "${SPHERE_SPECS}"
  for sphere_spec in "${sphere_specs[@]}"; do
    [ -n "${sphere_spec}" ] || continue
  done
fi

if [ -n "${BOX_SPECS}" ]; then
  IFS=';' read -r -a box_specs <<< "${BOX_SPECS}"
  for box_spec in "${box_specs[@]}"; do
    [ -n "${box_spec}" ] || continue
  done
fi

PREPROCESS_CMD=(uv run python ihd/datasets/preprocess_las_for_projection.py)
PREPROCESS_CMD+=(--las "${RAW_LAS}" --out-dir "${PREPROCESS_DIR}")
PREPROCESS_CMD+=(--profile-name "occlusion_cleanup_trial" --projection-voxel 0.03 --sor-k 50 --sor-std-ratio 2.0 --projection-use-sor)

if [ -n "${SPHERE_SPECS}" ]; then
  IFS=';' read -r -a sphere_specs <<< "${SPHERE_SPECS}"
  for sphere_spec in "${sphere_specs[@]}"; do
    [ -n "${sphere_spec}" ] || continue
    PREPROCESS_CMD+=("--exclude-sphere=${sphere_spec}")
  done
fi

if [ -n "${BOX_SPECS}" ]; then
  IFS=';' read -r -a box_specs <<< "${BOX_SPECS}"
  for box_spec in "${box_specs[@]}"; do
    [ -n "${box_spec}" ] || continue
    PREPROCESS_CMD+=("--exclude-box=${box_spec}")
  done
fi

"${PREPROCESS_CMD[@]}"

CLEAN_LAS="$(find "${PREPROCESS_DIR}" -maxdepth 1 -name '*_projection_clean.las' | head -n 1)"
if [ -z "${CLEAN_LAS}" ]; then
  echo "Could not locate cleaned LAS output" >&2
  exit 1
fi

uv run python -m ihd.datasets.render_overlay_from_workspace \
  --workspace-dir "${WORKSPACE_DIR}" \
  --las "${RAW_LAS}" \
  --out "${RAW_OVERLAY}" \
  --title-mode custom \
  --title-text "Raw overlay"

uv run python -m ihd.datasets.render_overlay_from_workspace \
  --workspace-dir "${WORKSPACE_DIR}" \
  --las "${CLEAN_LAS}" \
  --out "${CLEAN_OVERLAY}" \
  --title-mode custom \
  --title-text "Cleaned overlay"

printf 'workspace_dir,%s\nraw_las,%s\nclean_las,%s\ngap_m,%s\ngap_ratio,%s\nraw_overlay,%s\nclean_overlay,%s\n' \
  "${WORKSPACE_DIR}" "${RAW_LAS}" "${CLEAN_LAS}" "${GAP_M}" "${GAP_RATIO}" "${RAW_OVERLAY}" "${CLEAN_OVERLAY}" > "${MANIFEST}"
