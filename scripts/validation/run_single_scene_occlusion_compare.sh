#!/bin/bash
#SBATCH --job-name=occlusion_compare
#SBATCH --output=logs/out/%j_occlusion_compare.out
#SBATCH --error=logs/err/%j_occlusion_compare.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=prod

set -euo pipefail

mkdir -p logs/out logs/err

if [ "$#" -lt 3 ] || [ "$#" -gt 6 ]; then
  echo "Usage: sbatch $0 <workspace_dir> <las> <out_dir> [<gap_m>] [<gap_ratio>] [<title_prefix>]" >&2
  exit 1
fi

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1

WORKSPACE_DIR="$1"
LAS_PATH="$2"
OUT_DIR="$3"
GAP_M="${4:-1.0}"
GAP_RATIO="${5:-0.05}"
TITLE_PREFIX="${6:-Occlusion Compare}"

REFERENCE_SRC=""
for candidate in \
  "${WORKSPACE_DIR}/image_preview.png" \
  "/disk/IHTest_202104_DistStA/Path19_DistStA/Path19_Step1_DistStA/IHTest_202104_Path19_Step1_PseudoBB_collect0_DistStA.png"
do
  if [ -f "${candidate}" ]; then
    REFERENCE_SRC="${candidate}"
    break
  fi
done

srun bash -lc "
  set -euo pipefail
  mkdir -p '${OUT_DIR}'
  if [ -n '${REFERENCE_SRC}' ]; then
    cp '${REFERENCE_SRC}' '${OUT_DIR}/reference.png'
  fi

  uv run python -m ihd.datasets.render_overlay_from_workspace \
    --workspace-dir '${WORKSPACE_DIR}' \
    --las '${LAS_PATH}' \
    --out '${OUT_DIR}/overlay_radius0.png' \
    --npz-out '${OUT_DIR}/overlay_radius0.npz' \
    --occlusion-filter-radius-px 0 \
    --occlusion-min-depth-gap-m '${GAP_M}' \
    --occlusion-min-depth-gap-ratio '${GAP_RATIO}' \
    --title-mode custom \
    --title-text '${TITLE_PREFIX}: radius=0'

  uv run python -m ihd.datasets.render_overlay_from_workspace \
    --workspace-dir '${WORKSPACE_DIR}' \
    --las '${LAS_PATH}' \
    --out '${OUT_DIR}/overlay_radius1.png' \
    --npz-out '${OUT_DIR}/overlay_radius1.npz' \
    --occlusion-filter-radius-px 1 \
    --occlusion-min-depth-gap-m '${GAP_M}' \
    --occlusion-min-depth-gap-ratio '${GAP_RATIO}' \
    --title-mode custom \
    --title-text '${TITLE_PREFIX}: radius=1'

  uv run python -m ihd.datasets.render_overlay_from_workspace \
    --workspace-dir '${WORKSPACE_DIR}' \
    --las '${LAS_PATH}' \
    --out '${OUT_DIR}/overlay_radius2.png' \
    --npz-out '${OUT_DIR}/overlay_radius2.npz' \
    --occlusion-filter-radius-px 2 \
    --occlusion-min-depth-gap-m '${GAP_M}' \
    --occlusion-min-depth-gap-ratio '${GAP_RATIO}' \
    --title-mode custom \
    --title-text '${TITLE_PREFIX}: radius=2'

  printf 'workspace_dir,%s\nlas,%s\ngap_m,%s\ngap_ratio,%s\noutputs,%s\n' \
    '${WORKSPACE_DIR}' '${LAS_PATH}' '${GAP_M}' '${GAP_RATIO}' '${OUT_DIR}' > '${OUT_DIR}/comparison_manifest.csv'
"
