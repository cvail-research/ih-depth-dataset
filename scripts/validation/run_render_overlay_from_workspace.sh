#!/bin/bash
#SBATCH --job-name=render_overlay
#SBATCH --output=logs/out/%j_render_overlay.out
#SBATCH --error=logs/err/%j_render_overlay.err
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=prod

set -euo pipefail

mkdir -p logs/out logs/err

if [ "$#" -lt 3 ] || [ "$#" -gt 5 ]; then
  echo "Usage: sbatch $0 <workspace_dir> <las> <out_png> [<title_mode:none|auto|custom>] [<title_text>]" >&2
  exit 1
fi

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1

WORKSPACE_DIR="$1"
LAS_PATH="$2"
OUT_PNG="$3"
TITLE_MODE="${4:-none}"
TITLE_TEXT="${5:-}"

CMD=(
  uv run python -m ihd.datasets.render_overlay_from_workspace
  --workspace-dir "${WORKSPACE_DIR}"
  --las "${LAS_PATH}"
  --out "${OUT_PNG}"
  --title-mode "${TITLE_MODE}"
)

if [ "${TITLE_MODE}" = "custom" ]; then
  CMD+=(--title-text "${TITLE_TEXT}")
fi

srun "${CMD[@]}"
