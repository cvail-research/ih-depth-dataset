#!/bin/bash
#SBATCH --job-name=annotation_workspace
#SBATCH --output=logs/out/%j_annotation_workspace.out
#SBATCH --error=logs/err/%j_annotation_workspace.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=prod

set -euo pipefail

mkdir -p logs/out logs/err

if { [ "$#" -ne 1 ] || [ ! -f "$1" ]; } && [ "$#" -ne 4 ]; then
  echo "Usage: sbatch $0 <config.env>" >&2
  echo "   or: sbatch $0 <collection> <path_name> <step> <port>" >&2
  exit 1
fi

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"

SCRIPT_ARGS=("$@")
set -- "${SCRIPT_ARGS[@]}"

export PYTHONUNBUFFERED=1

require_var() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "Missing required variable: ${name}" >&2
    exit 1
  fi
}

if [ "$#" -eq 1 ] && [ -f "$1" ]; then
  CONFIG_PATH="$1"
  set -a
  # shellcheck disable=SC1090
  source "${CONFIG_PATH}"
  set +a

  require_var COLLECTION
  require_var PATH_NAME
  require_var STEP
  PORT="${PORT:-8000}"
  FORCE_GENERATED_CYL_MODE="${FORCE_GENERATED_CYL_MODE:-0}"
  USE_REFERENCE_TARGETS_IN_GENERATED_MODE="${USE_REFERENCE_TARGETS_IN_GENERATED_MODE:-0}"
  WORKSPACE_VARIANT="${WORKSPACE_VARIANT:-}"
  DEFAULT_INIT_CYL="${DEFAULT_INIT_CYL:-}"
  DEFAULT_FIT_OPT_MODE="${DEFAULT_FIT_OPT_MODE:-all}"
else
  COLLECTION="$1"
  PATH_NAME="$2"
  STEP="$3"
  PORT="$4"
  FORCE_GENERATED_CYL_MODE="0"
  USE_REFERENCE_TARGETS_IN_GENERATED_MODE="0"
  WORKSPACE_VARIANT=""
  DEFAULT_INIT_CYL=""
  DEFAULT_FIT_OPT_MODE="all"
fi

echo "Node: $(hostname)"
echo "Workspace URL: http://$(hostname):${PORT}"
echo "Suggested SSH tunnel from laptop:"
echo "  ssh -L ${PORT}:$(hostname):${PORT} <workstation>"

CMD=(
  uv run python -m ihd.annotation_workspace.app
  --collection "${COLLECTION}"
  --path-name "${PATH_NAME}"
  --step "${STEP}"
  --host 0.0.0.0
  --port "${PORT}"
)

if [ "${FORCE_GENERATED_CYL_MODE}" = "1" ]; then
  CMD+=(--force-generated-cyl-mode)
fi
if [ "${USE_REFERENCE_TARGETS_IN_GENERATED_MODE}" = "1" ]; then
  CMD+=(--use-reference-targets-in-generated-mode)
fi
if [ -n "${WORKSPACE_VARIANT}" ]; then
  CMD+=(--workspace-variant "${WORKSPACE_VARIANT}")
fi
if [ -n "${DEFAULT_INIT_CYL}" ]; then
  CMD+=(--default-init-cyl "${DEFAULT_INIT_CYL}")
fi
if [ -n "${DEFAULT_FIT_OPT_MODE}" ]; then
  CMD+=(--default-fit-opt-mode "${DEFAULT_FIT_OPT_MODE}")
fi

srun "${CMD[@]}"
