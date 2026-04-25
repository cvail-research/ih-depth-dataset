#!/bin/bash
#SBATCH --job-name=annotation_workspace_nocyl
#SBATCH --output=logs/out/%j_annotation_workspace_nocyl.out
#SBATCH --error=logs/err/%j_annotation_workspace_nocyl.err
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
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
else
  COLLECTION="$1"
  PATH_NAME="$2"
  STEP="$3"
  PORT="$4"
fi

echo "Node: $(hostname)"
echo "Workspace URL: http://$(hostname):${PORT}"
echo "Suggested SSH tunnel from laptop:"
echo "  ssh -L ${PORT}:$(hostname):${PORT} <workstation>"

srun uv run python -m ihd.annotation_workspace_nocyl.app \
  --collection "${COLLECTION}" \
  --path-name "${PATH_NAME}" \
  --step "${STEP}" \
  --host 0.0.0.0 \
  --port "${PORT}"
