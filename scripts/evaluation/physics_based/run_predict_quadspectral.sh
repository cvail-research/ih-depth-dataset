#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  exec "${PYTHON_BIN}" -m ihd.inference.physics_based.predict_quadspectral "$@"
fi

exec uv run --frozen --no-sync python -m ihd.inference.physics_based.predict_quadspectral "$@"
