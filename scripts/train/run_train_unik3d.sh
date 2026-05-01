#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  exec "${PYTHON_BIN}" -m ihd.training.unik3d_train "$@"
fi

exec uv run --extra unik3d python -m ihd.training.unik3d_train "$@"
