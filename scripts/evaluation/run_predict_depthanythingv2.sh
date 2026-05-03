#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  exec "${PYTHON_BIN}" -m ihd.inference.learning_pseudogrey.predict_depthanythingv2 "$@"
fi

exec uv run --frozen --no-sync --extra depthanythingv2 python -m ihd.inference.learning_pseudogrey.predict_depthanythingv2 "$@"
