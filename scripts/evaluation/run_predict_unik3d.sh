#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/home/guille/.conda/envs/deeptr/bin/python}"
exec "${PYTHON_BIN}" -m ihd.evaluation.predict_unik3d "$@"

