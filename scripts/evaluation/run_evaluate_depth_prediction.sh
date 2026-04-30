#!/bin/bash
# Lightweight wrapper for evaluating IH-Depth predictions. This can run on a
# login node for a small single-scene smoke test; use sbatch for large manifests.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

uv run python -m ihd.evaluation.evaluate_depth_prediction "$@"
