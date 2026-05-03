#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs/out logs/err

OUT_MANIFEST="${OUT_MANIFEST:-analysis/evaluation/manifests/bispectral_lwhsi1_test_manifest.csv}"
OUT_ROOT="${OUT_ROOT:-analysis/evaluation/bispectral_lwhsi1_test}"
SENSOR_ID="${SENSOR_ID:-LWHSI1}"
RELEASE_DECISION="${RELEASE_DECISION:-include}"
SCENE_FILTER="${SCENE_FILTER:-}"
SPLIT_FILTER="${SPLIT_FILTER:-}"
LIMIT="${LIMIT:-1}"

BUILD_ARGS=(
  --frozen-manifest "manifests/06_frozen_manifest_v0.csv"
  --split-manifest "manifests/07_split_definition_v0/scene_splits.csv"
  --out-csv "${OUT_MANIFEST}"
  --sensor-id "${SENSOR_ID}"
  --release-decision "${RELEASE_DECISION}"
  --limit "${LIMIT}"
)

if [[ -n "${SCENE_FILTER}" ]]; then
  BUILD_ARGS+=(--scene "${SCENE_FILTER}")
fi
if [[ -n "${SPLIT_FILTER}" ]]; then
  BUILD_ARGS+=(--split "${SPLIT_FILTER}")
fi

uv run python -m ihd.evaluation.build_bispectral_test_manifest "${BUILD_ARGS[@]}"

sbatch --job-name=ih_bispec_test \
  --output=logs/out/%j_ih_bispec_test.out \
  --error=logs/err/%j_ih_bispec_test.err \
  scripts/evaluation/run_one_baseline_model_predictions.sh \
  bispectral "${OUT_MANIFEST}" cpu "${OUT_ROOT}"
