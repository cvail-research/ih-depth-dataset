#!/usr/bin/env bash
set -euo pipefail

SCENE="${1:?Usage: bash $0 <scene_label>}"

uv run python -m ihd.evaluation.plot_scene_model_comparison \
  --scene "${SCENE}" \
  --prediction-root analysis/evaluation/depthpro_smoke_predictions/depthpro \
  --prediction-root analysis/evaluation/unidepthv2_smoke_predictions/unidepthv2 \
  --prediction-root analysis/evaluation/depthanythingv2_smoke_predictions/depthanythingv2 \
  --prediction-root analysis/evaluation/unik3d_smoke_predictions/unik3d \
  --output-dir analysis/evaluation/scene_model_comparison
