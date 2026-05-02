#!/usr/bin/env bash
set -euo pipefail

uv run python -m ihd.evaluation.build_scene_hardness_table \
  --metrics-csv analysis/evaluation/depthpro_smoke_predictions/depthpro/metrics_per_scene.csv \
  --metrics-csv analysis/evaluation/unidepthv2_smoke_predictions/unidepthv2/metrics_per_scene.csv \
  --metrics-csv analysis/evaluation/depthanythingv2_smoke_predictions/depthanythingv2/metrics_per_scene.csv \
  --metrics-csv analysis/evaluation/unik3d_smoke_predictions/unik3d/metrics_per_scene.csv \
  --score-col abs_rel \
  --out-csv analysis/evaluation/scene_hardness_table.csv \
  --out-json analysis/evaluation/scene_hardness_table_summary.json
