#!/usr/bin/env bash
set -euo pipefail

uv run python -m ihd.evaluation.build_scene_hardness_table \
  --metrics-csv analysis/evaluation/baseline_predictions_full/depthpro/metrics_per_scene.csv \
  --metrics-csv analysis/evaluation/baseline_predictions_full/unidepthv2/metrics_per_scene.csv \
  --metrics-csv analysis/evaluation/baseline_predictions_full/depthanythingv2/metrics_per_scene.csv \
  --metrics-csv analysis/evaluation/baseline_predictions_full/unik3d/metrics_per_scene.csv \
  --score-col abs_rel \
  --out-csv analysis/evaluation/scene_hardness_table.csv \
  --out-json analysis/evaluation/scene_hardness_table_summary.json
