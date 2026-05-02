#!/usr/bin/env bash
set -euo pipefail

uv run python ihd/datasets/build_frozen_manifest_v0.py \
  --quality-manifest manifests/05_scene_quality_manifest_current.csv \
  --cleanup-manifest manifests/06_occlusion_cleanup_manifest_current.csv \
  --output-csv manifests/07_frozen_manifest_v0.csv \
  --output-summary-json manifests/07_frozen_manifest_v0_summary.json
