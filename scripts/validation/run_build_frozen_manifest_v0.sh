#!/usr/bin/env bash
set -euo pipefail

uv run python ihd/datasets/build_scene_facts_manifest_v0.py \
  --with-prior-manifest manifests/archive/legacy_pool_manifests_v0/01_with_prior_cyl_n60.csv \
  --without-prior-manifest manifests/archive/legacy_pool_manifests_v0/02_without_prior_cyl_n246.csv \
  --fitted-manifest manifests/archive/legacy_pool_manifests_v0/03_unified_own_fitted_cyl_n232.csv \
  --output-csv manifests/01_scene_facts_n306.csv \
  --output-summary-json manifests/01_scene_facts_n306_summary.json

uv run python ihd/datasets/build_frozen_manifest_v0.py \
  --scene-facts-manifest manifests/01_scene_facts_n306.csv \
  --quality-manifest manifests/03_scene_quality_manifest_current.csv \
  --cleanup-manifest manifests/04_occlusion_cleanup_manifest_current.csv \
  --output-csv manifests/06_frozen_manifest_v0.csv \
  --output-summary-json manifests/06_frozen_manifest_v0_summary.json
