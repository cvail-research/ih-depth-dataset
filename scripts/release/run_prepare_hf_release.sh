#!/usr/bin/env bash
set -euo pipefail

uv run python ihd/datasets/prepare_hf_release.py \
  --frozen-manifest manifests/06_frozen_manifest_v0.csv \
  --output-dir analysis/huggingface_release \
  --repo-id cvail-research/ih-depth-dataset
