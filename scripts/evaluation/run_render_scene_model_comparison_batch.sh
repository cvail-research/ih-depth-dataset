#!/usr/bin/env bash
set -euo pipefail

uv run python -m ihd.evaluation.render_scene_model_comparison_batch "$@"
