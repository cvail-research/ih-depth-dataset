#!/usr/bin/env bash
set -euo pipefail

uv run python -m ihd.evaluation.build_common_scene_manifest "$@"
