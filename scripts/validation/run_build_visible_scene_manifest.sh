#!/bin/bash
set -euo pipefail

uv run python ihd/datasets/build_visible_scene_manifest_v0.py "$@"
