#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "Submitting scene model comparison batch render."
sbatch --job-name=ih_scene_cmp_batch --output=logs/out/%j_scene_cmp_batch.out --error=logs/err/%j_scene_cmp_batch.err \
  --time=04:00:00 --ntasks=1 --cpus-per-task=4 --mem=16G --partition=prod \
  scripts/evaluation/run_render_scene_model_comparison_batch.sh "$@"
