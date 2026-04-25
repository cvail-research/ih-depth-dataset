#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "Submitting one Slurm job for the full platform-sphere preprocessing batch."
echo "Use scripts/validation/run_platform_sphere_preprocessing_batch.sh directly if you prefer."

sbatch scripts/validation/run_platform_sphere_preprocessing_batch.sh "$@"
