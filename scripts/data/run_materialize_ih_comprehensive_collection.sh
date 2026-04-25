#!/bin/bash
#SBATCH --job-name=ih_comprehensive
#SBATCH --output=logs/out/%j_ih_comprehensive.out
#SBATCH --error=logs/err/%j_ih_comprehensive.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=prod

set -euo pipefail

mkdir -p logs/out logs/err

: "${COLLECTION:?Set COLLECTION, e.g. IHTest_202104_DistStA}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
MANIFEST="${REPO_ROOT}/ihd/datasets/manifests/ihdataset_comprehensive.csv"

echo "[$(date -Iseconds)] Materializing comprehensive dataset files"
echo "Collection: ${COLLECTION}"
echo "Manifest: ${MANIFEST}"

uv run python -m ihd.datasets.data_utils.ihdataset.materialize_comprehensive_collection \
  --manifest "${MANIFEST}" \
  --collection "${COLLECTION}" \
  --disk-root /disk

echo "[$(date -Iseconds)] Materialization finished for ${COLLECTION}"
