#!/bin/bash
#SBATCH --job-name=sync_ih_scene
#SBATCH --output=logs/out/%j_sync_ih_scene.out
#SBATCH --error=logs/err/%j_sync_ih_scene.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=prod

set -euo pipefail

mkdir -p logs/out logs/err

if [ "$#" -ne 1 ] || [ ! -f "$1" ]; then
  echo "Usage: sbatch $0 <config.env>" >&2
  exit 1
fi

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"

set -a
# shellcheck disable=SC1090
source "$1"
set +a

: "${COLLECTION:?Missing COLLECTION}"
: "${PATH_NAME:?Missing PATH_NAME}"
: "${STEP:?Missing STEP}"

SCENE_DIR_NAME="${SCENE_DIR_NAME:-${PATH_NAME%%_DistStA}_Step${STEP}}"
SCENE_PREFIX="s3://ihdataset-01/${COLLECTION}/${PATH_NAME}/${SCENE_DIR_NAME}/"
SCENE_DST="/disk/${COLLECTION}/${PATH_NAME}/${SCENE_DIR_NAME}/"

mkdir -p "${SCENE_DST}"

echo "[$(date -Iseconds)] Syncing scene assets"
echo "Source: ${SCENE_PREFIX}"
echo "Destination: ${SCENE_DST}"

aws s3 sync \
  --no-sign-request \
  --no-progress \
  "${SCENE_PREFIX}" "${SCENE_DST}" \
  --exclude "*" \
  --include "*LWHSI1*.bsq" \
  --include "*LWHSI1*.hdr" \
  --include "*LWHSI1*.txt" \
  --include "*LWHSI1*.cyl" \
  --include "*HiResLIDAR*.las"

echo "[$(date -Iseconds)] Scene sync finished"
