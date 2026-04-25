#!/bin/bash
#SBATCH --job-name=sync_ih_202104_lwhsi_lidar
#SBATCH --output=logs/out/%j_sync_ih_202104_lwhsi_lidar.out
#SBATCH --error=logs/err/%j_sync_ih_202104_lwhsi_lidar.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=prod

set -euo pipefail

mkdir -p logs/out logs/err

SRC="s3://ihdataset-01/IHTest_202104_DistStA/"
DST="/disk/IHTest_202104_DistStA/"

echo "[$(date -Iseconds)] Syncing filtered DARPA Invisible Headlights assets"
echo "Source: ${SRC}"
echo "Destination: ${DST}"

mkdir -p "${DST}"

aws s3 sync \
  --no-sign-request \
  --no-progress \
  "${SRC}" "${DST}" \
  --exclude "*" \
  --include "*LWHSI1*.bsq" \
  --include "*LWHSI1*.hdr" \
  --include "*LWHSI1*.txt" \
  --include "*LWHSI1*.cyl" \
  --include "*HiResLIDAR*.las"

echo "[$(date -Iseconds)] Sync finished"
