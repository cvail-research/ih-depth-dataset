#!/bin/bash
#SBATCH --job-name=sync_ih_scene_bbvis_chunk
#SBATCH --output=logs/out/%j_sync_ih_scene_bbvis_chunk.out
#SBATCH --error=logs/err/%j_sync_ih_scene_bbvis_chunk.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=prod

set -euo pipefail

mkdir -p logs/out logs/err

if [ "$#" -ne 1 ] || [ ! -f "$1" ]; then
  echo "Usage: sbatch $0 <scene_list.csv>" >&2
  exit 1
fi

SCENE_LIST="$1"

echo "[$(date -Iseconds)] Syncing BBVIS for scene list: ${SCENE_LIST}"

python - <<'PY' "$SCENE_LIST"
import csv
import subprocess
import sys

scene_list = sys.argv[1]

with open(scene_list, newline='') as f:
    rows = list(csv.DictReader(f))

for r in rows:
    collection = r['collection'].strip()
    path_num = int(r['path_num'])
    step_num = int(r['step_num'])

    path_name = f'Path{path_num}_DistStA'
    scene_dir = f'Path{path_num}_Step{step_num}_DistStA'
    src = f's3://ihdataset-01/{collection}/{path_name}/{scene_dir}/'
    dst = f'/disk/{collection}/{path_name}/{scene_dir}/'

    print(f'SYNC {collection} path{path_num} step{step_num}')
    subprocess.run(['mkdir', '-p', dst], check=True)
    subprocess.run([
        'aws', 's3', 'sync',
        '--no-sign-request',
        '--no-progress',
        src, dst,
        '--exclude', '*',
        '--include', '*BBVIS*.hdr',
        '--include', '*BBVIS*.raw',
        '--include', '*BBVIS*.bsq',
        '--include', '*BBVis*.hdr',
        '--include', '*BBVis*.raw',
        '--include', '*BBVis*.bsq',
    ], check=False)

print('DONE')
PY

echo "[$(date -Iseconds)] Chunk sync finished"
