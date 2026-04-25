#!/bin/bash
#SBATCH --job-name=dl_lwhsi_p5s1
#SBATCH --output=logs/out/%j_dl_lwhsi_p5s1.out
#SBATCH --error=logs/err/%j_dl_lwhsi_p5s1.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=prod

set -euo pipefail

mkdir -p logs/out logs/err

export PYTHONUNBUFFERED=1


srun uv run python ihd/datasets/data_utils/ihdataset/download_lwhsi_scene.py \
  --test 202104 --path 5 --step 1 \
  --dest /disk/raw \
  --contains LWHSI LWHSI1 \
  --ext .hdr .bsq
