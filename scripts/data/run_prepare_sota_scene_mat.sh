#!/bin/bash
#SBATCH --job-name=prep_sota_mat
#SBATCH --output=logs/out/%j_prep_sota_mat.out
#SBATCH --error=logs/err/%j_prep_sota_mat.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --partition=prod

set -euo pipefail

mkdir -p logs/out logs/err

export PYTHONUNBUFFERED=1

HDR_PATH="/disk/raw/IHTest_202104_DistStA/Path5_DistStA/Path5_Step1_DistStA/IHTest_202104_Path5_Step1_LWHSI1_collect0_DistStA.hdr"

OUT_MAT="third_party/sota_ozone/scenes/Path5Step1_scene.mat"

srun uv run python ihd/datasets/data_utils/ihdataset/prepare_sota_scene_mat.py \
  --hdr-path "${HDR_PATH}" \
  --sota-lambda-mat third_party/sota_ozone/data/lambda.mat \
  --out-mat "${OUT_MAT}" \
  --t-air 289.7
