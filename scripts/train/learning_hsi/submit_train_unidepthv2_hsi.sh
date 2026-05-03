#!/bin/bash
#SBATCH --job-name=ih_train_hsi_unidepthv2
#SBATCH --output=logs/out/%j_ih_train_hsi_unidepthv2.out
#SBATCH --error=logs/err/%j_ih_train_hsi_unidepthv2.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs/out logs/err

TRAIN_MANIFEST="${1:?Usage: sbatch $0 <train_manifest.csv> <val_manifest.csv> [out_dir]}"
VAL_MANIFEST="${2:?Usage: sbatch $0 <train_manifest.csv> <val_manifest.csv> [out_dir]}"
OUT_DIR="${3:-analysis/training/unidepthv2_hsi/$(date +%Y%m%d_%H%M%S)}"

scripts/train/learning_hsi/run_train_unidepthv2_hsi.sh \
  --train-manifest "${TRAIN_MANIFEST}" \
  --val-manifest "${VAL_MANIFEST}" \
  --out-dir "${OUT_DIR}" \
  --device cuda \
  --wandb-entity "${WANDB_ENTITY:-ai-uis}" \
  --wandb-project "${WANDB_PROJECT:-ih-depth}" \
  "${@:4}"
