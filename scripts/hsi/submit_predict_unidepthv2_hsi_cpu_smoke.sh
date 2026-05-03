#!/bin/bash
#SBATCH --job-name=ih_hsi_udv2_cpu
#SBATCH --output=logs/out/%j_ih_hsi_udv2_cpu.out
#SBATCH --error=logs/err/%j_ih_hsi_udv2_cpu.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=prod

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs/out logs/err

HDR_PATH="${1:-/disk/IHTest_202009_DistStA/Path3_DistStA/Path3_Step4_DistStA/IHTest_202009_Path3_Step4_LWHSI1__DistStA.hdr}"
OUT_ROOT="${2:-analysis/evaluation/unidepthv2_hsi_cpu_smoke}"
LABEL_PATH="${3:-analysis/depth_labels/platform_sphere_r2p5/IHTest_202009_DistStA/path3/path3_step4/projected_lidar_depth_label.npz}"

scripts/hsi/run_predict_unidepthv2_hsi.sh \
  --hdr "${HDR_PATH}" \
  --label-path "${LABEL_PATH}" \
  --out-dir "${OUT_ROOT}" \
  --device cpu \
  --no-vis
