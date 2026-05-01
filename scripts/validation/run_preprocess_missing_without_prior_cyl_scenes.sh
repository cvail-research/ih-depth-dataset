#!/bin/bash
#SBATCH --job-name=preprocess_missing_nocyl
#SBATCH --output=logs/out/%j_preprocess_missing_nocyl.out
#SBATCH --error=logs/err/%j_preprocess_missing_nocyl.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --partition=prod

set -euo pipefail

mkdir -p logs/out logs/err

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"

MISSING_LIST="analysis/annotation_workspace_nocyl/missing_without_prior_cyl_to_sync.tsv"
SPHERE="-0.109999,-0.001428,-0.155019,4.0"
PROFILE="projection_platform_sphere_r4p0_voxel0p03_sor50_2p0"
SUFFIX="platform_sphere_r4p0"

if [ ! -f "${MISSING_LIST}" ]; then
  echo "Missing scene list not found: ${MISSING_LIST}" >&2
  echo "Run scripts/data/run_sync_missing_without_prior_cyl_scenes.sh first." >&2
  exit 1
fi

TOTAL_SCENES="$(wc -l < "${MISSING_LIST}" | tr -d ' ')"
echo "[$(date -Iseconds)] Missing scenes to preprocess: ${TOTAL_SCENES}"
if [ "${TOTAL_SCENES}" -eq 0 ]; then
  echo "No missing without-prior-.cyl scenes found."
  exit 0
fi

INDEX=0
while IFS=$'\t' read -r collection path_name step_name _s3_step_dir; do
  INDEX=$((INDEX + 1))
  path_prefix="${path_name%%_DistStA}"
  path_key="${path_prefix,,}"
  step_num="${step_name##*_Step}"
  scene_label="${path_prefix} Step${step_num}"
  out_subdir="${path_key}_step${step_num}_${SUFFIX}"

  echo "[$(date -Iseconds)] (${INDEX}/${TOTAL_SCENES}) Preprocessing ${collection} ${path_name} ${step_name}"
  bash scripts/validation/run_preprocess_las_for_projection.sh \
    "${scene_label}" \
    "${collection}" \
    "${path_name}" \
    "${step_num}" \
    "${out_subdir}" \
    0.03 \
    50 \
    2.0 \
    "" \
    "" \
    1 \
    "" \
    "" \
    "${PROFILE}" \
    "" \
    "" \
    "" \
    0.0 \
    0.0 \
    "${SPHERE}" \
    ""
done < "${MISSING_LIST}"

echo "[$(date -Iseconds)] Missing without-prior-.cyl LiDAR preprocessing finished"
