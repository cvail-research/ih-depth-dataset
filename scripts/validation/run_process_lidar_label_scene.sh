#!/bin/bash
#SBATCH --job-name=lidar_label_scene
#SBATCH --output=logs/out/%j_lidar_label_scene.out
#SBATCH --error=logs/err/%j_lidar_label_scene.err
#SBATCH --time=00:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=prod

set -euo pipefail

mkdir -p logs/out logs/err

if [ "$#" -lt 8 ] || [ "$#" -gt 9 ]; then
  echo "Usage: sbatch $0 <scene_label> <collection> <path_name> <step> <manual_csv> <annotation_minutes> <verdict> <out_subdir> [<chunk>]" >&2
  exit 1
fi

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"

SCRIPT_ARGS=("$@")
set -- "${SCRIPT_ARGS[@]}"

export PYTHONUNBUFFERED=1

SCENE_LABEL="$1"
COLLECTION="$2"
PATH_NAME="$3"
STEP="$4"
MANUAL_CSV="$5"
ANNOTATION_MINUTES="$6"
VERDICT="$7"
OUT_SUBDIR="$8"
CHUNK="${9:-2000000}"

PATH_PREFIX="${PATH_NAME%%_DistStA}"
PATH_KEY="${PATH_PREFIX,,}"
COLLECTION_TAG="$(echo "${COLLECTION}" | sed -E 's/^IHTest_([0-9]{6})_DistStA$/IHTest_\1/')"
OUT_DIR="${REPO_ROOT}/analysis/lidar_labeling/${COLLECTION}/${PATH_KEY}/${OUT_SUBDIR}"

mkdir -p "${OUT_DIR}"
cp "${MANUAL_CSV}" "${OUT_DIR}/manual_las_points.csv"

resolve_scene_dir() {
  local collection="$1"
  local path_name="$2"
  local path_prefix="$3"
  local step="$4"
  local candidate_diststa="/disk/${collection}/${path_name}/${path_prefix}_Step${step}_DistStA"
  local candidate_plain="/disk/${collection}/${path_name}/${path_prefix}_Step${step}"

  if [ -d "${candidate_diststa}" ]; then
    printf '%s\n' "${candidate_diststa}"
    return 0
  fi
  if [ -d "${candidate_plain}" ]; then
    printf '%s\n' "${candidate_plain}"
    return 0
  fi

  echo "Could not resolve scene directory for ${collection} ${path_name} step ${step}" >&2
  return 1
}

resolve_lwhsi_file() {
  local scene_dir="$1"
  local stem="$2"
  local ext="$3"
  local candidate_collect0="${scene_dir}/${stem}_LWHSI1_collect0_DistStA${ext}"
  local candidate_plain="${scene_dir}/${stem}_LWHSI1_DistStA${ext}"

  if [ -f "${candidate_collect0}" ]; then
    printf '%s\n' "${candidate_collect0}"
    return 0
  fi
  if [ -f "${candidate_plain}" ]; then
    printf '%s\n' "${candidate_plain}"
    return 0
  fi

  echo "Could not resolve LWHSI file for stem ${stem} and extension ${ext}" >&2
  return 1
}

SCENE_DIR="$(resolve_scene_dir "${COLLECTION}" "${PATH_NAME}" "${PATH_PREFIX}" "${STEP}")"
STEM="${COLLECTION_TAG}_${PATH_PREFIX}_Step${STEP}"
CORR="$(resolve_lwhsi_file "${SCENE_DIR}" "${STEM}" "_corresp.txt")"
CYL="$(resolve_lwhsi_file "${SCENE_DIR}" "${STEM}" ".cyl")"
HDR="$(resolve_lwhsi_file "${SCENE_DIR}" "${STEM}" ".hdr")"

srun uv run python ihd/datasets/fit_manual_rigid_and_compare_projection.py \
  --corresp "${CORR}" \
  --manual-las-csv "${OUT_DIR}/manual_las_points.csv" \
  --cyl "${CYL}" \
  --hsi-hdr "${HDR}" \
  --las "${SCENE_DIR}/${COLLECTION_TAG}_${PATH_PREFIX}_Step${STEP}_HiResLIDAR_DistStA.las" \
  --out-dir "${OUT_DIR}" \
  --scene-label "${SCENE_LABEL}" \
  --annotation-minutes "${ANNOTATION_MINUTES}" \
  --verdict "${VERDICT}" \
  --chunk "${CHUNK}"
