#!/bin/bash
#SBATCH --job-name=corr_guides_batch
#SBATCH --output=logs/out/%j_corr_guides_batch.out
#SBATCH --error=logs/err/%j_corr_guides_batch.err
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --partition=prod

set -euo pipefail

mkdir -p logs/out logs/err

if [ "$#" -lt 4 ]; then
  echo "Usage: sbatch $0 <collection> <path_name> <out_subdir> <step1> [<step2> ...]" >&2
  exit 1
fi

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"

SCRIPT_ARGS=("$@")
set -- "${SCRIPT_ARGS[@]}"

export PYTHONUNBUFFERED=1

COLLECTION="$1"
PATH_NAME="$2"
OUT_SUBDIR="$3"
shift 3

OUT_ROOT="${REPO_ROOT}/analysis/lidar_labeling/indexed_corresp_guides/${COLLECTION}/${OUT_SUBDIR}"
mkdir -p "${OUT_ROOT}"

PATH_PREFIX="${PATH_NAME%%_DistStA}"
COLLECTION_TAG="$(echo "${COLLECTION}" | sed -E 's/^IHTest_([0-9]{6})_DistStA$/IHTest_\1/')"

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

for STEP in "$@"; do
  SCENE_DIR="$(resolve_scene_dir "${COLLECTION}" "${PATH_NAME}" "${PATH_PREFIX}" "${STEP}")"
  STEM="${COLLECTION_TAG}_${PATH_PREFIX}_Step${STEP}"
  CORR="$(resolve_lwhsi_file "${SCENE_DIR}" "${STEM}" "_corresp.txt")"
  HDR="$(resolve_lwhsi_file "${SCENE_DIR}" "${STEM}" ".hdr")"
  OUT="${OUT_ROOT}/${PATH_PREFIX,,}_step${STEP}_indexed_corresp.png"

  echo "Generating indexed guide for ${PATH_PREFIX} Step${STEP}"
  srun --exclusive -N1 -n1 uv run python ihd/datasets/generate_indexed_corresp_guide.py \
    --corresp "${CORR}" \
    --hsi-hdr "${HDR}" \
    --out "${OUT}"
done
