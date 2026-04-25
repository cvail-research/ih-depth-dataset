#!/bin/bash
#SBATCH --job-name=las_preprocess
#SBATCH --output=logs/out/%j_las_preprocess.out
#SBATCH --error=logs/err/%j_las_preprocess.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --partition=prod

set -euo pipefail

mkdir -p logs/out logs/err

if { [ "$#" -ne 1 ] || [ ! -f "$1" ]; } && { [ "$#" -lt 5 ] || [ "$#" -gt 21 ]; }; then
  echo "Usage: sbatch $0 <config.env>" >&2
  echo "   config.env requires COLLECTION and PATH_NAME, plus either STEP or STEP_LIST" >&2
  echo "   or: sbatch $0 <scene_label> <collection> <path_name> <step> <out_subdir> [<projection_voxel>] [<sor_k>] [<sor_std_ratio>] [<range_max>] [<z_min>] [<projection_use_sor:0|1>] [<projection_sor_k>] [<projection_sor_std_ratio>] [<profile_name>] [<platform_radius>] [<platform_z_min>] [<platform_z_max>] [<platform_center_x>] [<platform_center_y>] [<exclude_spheres>] [<exclude_boxes>]" >&2
  echo "      exclude_spheres format: x,y,z,radius;x,y,z,radius;..." >&2
  echo "      exclude_boxes format: x_min,x_max,y_min,y_max,z_min,z_max;x_min,x_max,y_min,y_max,z_min,z_max;..." >&2
  exit 1
fi

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"

SCRIPT_ARGS=("$@")
set -- "${SCRIPT_ARGS[@]}"

export PYTHONUNBUFFERED=1

require_var() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "Missing required variable: ${name}" >&2
    exit 1
  fi
}

parse_steps() {
  local raw="$1"
  raw="${raw//,/ }"
  # shellcheck disable=SC2206
  local parsed=(${raw})
  printf '%s\n' "${parsed[@]}"
}

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

resolve_las_path() {
  local scene_dir="$1"
  local path_prefix="$2"
  local step="$3"
  local -a candidates=(
    "${scene_dir}"/*"${path_prefix}_Step${step}"*HiResLIDAR*.las
    "${scene_dir}"/*HiResLIDAR*.las
  )
  local path
  for path in "${candidates[@]}"; do
    if [ -f "${path}" ]; then
      printf '%s\n' "${path}"
      return 0
    fi
  done

  echo "Could not resolve HiResLIDAR LAS in ${scene_dir}" >&2
  return 1
}

if [ "$#" -eq 1 ] && [ -f "$1" ]; then
  CONFIG_PATH="$1"
  set -a
  # shellcheck disable=SC1090
  source "${CONFIG_PATH}"
  set +a

  require_var COLLECTION
  require_var PATH_NAME
  if [ -z "${STEP:-}" ] && [ -z "${STEP_LIST:-}" ]; then
    echo "Missing required variable: STEP or STEP_LIST" >&2
    exit 1
  fi

  PROJECTION_VOXEL="${PROJECTION_VOXEL:-0.03}"
  SOR_K="${SOR_K:-50}"
  SOR_STD_RATIO="${SOR_STD_RATIO:-2.0}"
  RANGE_MAX="${RANGE_MAX:-}"
  Z_MIN="${Z_MIN:-}"
  PROJECTION_USE_SOR="${PROJECTION_USE_SOR:-1}"
  PROJECTION_SOR_K="${PROJECTION_SOR_K:-}"
  PROJECTION_SOR_STD_RATIO="${PROJECTION_SOR_STD_RATIO:-}"
  PROFILE_NAME="${PROFILE_NAME:-projection_sor50_2p0_voxel0p03}"
  PLATFORM_RADIUS="${PLATFORM_RADIUS:-}"
  PLATFORM_Z_MIN="${PLATFORM_Z_MIN:-}"
  PLATFORM_Z_MAX="${PLATFORM_Z_MAX:-}"
  PLATFORM_CENTER_X="${PLATFORM_CENTER_X:-0.0}"
  PLATFORM_CENTER_Y="${PLATFORM_CENTER_Y:-0.0}"
  EXCLUDE_SPHERES="${EXCLUDE_SPHERES:-}"
  EXCLUDE_BOXES="${EXCLUDE_BOXES:-}"
  if [ -n "${STEP_LIST:-}" ]; then
    mapfile -t STEPS < <(parse_steps "${STEP_LIST}")
  else
    STEPS=("${STEP}")
  fi
else
  SCENE_LABEL="$1"
  COLLECTION="$2"
  PATH_NAME="$3"
  STEP="$4"
  OUT_SUBDIR="$5"
  PROJECTION_VOXEL="${6:-0.03}"
  SOR_K="${7:-50}"
  SOR_STD_RATIO="${8:-2.0}"
  RANGE_MAX="${9:-}"
  Z_MIN="${10:-}"
  PROJECTION_USE_SOR="${11:-1}"
  PROJECTION_SOR_K="${12:-}"
  PROJECTION_SOR_STD_RATIO="${13:-}"
  PROFILE_NAME="${14:-projection_sor50_2p0_voxel0p03}"
  PLATFORM_RADIUS="${15:-}"
  PLATFORM_Z_MIN="${16:-}"
  PLATFORM_Z_MAX="${17:-}"
  PLATFORM_CENTER_X="${18:-0.0}"
  PLATFORM_CENTER_Y="${19:-0.0}"
  EXCLUDE_SPHERES="${20:-}"
  EXCLUDE_BOXES="${21:-}"
  STEPS=("${STEP}")
fi

PATH_PREFIX="${PATH_NAME%%_DistStA}"
PATH_KEY="${PATH_PREFIX,,}"

run_one_step() {
  local step="$1"
  local scene_dir las_path scene_label out_subdir out_dir
  scene_label="${SCENE_LABEL:-${PATH_PREFIX} Step${step}}"
  if [ "${#STEPS[@]}" -eq 1 ] && [ -n "${OUT_SUBDIR:-}" ]; then
    out_subdir="${OUT_SUBDIR}"
  else
    out_subdir="${PATH_KEY}_step${step}"
  fi
  out_dir="${REPO_ROOT}/analysis/lidar_preprocessing/${COLLECTION}/${PATH_KEY}/${out_subdir}"

  if compgen -G "${out_dir}/*_projection_clean.las" > /dev/null; then
    echo "[$(date -Iseconds)] Skipping ${PATH_PREFIX} Step${step}: projection LAS already exists"
    return 0
  fi

  scene_dir="$(resolve_scene_dir "${COLLECTION}" "${PATH_NAME}" "${PATH_PREFIX}" "${step}")"
  las_path="$(resolve_las_path "${scene_dir}" "${PATH_PREFIX}" "${step}")"

  CMD=(
    uv run python ihd/datasets/preprocess_las_for_annotation.py
    --las "${las_path}"
    --out-dir "${out_dir}"
    --scene-label "${scene_label}"
    --profile-name "${PROFILE_NAME}"
    --projection-voxel "${PROJECTION_VOXEL}"
    --sor-k "${SOR_K}"
    --sor-std-ratio "${SOR_STD_RATIO}"
  )

  if [ -n "${RANGE_MAX}" ]; then
    CMD+=(--range-max "${RANGE_MAX}")
  fi
  if [ -n "${Z_MIN}" ]; then
    CMD+=(--z-min "${Z_MIN}")
  fi
  if [ "${PROJECTION_USE_SOR}" = "1" ]; then
    CMD+=(--projection-use-sor)
  fi
  if [ -n "${PROJECTION_SOR_K}" ]; then
    CMD+=(--projection-sor-k "${PROJECTION_SOR_K}")
  fi
  if [ -n "${PROJECTION_SOR_STD_RATIO}" ]; then
    CMD+=(--projection-sor-std-ratio "${PROJECTION_SOR_STD_RATIO}")
  fi
  if [ -n "${PLATFORM_RADIUS}" ]; then
    CMD+=(--platform-radius "${PLATFORM_RADIUS}")
  fi
  if [ -n "${PLATFORM_Z_MIN}" ]; then
    CMD+=(--platform-z-min "${PLATFORM_Z_MIN}")
  fi
  if [ -n "${PLATFORM_Z_MAX}" ]; then
    CMD+=(--platform-z-max "${PLATFORM_Z_MAX}")
  fi
  CMD+=(--platform-center-x "${PLATFORM_CENTER_X}")
  CMD+=(--platform-center-y "${PLATFORM_CENTER_Y}")
  if [ -n "${EXCLUDE_SPHERES}" ]; then
    IFS=';' read -r -a sphere_specs <<< "${EXCLUDE_SPHERES}"
    for sphere_spec in "${sphere_specs[@]}"; do
      if [ -n "${sphere_spec}" ]; then
        CMD+=("--exclude-sphere=${sphere_spec}")
      fi
    done
  fi
  if [ -n "${EXCLUDE_BOXES}" ]; then
    IFS=';' read -r -a box_specs <<< "${EXCLUDE_BOXES}"
    for box_spec in "${box_specs[@]}"; do
      if [ -n "${box_spec}" ]; then
        CMD+=("--exclude-box=${box_spec}")
      fi
    done
  fi

  echo "[$(date -Iseconds)] Preprocessing ${PATH_PREFIX} Step${step}"
  "${CMD[@]}"
}

for step in "${STEPS[@]}"; do
  run_one_step "${step}"
done
