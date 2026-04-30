#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 <collection> [<path_filter_regex>]" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

COLLECTION="$1"
PATH_FILTER="${2:-.*}"
SUMMARY_ROOT="${REPO_ROOT}/analysis/lidar_labeling/${COLLECTION}"

if [ ! -d "${SUMMARY_ROOT}" ]; then
  echo "Collection summaries not found: ${SUMMARY_ROOT}" >&2
  exit 1
fi

find "${SUMMARY_ROOT}" -name summary.json | sort | while read -r summary_path; do
  rel_path="${summary_path#${SUMMARY_ROOT}/}"
  path_key="$(echo "${rel_path}" | cut -d/ -f1)"
  if ! echo "${path_key}" | grep -Eq "${PATH_FILTER}"; then
    continue
  fi

  step_dir="$(basename "$(dirname "${summary_path}")")"
  step_num="${step_dir##*_step}"
  path_capitalized="$(echo "${path_key}" | sed -E 's/^path([0-9]+)/Path\1_DistStA/')"
  scene_label="$(python3 - <<'PY' "${summary_path}"
import json, sys
with open(sys.argv[1], "r") as f:
    data = json.load(f)
print(data.get("scene_label", ""))
PY
)"
  if [ -z "${scene_label}" ]; then
    scene_label="${path_capitalized/_DistStA/} Step${step_num}"
  fi

  out_subdir="${step_dir}"
  echo "Submitting preprocessing for ${COLLECTION} ${path_capitalized} step ${step_num}"
  sbatch scripts/validation/run_preprocess_las_for_projection.sh \
    "${scene_label}" \
    "${COLLECTION}" \
    "${path_capitalized}" \
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
    projection_sor50_2p0_voxel0p03
done
