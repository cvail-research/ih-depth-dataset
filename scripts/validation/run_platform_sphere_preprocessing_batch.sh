#!/bin/bash
#SBATCH --job-name=las_preprocess_batch
#SBATCH --output=logs/out/%j_las_preprocess_batch.out
#SBATCH --error=logs/err/%j_las_preprocess_batch.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --partition=prod

set -euo pipefail

if [ "$#" -gt 2 ]; then
  echo "Usage: sbatch $0 [<collection_filter_regex>] [<path_filter_regex>]" >&2
  exit 1
fi

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"

mkdir -p logs/out logs/err analysis/lidar_preprocessing

COLLECTION_FILTER="${1:-.*}"
PATH_FILTER="${2:-.*}"
SPHERE="-0.109999,-0.001428,-0.155019,2.5"
PROFILE="projection_platform_sphere_r2p5_voxel0p03_sor50_2p0"
SUFFIX="platform_sphere_r2p5"
SCENE_MANIFEST="analysis/lidar_preprocessing/${SUFFIX}_batch_scenes.tsv"

export PYTHONUNBUFFERED=1

echo "[$(date -Iseconds)] Building scene list"
uv run python - <<'PY' "${COLLECTION_FILTER}" "${PATH_FILTER}" "${SUFFIX}" > "${SCENE_MANIFEST}"
import re
import sys
from pathlib import Path

from ihd.qc_review.scene_service import ANALYSIS_ROOT, QC_ROOT, discover_qc_scenes

collection_re = re.compile(sys.argv[1])
path_re = re.compile(sys.argv[2])
suffix = sys.argv[3]

scenes = discover_qc_scenes(
    results_root=ANALYSIS_ROOT / "lidar_labeling",
    data_root=Path("/disk"),
    cache_root=QC_ROOT / "cache",
)

for scene in scenes:
    if not collection_re.search(scene.collection):
        continue
    if not path_re.search(scene.path_key):
        continue

    step_match = re.search(r"_step(\d+)$", scene.step_dir)
    path_num_match = re.fullmatch(r"path(\d+)", scene.path_key)
    if not step_match or not path_num_match:
        print(
            f"Skipping malformed scene key: {scene.collection}/{scene.path_key}/{scene.step_dir}",
            file=sys.stderr,
        )
        continue

    path_num = int(path_num_match.group(1))
    step_num = int(step_match.group(1))
    path_name = f"Path{path_num}_DistStA"
    scene_label = f"Path{path_num} Step{step_num}"
    out_subdir = f"{scene.step_dir}_{suffix}"
    print("\t".join([scene_label, scene.collection, path_name, str(step_num), out_subdir]))
PY

TOTAL_SCENES="$(wc -l < "${SCENE_MANIFEST}" | tr -d ' ')"
echo "[$(date -Iseconds)] Scenes to preprocess: ${TOTAL_SCENES}"
if [ "${TOTAL_SCENES}" -eq 0 ]; then
  echo "No scenes matched filters: collection=${COLLECTION_FILTER}, path=${PATH_FILTER}" >&2
  exit 1
fi

INDEX=0
while IFS=$'\t' read -r scene_label collection path_name step out_subdir; do
  INDEX=$((INDEX + 1))
  echo "[$(date -Iseconds)] (${INDEX}/${TOTAL_SCENES}) ${collection} ${path_name} Step${step}"
  bash scripts/validation/run_preprocess_las_for_annotation.sh \
    "${scene_label}" \
    "${collection}" \
    "${path_name}" \
    "${step}" \
    "${out_subdir}" \
    0.03 \
    50 \
    2.0 \
    "" \
    "" \
    1 \
    50 \
    2.0 \
    "${PROFILE}" \
    "" \
    "" \
    "" \
    0.0 \
    0.0 \
    "${SPHERE}" \
    ""
done < "${SCENE_MANIFEST}"

echo "[$(date -Iseconds)] Finished platform-sphere preprocessing batch"
