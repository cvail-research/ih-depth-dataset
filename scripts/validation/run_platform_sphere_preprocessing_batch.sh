#!/bin/bash
#SBATCH --job-name=las_preprocess_r4p0
#SBATCH --output=logs/out/%j_las_preprocess_r4p0.out
#SBATCH --error=logs/err/%j_las_preprocess_r4p0.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --partition=prod

set -euo pipefail

if [ "$#" -gt 3 ]; then
  echo "Usage: sbatch $0 [<collection_filter_regex>] [<path_filter_regex>] [<metadata_csv>]" >&2
  exit 1
fi

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"

mkdir -p logs/out logs/err analysis/lidar_preprocessing

COLLECTION_FILTER="${1:-.*}"
PATH_FILTER="${2:-.*}"
METADATA_CSV="${3:-manifests/02_las_preprocessing_metadata_n4.csv}"
PROFILE="projection_platform_sphere_r4p0_voxel0p03_sor50_2p0"
SUFFIX="platform_sphere_r4p0"
SCENE_MANIFEST="analysis/lidar_preprocessing/${SUFFIX}_batch_scenes.tsv"

export PYTHONUNBUFFERED=1

echo "[$(date -Iseconds)] Building scene list"
uv run python - <<'PY' "${COLLECTION_FILTER}" "${PATH_FILTER}" "${SUFFIX}" "${METADATA_CSV}" > "${SCENE_MANIFEST}"
import csv
import re
import sys
from pathlib import Path

from ihd.qc_review.scene_service import ANALYSIS_ROOT, QC_ROOT, discover_qc_scenes

collection_re = re.compile(sys.argv[1])
path_re = re.compile(sys.argv[2])
suffix = sys.argv[3]
metadata_csv = Path(sys.argv[4])

metadata_rows = []
with metadata_csv.open("r", newline="") as f:
    for row in csv.DictReader(f):
        metadata_rows.append(row)


def metadata_for(collection: str, path_name: str) -> dict | None:
    for row in metadata_rows:
        if (
            row.get("collection") == collection
            and row.get("scope") == "path"
            and row.get("path_name") == path_name
        ):
            return row
    for row in metadata_rows:
        if row.get("collection") == collection and row.get("scope") == "collection":
            return row
    return None

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
    metadata = metadata_for(scene.collection, path_name)
    if metadata is None:
        print(
            f"Skipping {scene.collection}/{path_name}/{scene.step_dir}: no platform metadata row",
            file=sys.stderr,
        )
        continue
    scene_label = f"Path{path_num} Step{step_num}"
    out_subdir = f"{scene.step_dir}_{suffix}"
    sphere = ",".join(
        [
            metadata["platform_center_x"],
            metadata["platform_center_y"],
            metadata["platform_center_z"],
            metadata["platform_sphere_radius_m"],
        ]
    )
    print(
        "\t".join(
            [
                scene_label,
                scene.collection,
                path_name,
                str(step_num),
                out_subdir,
                metadata["projection_voxel_m"],
                metadata["sor_k"],
                metadata["sor_std_ratio"],
                sphere,
            ]
        )
    )
PY

TOTAL_SCENES="$(wc -l < "${SCENE_MANIFEST}" | tr -d ' ')"
echo "[$(date -Iseconds)] Scenes to preprocess: ${TOTAL_SCENES}"
if [ "${TOTAL_SCENES}" -eq 0 ]; then
  echo "No scenes matched filters: collection=${COLLECTION_FILTER}, path=${PATH_FILTER}" >&2
  exit 1
fi

INDEX=0
while IFS=$'\t' read -r scene_label collection path_name step out_subdir projection_voxel sor_k sor_std_ratio sphere; do
  INDEX=$((INDEX + 1))
  echo "[$(date -Iseconds)] (${INDEX}/${TOTAL_SCENES}) ${collection} ${path_name} Step${step}"
  bash scripts/validation/run_preprocess_las_for_projection.sh \
    "${scene_label}" \
    "${collection}" \
    "${path_name}" \
    "${step}" \
    "${out_subdir}" \
    "${projection_voxel}" \
    "${sor_k}" \
    "${sor_std_ratio}" \
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
    "${sphere}" \
    ""
done < "${SCENE_MANIFEST}"

echo "[$(date -Iseconds)] Finished platform-sphere preprocessing batch (${SUFFIX})"
