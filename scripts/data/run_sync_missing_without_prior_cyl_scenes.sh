#!/bin/bash
#SBATCH --job-name=sync_missing_nocyl
#SBATCH --output=logs/out/%j_sync_missing_nocyl.out
#SBATCH --error=logs/err/%j_sync_missing_nocyl.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=prod

set -euo pipefail

mkdir -p logs/out logs/err

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"

MISSING_LIST="analysis/annotation_workspace_nocyl/missing_without_prior_cyl_to_sync.tsv"

echo "[$(date -Iseconds)] Building missing without-prior-.cyl scene list"
uv run python - <<'PY' > "${MISSING_LIST}"
from pathlib import Path
import json
import pandas as pd

pool = pd.read_csv("manifests/02_without_prior_cyl_n246.csv")
raw = pd.read_csv("ihd/datasets/manifests/ihdataset_relevant_s3.csv")
workspace = set()

for scene_json in Path("analysis/annotation_workspace_nocyl").rglob("scene.json"):
    data = json.loads(scene_json.read_text())
    if data.get("capabilities", {}).get("migrated_from_lidar_labeling"):
        continue
    parts = scene_json.parts
    idx = parts.index("annotation_workspace_nocyl")
    workspace.add((parts[idx + 1], parts[idx + 2], parts[idx + 3]))

for row in pool.sort_values(["collection", "path", "step"]).itertuples(index=False):
    key = (row.collection, row.path, row.step)
    if key not in workspace:
        scene_rows = raw[
            (raw["collect"] == row.collection)
            & (raw["path"] == row.path_name)
            & (raw["step"].astype(str).str.replace("_DistStA", "", regex=False) == row.step_name)
        ]
        if scene_rows.empty:
            raise SystemExit(f"No raw S3 manifest rows for {row.scene_id}")
        s3_key = str(scene_rows.iloc[0]["s3_key"])
        parts = s3_key.split("/")
        s3_step_dir = parts[2]
        print("\t".join([row.collection, row.path_name, row.step_name, s3_step_dir]))
PY

TOTAL_SCENES="$(wc -l < "${MISSING_LIST}" | tr -d ' ')"
echo "[$(date -Iseconds)] Missing scenes to sync: ${TOTAL_SCENES}"
if [ "${TOTAL_SCENES}" -eq 0 ]; then
  echo "No missing without-prior-.cyl scenes found."
  exit 0
fi

INDEX=0
while IFS=$'\t' read -r collection path_name step_name s3_step_dir; do
  INDEX=$((INDEX + 1))
  src_step="${s3_step_dir}"
  dst_step="${s3_step_dir}"
  src="s3://ihdataset-01/${collection}/${path_name}/${src_step}/"
  dst="/disk/${collection}/${path_name}/${dst_step}/"

  echo "[$(date -Iseconds)] (${INDEX}/${TOTAL_SCENES}) Syncing ${collection}/${path_name}/${step_name}"
  mkdir -p "${dst}"
  aws s3 sync \
    --no-sign-request \
    --no-progress \
    "${src}" "${dst}" \
    --exclude "*" \
    --include "*LWHSI1*.bsq" \
    --include "*LWHSI1*.hdr" \
    --include "*LWHSI1*.txt" \
    --include "*LWHSI1*.cyl" \
    --include "*HiResLIDAR*.las"
done < "${MISSING_LIST}"

echo "[$(date -Iseconds)] Missing without-prior-.cyl scene sync finished"
