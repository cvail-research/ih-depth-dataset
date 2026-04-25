#!/bin/bash
#SBATCH --job-name=refresh_ih_manifests
#SBATCH --output=logs/out/%j_refresh_ih_manifests.out
#SBATCH --error=logs/err/%j_refresh_ih_manifests.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=prod

set -euo pipefail

mkdir -p logs/out logs/err

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"

RAW_MANIFEST="${REPO_ROOT}/ihd/datasets/manifests/ihdataset_relevant_s3.csv"
STRICT_MANIFEST="${REPO_ROOT}/ihd/datasets/manifests/ihdataset_strict.csv"
COMPREHENSIVE_MANIFEST="${REPO_ROOT}/ihd/datasets/manifests/ihdataset_comprehensive.csv"

uv run python ihd/datasets/data_utils/ihdataset/create_relevant_s3_manifest.py \
  --bucket ihdataset-01 \
  --out "${RAW_MANIFEST}"

uv run python ihd/datasets/data_utils/ihdataset/create_lwhsi_lidar_manifest.py \
  --manifest "${RAW_MANIFEST}" \
  --strict "${STRICT_MANIFEST}" \
  --comprehensive "${COMPREHENSIVE_MANIFEST}"

python3 - <<'PY'
import pandas as pd
from pathlib import Path

strict_path = Path("ihd/datasets/manifests/ihdataset_strict.csv")
comp_path = Path("ihd/datasets/manifests/ihdataset_comprehensive.csv")

strict = pd.read_csv(strict_path)
comp = pd.read_csv(comp_path)

def scene_ids(df: pd.DataFrame) -> set[tuple[str, str, str]]:
    return set(zip(df["collect"], df["path"], df["step"]))

strict_scenes = scene_ids(strict)
comp_scenes = scene_ids(comp)
no_cyl_scenes = strict_scenes - comp_scenes

print("\n--- Current S3 Inventory Summary ---")
print(f"Strict scenes (HiResLIDAR + basic LWHSI): {len(strict_scenes)}")
print(f"Comprehensive scenes (HiResLIDAR + LWHSI + .cyl + .txt): {len(comp_scenes)}")
print(f"No-.cyl candidate scenes: {len(no_cyl_scenes)}")

strict_by_collection = (
    strict[["collect", "path", "step"]]
    .drop_duplicates()
    .groupby("collect")
    .size()
    .sort_index()
)
comp_by_collection = (
    comp[["collect", "path", "step"]]
    .drop_duplicates()
    .groupby("collect")
    .size()
    .sort_index()
)

print("\nStrict by collection:")
for collect, count in strict_by_collection.items():
    print(f"  {collect}: {count}")

print("\nComprehensive by collection:")
for collect, count in comp_by_collection.items():
    print(f"  {collect}: {count}")

sample = sorted(no_cyl_scenes)[:25]
print("\nFirst no-.cyl candidates:")
for collect, path, step in sample:
    print(f"  {collect} | {path} | {step}")
PY
