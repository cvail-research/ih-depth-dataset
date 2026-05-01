#!/bin/bash
#SBATCH --job-name=verify_platform_spheres
#SBATCH --output=logs/out/%j_verify_platform_spheres.out
#SBATCH --error=logs/err/%j_verify_platform_spheres.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --partition=prod

set -euo pipefail

mkdir -p logs/out logs/err analysis/lidar_preprocessing

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1

SCENE_MANIFEST="${1:-manifests/05_scene_quality_manifest_current.csv}"
METADATA_CSV="${2:-manifests/04_las_preprocessing_metadata_n4.csv}"
OUT_CSV="${3:-analysis/lidar_preprocessing/platform_sphere_r4p0_metadata_verification.csv}"

srun uv run python - <<'PY' "${SCENE_MANIFEST}" "${METADATA_CSV}" "${OUT_CSV}"
import csv
import math
import re
import sys
from pathlib import Path

import laspy
import numpy as np


scene_manifest = Path(sys.argv[1])
metadata_csv = Path(sys.argv[2])
out_csv = Path(sys.argv[3])
data_root = Path("/disk")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def path_name_from_key(path_key: str) -> str:
    match = re.fullmatch(r"path(\d+)", path_key)
    if not match:
        return path_key
    return f"Path{int(match.group(1))}_DistStA"


def step_number(step_dir: str) -> int | None:
    match = re.search(r"_step(\d+)$", step_dir)
    return int(match.group(1)) if match else None


def metadata_for(rows: list[dict[str, str]], collection: str, path_name: str) -> dict[str, str] | None:
    for row in rows:
        if (
            row.get("collection") == collection
            and row.get("scope") == "path"
            and row.get("path_name") == path_name
        ):
            return row
    for row in rows:
        if row.get("collection") == collection and row.get("scope") == "collection":
            return row
    return None


def resolve_scene_dir(collection: str, path_name: str, step: int) -> Path | None:
    prefix = path_name.replace("_DistStA", "")
    candidates = (
        data_root / collection / path_name / f"{prefix}_Step{step}_DistStA",
        data_root / collection / path_name / f"{prefix}_Step{step}",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def resolve_las(scene_dir: Path, path_name: str, step: int) -> Path | None:
    prefix = path_name.replace("_DistStA", "")
    candidates = sorted(scene_dir.glob(f"*{prefix}_Step{step}*HiResLIDAR*.las"))
    if not candidates:
        candidates = sorted(scene_dir.glob("*HiResLIDAR*.las"))
    return candidates[0] if candidates else None


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


metadata_rows = read_csv(metadata_csv)
scene_rows = read_csv(scene_manifest)

representatives: dict[tuple[str, str], dict[str, str]] = {}
for row in scene_rows:
    if row.get("fit_ready") != "True":
        continue
    step = step_number(row.get("step", ""))
    if step is None:
        continue
    key = (row["collection"], row["path"])
    current = representatives.get(key)
    if current is None or step < step_number(current["step"]):
        representatives[key] = row

out_rows = []
for (collection, path_key), scene in sorted(representatives.items()):
    path_name = path_name_from_key(path_key)
    step = step_number(scene["step"])
    assert step is not None
    metadata = metadata_for(metadata_rows, collection, path_name)
    row = {
        "collection": collection,
        "path": path_key,
        "path_name": path_name,
        "representative_step": scene["step"],
        "metadata_scope": "",
        "platform_center_x": "",
        "platform_center_y": "",
        "platform_center_z": "",
        "platform_sphere_radius_m": "",
        "raw_las": "",
        "raw_points": "",
        "points_inside_sphere": "",
        "fraction_inside_sphere": "",
        "sphere_affects_cloud": "",
        "status": "",
    }
    if metadata is None:
        row["status"] = "no_platform_metadata"
        out_rows.append(row)
        continue

    row["metadata_scope"] = metadata.get("scope", "")
    cx = as_float(metadata.get("platform_center_x", ""))
    cy = as_float(metadata.get("platform_center_y", ""))
    cz = as_float(metadata.get("platform_center_z", ""))
    radius = as_float(metadata.get("platform_sphere_radius_m", ""))
    row["platform_center_x"] = cx
    row["platform_center_y"] = cy
    row["platform_center_z"] = cz
    row["platform_sphere_radius_m"] = radius

    scene_dir = resolve_scene_dir(collection, path_name, step)
    if scene_dir is None:
        row["status"] = "missing_scene_dir"
        out_rows.append(row)
        continue
    las_path = resolve_las(scene_dir, path_name, step)
    if las_path is None:
        row["status"] = "missing_las"
        out_rows.append(row)
        continue

    las = laspy.read(las_path)
    xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float32, copy=False)
    center = np.array([cx, cy, cz], dtype=np.float32)
    diff = xyz - center.reshape(1, 3)
    inside = np.einsum("ij,ij->i", diff, diff) <= (radius * radius)
    inside_count = int(inside.sum())
    raw_count = int(xyz.shape[0])

    row["raw_las"] = str(las_path)
    row["raw_points"] = raw_count
    row["points_inside_sphere"] = inside_count
    row["fraction_inside_sphere"] = inside_count / max(1, raw_count)
    row["sphere_affects_cloud"] = inside_count > 0
    row["status"] = "sphere_removes_points" if inside_count > 0 else "sphere_removes_zero_points"
    out_rows.append(row)

out_csv.parent.mkdir(parents=True, exist_ok=True)
fieldnames = list(out_rows[0].keys()) if out_rows else []
with out_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_rows)

print(f"Wrote {len(out_rows)} representative path checks to {out_csv}")
PY
