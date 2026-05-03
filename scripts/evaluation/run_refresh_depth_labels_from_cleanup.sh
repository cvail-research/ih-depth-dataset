#!/bin/bash
#SBATCH --job-name=ih_refresh_labels_cleanup
#SBATCH --output=logs/out/%j_refresh_labels_cleanup.out
#SBATCH --error=logs/err/%j_refresh_labels_cleanup.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=prod

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1
mkdir -p logs/out logs/err

DEPTH_LABEL_ROOT="${1:-analysis/depth_labels/platform_sphere_r4p0}"
TARGET_SCENE_MANIFEST="${2:-analysis/evaluation/scene_manifests/cleanup_refresh_include_scenes.csv}"
OCCLUSION_FILTER_RADIUS_PX="${3:-0}"
OCCLUSION_MIN_DEPTH_GAP_M="${4:-1.0}"
OCCLUSION_MIN_DEPTH_GAP_RATIO="${5:-0.05}"

export DEPTH_LABEL_ROOT
export TARGET_SCENE_MANIFEST
export OCCLUSION_FILTER_RADIUS_PX
export OCCLUSION_MIN_DEPTH_GAP_M
export OCCLUSION_MIN_DEPTH_GAP_RATIO

srun uv run python - <<'PY'
import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ihd.datasets.cylindrical_camera import read_cam
from ihd.datasets.depth_rasterization import rasterize, suppress_far_occlusion_bleed
from ihd.datasets.render_overlay_from_workspace import (
    load_gray,
    project_las,
    read_json,
    resolve_local_artifact,
)

repo_root = Path(".").resolve()
depth_label_root = repo_root / os.environ["DEPTH_LABEL_ROOT"]
target_scene_manifest = repo_root / os.environ["TARGET_SCENE_MANIFEST"]
occlusion_filter_radius_px = int(os.environ["OCCLUSION_FILTER_RADIUS_PX"])
occlusion_min_depth_gap_m = float(os.environ["OCCLUSION_MIN_DEPTH_GAP_M"])
occlusion_min_depth_gap_ratio = float(os.environ["OCCLUSION_MIN_DEPTH_GAP_RATIO"])

cleanup_rows = list(csv.DictReader((repo_root / "manifests/04_occlusion_cleanup_manifest_current.csv").open()))
frozen_rows = list(csv.DictReader((repo_root / "manifests/06_frozen_manifest_v0.csv").open()))
frozen_by_key = {(r["collection"], r["path"], r["step"]): r for r in frozen_rows}

selected = []
for row in cleanup_rows:
    raw_path = row.get("path_name") or row.get("path_key") or ""
    match = re.search(r"Path(\d+)_DistStA", raw_path, flags=re.IGNORECASE)
    if not match:
        continue
    path = f"path{int(match.group(1))}"
    step = f"{path}_step{int(row['step'])}"
    key = (row["collection"], path, step)
    frozen = frozen_by_key.get(key)
    if not frozen:
        continue
    if frozen.get("release_decision") != "include":
        continue
    cleaned_las = row.get("cleaned_las", "")
    if not cleaned_las:
        continue
    cleaned_las_path = Path(cleaned_las)
    if not cleaned_las_path.exists():
        continue
    selected.append(
        {
            "collection": row["collection"],
            "path": path,
            "step": step,
            "scene": frozen.get("scene", f"{row['collection']} / {path} / {step}"),
            "cleaned_las": cleaned_las_path,
        }
    )

selected.sort(key=lambda r: (r["collection"], r["path"], r["step"]))
if not selected:
    raise SystemExit("No include scenes found in cleanup manifest to refresh labels.")

# Write the exact scene subset used for label refresh and inference.
target_scene_manifest.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(
    [{"collection": r["collection"], "path": r["path"], "step": r["step"], "scene": r["scene"]} for r in selected]
).to_csv(target_scene_manifest, index=False)

written = []
for item in selected:
    workspace_dir = (
        repo_root / "analysis/annotation_workspace_nocyl" / item["collection"] / item["path"] / item["step"]
    )
    fit_path = workspace_dir / "fit.json"
    scene_json_path = workspace_dir / "scene.json"
    if not fit_path.exists() or not scene_json_path.exists():
        continue

    fit_data = read_json(fit_path)
    if not bool(fit_data.get("ready")):
        continue
    scene_data = read_json(scene_json_path)
    mode = str(fit_data.get("mode", ""))

    cyl_path = resolve_local_artifact(workspace_dir, fit_data, "fitted_cyl", "fitted.cyl")
    cam = read_cam(str(cyl_path))
    gray = load_gray(workspace_dir, scene_data)
    height, width = gray.shape

    i_vals, j_vals, d_vals = project_las(item["cleaned_las"], cam, fit_data, mode)
    inside = (i_vals >= 0) & (i_vals < width) & (j_vals >= 0) & (j_vals < height)
    i_vals = i_vals[inside]
    j_vals = j_vals[inside]
    d_vals = d_vals[inside]

    depth_img = rasterize(width, height, i_vals, j_vals, d_vals)
    depth_img = suppress_far_occlusion_bleed(
        depth_img,
        occlusion_filter_radius_px,
        occlusion_min_depth_gap_m,
        occlusion_min_depth_gap_ratio,
    )

    out_dir = depth_label_root / item["collection"] / item["path"] / item["step"]
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "projected_lidar_depth_label.npz"
    np.savez_compressed(
        npz_path,
        depth_m=depth_img.astype(np.float32),
        valid_mask=np.isfinite(depth_img).astype(np.uint8),
        projected_u=i_vals.astype(np.float32),
        projected_v=j_vals.astype(np.float32),
        projected_depth_m=d_vals.astype(np.float32),
        collection=np.asarray(item["collection"]),
        path_key=np.asarray(item["path"]),
        step_dir=np.asarray(item["step"]),
        workspace_dir=np.asarray(str(workspace_dir)),
        fit_path=np.asarray(str(fit_path)),
        las_path=np.asarray(str(item["cleaned_las"])),
        cyl_path=np.asarray(str(cyl_path)),
        preprocess_suffix=np.asarray("cleanup_workspace"),
        occlusion_filter_radius_px=np.asarray(occlusion_filter_radius_px),
        occlusion_min_depth_gap_m=np.asarray(occlusion_min_depth_gap_m),
        occlusion_min_depth_gap_ratio=np.asarray(occlusion_min_depth_gap_ratio),
    )
    json_path = out_dir / "projected_lidar_depth_label.json"
    json_path.write_text(
        json.dumps(
            {
                "collection": item["collection"],
                "path_key": item["path"],
                "step_dir": item["step"],
                "depth_npz": str(npz_path),
                "depth_key": "depth_m",
                "mask_key": "valid_mask",
                "units": "meters",
                "invalid_value": "NaN",
                "workspace_dir": str(workspace_dir),
                "fit_path": str(fit_path),
                "las_path": str(item["cleaned_las"]),
                "cyl_path": str(cyl_path),
                "preprocess_suffix": "cleanup_workspace",
                "occlusion_filter_radius_px": occlusion_filter_radius_px,
                "occlusion_min_depth_gap_m": occlusion_min_depth_gap_m,
                "occlusion_min_depth_gap_ratio": occlusion_min_depth_gap_ratio,
                "valid_pixels": int(np.isfinite(depth_img).sum()),
                "image_height": int(depth_img.shape[0]),
                "image_width": int(depth_img.shape[1]),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    written.append((item["collection"], item["path"], item["step"], str(npz_path)))

print(f"Refreshed labels: {len(written)} scenes")
for collection, path, step, npz in written:
    print(collection, path, step, npz)
print(f"Scene subset manifest: {target_scene_manifest}")
PY
