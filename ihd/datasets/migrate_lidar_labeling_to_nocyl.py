import argparse
import csv
import json
from pathlib import Path
from typing import Any

import laspy
import numpy as np

from ihd.annotation_workspace.scene_service import (
    PREPROCESS_ROOT,
    _now,
    depth_range,
    load_gray_preview,
    metrics_from_residual,
    path_key,
    rasterize,
    resolve_lwhsi_file,
    resolve_scene_dir,
    save_manual_projection_plot,
    save_overlay,
    scene_key,
)
from ihd.annotation_workspace_nocyl.scene_service import WORKSPACE_ROOT_NO_CYL, build_default_cyl_camera
from ihd.datasets.calibration_lidar_cylindrical import calibrate_single, write_cyl
from ihd.datasets.cylindrical_camera import project_vect_safe


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Migrate old lidar_labeling scenes into annotation_workspace_nocyl by "
            "generating our own correspondence .txt and fitting our own .cyl."
        )
    )
    ap.add_argument("--results-root", default="analysis/lidar_labeling")
    ap.add_argument("--workspace-root", default=str(WORKSPACE_ROOT_NO_CYL))
    ap.add_argument("--preprocess-suffix", default="platform_sphere_r2p5")
    ap.add_argument("--collection-filter", default="")
    ap.add_argument("--path-filter", default="")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--manifest-out", default="analysis/qc_review/migrated_lidar_labeling_to_nocyl_manifest.csv")
    return ap.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def parse_scene(summary_path: Path, results_root: Path) -> tuple[str, str, str, str, int]:
    rel = summary_path.relative_to(results_root)
    collection, path_key_value, step_dir = rel.parts[:3]
    step = int(step_dir.rsplit("_step", 1)[1])
    path_name = f"Path{int(path_key_value.replace('path', ''))}_DistStA"
    return collection, path_key_value, step_dir, path_name, step


def read_manual_las_points(path: Path) -> dict[int, np.ndarray]:
    points: dict[int, np.ndarray] = {}
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            idx = int(float(row[0]))
            points[idx] = np.asarray([float(row[1]), float(row[2]), float(row[3])], dtype=np.float64)
    return points


def read_manual_uv(path: Path) -> dict[int, np.ndarray]:
    uv: dict[int, np.ndarray] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(float(row["idx"]))
            uv[idx] = np.asarray([float(row["gt_u"]), float(row["gt_v"])], dtype=np.float64)
    return uv


def write_generated_corresp(path: Path, uv: np.ndarray, xyz: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        f.write(f"{len(uv)}\n")
        for point_uv, point_xyz in zip(uv, xyz):
            f.write(
                f"{point_uv[0]:.9f} {point_uv[1]:.9f} "
                f"{point_xyz[0]:.9f} {point_xyz[1]:.9f} {point_xyz[2]:.9f}\n"
            )


def resolve_platform_las(collection: str, path_key_value: str, step_dir: str, preprocess_suffix: str) -> Path | None:
    pre_dir = PREPROCESS_ROOT / collection / path_key_value / f"{step_dir}_{preprocess_suffix}"
    candidates = sorted(pre_dir.glob("*_projection_clean.las"))
    if candidates:
        return candidates[0]
    fallback_dir = PREPROCESS_ROOT / collection / path_key_value / step_dir
    candidates = sorted(fallback_dir.glob("*_projection_clean.las"))
    return candidates[0] if candidates else None


def save_picks_json(path: Path, uv: np.ndarray, xyz: np.ndarray, scene_key_value: str) -> None:
    picks = []
    for idx, (point_uv, point_xyz) in enumerate(zip(uv, xyz)):
        picks.append(
            {
                "index": idx,
                "image_uv": [float(point_uv[0]), float(point_uv[1])],
                "las_xyz": [float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2])],
                "status": "picked",
                "created_at": _now(),
                "updated_at": _now(),
            }
        )
    write_json(path, {"scene_key": scene_key_value, "picks": picks})


def migrate_one(summary_path: Path, results_root: Path, workspace_root: Path, preprocess_suffix: str, overwrite: bool) -> dict[str, Any]:
    collection, path_key_value, step_dir, path_name, step = parse_scene(summary_path, results_root)
    scene_dir = resolve_scene_dir(collection, path_name, step)
    workspace_dir = workspace_root / collection / path_key_value / step_dir
    fit_json = workspace_dir / "fit.json"
    if fit_json.exists() and not overwrite:
        return {
            "collection": collection,
            "path": path_key_value,
            "step": step_dir,
            "status": "skip_exists",
            "workspace_dir": str(workspace_dir),
        }

    manual_points_path = summary_path.parent / "manual_las_points.csv"
    manual_residuals_path = summary_path.parent / "manual_projection_residuals.csv"
    if not manual_points_path.exists() or not manual_residuals_path.exists():
        return {
            "collection": collection,
            "path": path_key_value,
            "step": step_dir,
            "status": "skip_missing_manual_inputs",
            "workspace_dir": str(workspace_dir),
        }

    manual_points = read_manual_las_points(manual_points_path)
    manual_uv = read_manual_uv(manual_residuals_path)
    ids = sorted(set(manual_points) & set(manual_uv))
    if len(ids) < 8:
        return {
            "collection": collection,
            "path": path_key_value,
            "step": step_dir,
            "status": "skip_insufficient_points",
            "num_points": len(ids),
            "workspace_dir": str(workspace_dir),
        }

    uv = np.asarray([manual_uv[idx] for idx in ids], dtype=np.float64)
    xyz = np.asarray([manual_points[idx] for idx in ids], dtype=np.float64)
    projection_las = resolve_platform_las(collection, path_key_value, step_dir, preprocess_suffix)
    if projection_las is None:
        return {
            "collection": collection,
            "path": path_key_value,
            "step": step_dir,
            "status": "skip_missing_projection_las",
            "workspace_dir": str(workspace_dir),
        }

    hdr_path = resolve_lwhsi_file(scene_dir, collection, path_name, step, ".hdr", required=True)
    gray8 = load_gray_preview(hdr_path)
    gray = gray8.astype(np.float64) / 255.0
    height, width = gray.shape
    cam_init, default_camera = build_default_cyl_camera(width, height)
    cam_opt, optimizer_stats = calibrate_single(
        uv[:, 0],
        uv[:, 1],
        xyz,
        cam_init,
        opt_mode="all",
        image_width=width,
        image_height=height,
        max_iters=400,
    )

    workspace_dir.mkdir(parents=True, exist_ok=True)
    generated_corresp = workspace_dir / "generated_corresp.txt"
    fitted_cyl = workspace_dir / "fitted.cyl"
    reprojection_preview = workspace_dir / "reprojection_preview.png"
    overlay_preview = workspace_dir / "overlay_preview.png"
    image_preview = workspace_dir / "image_preview.png"
    picks_json = workspace_dir / "picks.json"
    scene_json = workspace_dir / "scene.json"

    write_generated_corresp(generated_corresp, uv, xyz)
    write_cyl(cam_opt, str(fitted_cyl))
    save_picks_json(picks_json, uv, xyz, step_dir)
    if not image_preview.exists() or overwrite:
        import cv2

        cv2.imwrite(str(image_preview), gray8)

    uv_fit = project_vect_safe(xyz, cam_opt)
    residual = uv_fit - uv
    fit_metrics = metrics_from_residual(residual)
    scene_label = f"{path_name.split('_DistStA')[0]} Step{step}"
    save_manual_projection_plot(gray, uv, uv_fit, reprojection_preview, scene_label)

    las = laspy.read(projection_las)
    xyz_las = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    pc = (cam_opt.Rot @ xyz_las.T).T + cam_opt.t.reshape(1, 3)
    d = depth_range(pc)
    ij = project_vect_safe(xyz_las, cam_opt)
    valid = np.isfinite(ij[:, 0]) & np.isfinite(ij[:, 1]) & np.isfinite(d)
    i_vals = ij[valid, 0]
    j_vals = ij[valid, 1]
    d = d[valid]
    inside = (i_vals >= 0) & (i_vals < width) & (j_vals >= 0) & (j_vals < height)
    depth_img = rasterize(width, height, i_vals[inside].astype(np.float32), j_vals[inside].astype(np.float32), d[inside].astype(np.float32))
    save_overlay(gray, depth_img, overlay_preview, scene_label)

    scene_data = {
        "collection": collection,
        "path_key": path_key_value,
        "path_name": path_name,
        "scene_key": step_dir,
        "scene_label": scene_label,
        "step": step,
        "workspace_dir": str(workspace_dir),
        "source_paths": {
            "scene_dir": str(scene_dir),
            "hsi_hdr": str(hdr_path),
            "projection_las": str(projection_las),
            "generated_corresp_txt": str(generated_corresp),
            "fitted_cyl": str(fitted_cyl),
            "source_lidar_labeling_summary": str(summary_path),
        },
        "preprocessing": {
            "ready": True,
            "projection_las": str(projection_las),
            "preprocess_suffix": preprocess_suffix,
        },
        "capabilities": {
            "can_fit_generated_cyl": True,
            "migrated_from_lidar_labeling": True,
        },
        "defaults": {
            "default_camera": default_camera,
            "fit_opt_mode": "all",
            "min_fit_points": 8,
        },
    }
    write_json(scene_json, scene_data)

    fit_data = {
        "state": "ready",
        "ready": True,
        "mode": "generated_cyl",
        "updated_at": _now(),
        "migrated_from_lidar_labeling": True,
        "source_lidar_labeling_summary": str(summary_path),
        "picked_generated_points": int(len(ids)),
        "min_fit_points": 8,
        "fit_rmse_total": fit_metrics["rmse_total"],
        "fit_rmse_u": fit_metrics["rmse_u"],
        "fit_rmse_v": fit_metrics["rmse_v"],
        "fit_mean_abs_du": fit_metrics["mean_abs_du"],
        "fit_mean_abs_dv": fit_metrics["mean_abs_dv"],
        "optimizer_stats": optimizer_stats,
        "overlay_preview": str(overlay_preview),
        "reprojection_preview": str(reprojection_preview),
        "fitted_cyl": str(fitted_cyl),
        "fit_reference_uv": uv.tolist(),
        "fit_projected_uv": uv_fit.tolist(),
        "default_camera": default_camera,
        "optimized_camera": {
            "R": float(cam_opt.R),
            "w": float(cam_opt.w),
            "y": float(cam_opt.y),
            "f": float(cam_opt.f),
            "j0": float(cam_opt.j0),
            "Rot": cam_opt.Rot.tolist(),
            "t": cam_opt.t.tolist(),
        },
    }
    write_json(fit_json, fit_data)

    return {
        "collection": collection,
        "path": path_key_value,
        "step": step_dir,
        "status": "migrated",
        "num_points": len(ids),
        "fit_rmse_total": fit_metrics["rmse_total"],
        "workspace_dir": str(workspace_dir),
        "fitted_cyl": str(fitted_cyl),
        "generated_corresp": str(generated_corresp),
    }


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    workspace_root = Path(args.workspace_root)
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(results_root.rglob("summary.json")):
        collection, path_key_value, step_dir, _, _ = parse_scene(summary_path, results_root)
        if args.collection_filter and args.collection_filter not in collection:
            continue
        if args.path_filter and args.path_filter not in path_key_value:
            continue
        try:
            rows.append(
                migrate_one(
                    summary_path,
                    results_root,
                    workspace_root,
                    args.preprocess_suffix,
                    args.overwrite,
                )
            )
        except Exception as exc:
            rows.append(
                {
                    "collection": collection,
                    "path": path_key_value,
                    "step": step_dir,
                    "status": f"error:{type(exc).__name__}:{exc}",
                }
            )

    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", ""))
        status_counts[status] = status_counts.get(status, 0) + 1
    print(f"Scenes considered: {len(rows)}")
    print(f"Status counts: {status_counts}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
