import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

from ihd.datasets.render_overlay_from_workspace import (
    auto_title,
    load_gray,
    project_las,
    rasterize,
    read_json,
    resolve_local_artifact,
    save_overlay,
    suppress_far_occlusion_bleed,
)
from ihd.datasets.cylindrical_camera import project_vect_safe, read_cam
from ihd.qc_review.scene_service import (
    ANALYSIS_ROOT,
    QC_ROOT,
    build_reference_preview,
    discover_qc_scenes,
    resolve_hsi_hdr,
    resolve_scene_dir,
)
from ihd.qc_review.stage_to_disk import derive_output_names


def depth_range(points_cam: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points_cam, axis=1)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Render title-free QC overlays from a named preprocessing output and stage "
            "the DepthOverlay PNGs into the original /disk scene folders."
        )
    )
    ap.add_argument(
        "--results-root",
        default=str(ANALYSIS_ROOT / "lidar_labeling"),
        help="Primary QC results root; workspace roots are discovered next to it.",
    )
    ap.add_argument("--data-root", default="/disk", help="Shared dataset root.")
    ap.add_argument(
        "--preprocess-suffix",
        default="platform_sphere_r2p5",
        help="Suffix appended to each step dir under analysis/lidar_preprocessing.",
    )
    ap.add_argument(
        "--out-root",
        default=str(ANALYSIS_ROOT / "overlay_checks" / "platform_sphere_r2p5"),
        help="Local root for rendered overlay copies before staging to /disk.",
    )
    ap.add_argument(
        "--title-mode",
        choices=["none", "auto"],
        default="none",
        help="Whether local rendered copies include a path/step title.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing disk overlay PNGs.")
    ap.add_argument(
        "--occlusion-filter-radius-px",
        type=int,
        default=0,
        help="Suppress far depth pixels if a much closer pixel exists within this radius. Disabled at 0.",
    )
    ap.add_argument(
        "--occlusion-min-depth-gap-m",
        type=float,
        default=1.0,
        help="Minimum absolute range gap required for occlusion suppression.",
    )
    ap.add_argument(
        "--occlusion-min-depth-gap-ratio",
        type=float,
        default=0.05,
        help="Minimum relative range gap required for occlusion suppression, e.g. 0.05 = 5%%.",
    )
    ap.add_argument(
        "--manifest-out",
        default=str(QC_ROOT / "staged_rendered_overlays_manifest.csv"),
        help="CSV manifest summarizing rendered/staged overlays.",
    )
    return ap.parse_args()


def workspace_dir_for_scene(scene) -> Path | None:
    if scene.summary_path is None:
        return None
    if scene.summary_path.name == "fit.json":
        return scene.summary_path.parent
    return None


def resolve_cyl(scene_dir: Path, hdr_path: Path) -> Path | None:
    stem = hdr_path.stem
    candidates = [
        scene_dir / f"{stem}.cyl",
        *sorted(scene_dir.glob("*LWHSI1*.cyl")),
        *sorted(scene_dir.glob("*.cyl")),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def preprocessed_las_for_scene(scene, preprocess_suffix: str) -> Path | None:
    pre_dir = (
        ANALYSIS_ROOT
        / "lidar_preprocessing"
        / scene.collection
        / scene.path_key
        / f"{scene.step_dir}_{preprocess_suffix}"
    )
    candidates = sorted(pre_dir.glob("*_projection_clean.las"))
    return candidates[0] if candidates else None


def ensure_reference_png(scene, scene_dir: Path, ref_target: Path) -> str:
    if scene.reference_png_path is not None and scene.reference_png_path.exists():
        shutil.copy2(scene.reference_png_path, ref_target)
        return str(scene.reference_png_path)
    hdr_path = scene.reference_hdr_path or resolve_hsi_hdr(scene_dir, scene.collection, scene.path_key, scene.step_dir)
    if hdr_path is None:
        raise FileNotFoundError(f"Missing reference source for {scene.collection}/{scene.path_key}/{scene.step_dir}")
    build_reference_preview(hdr_path, ref_target)
    return str(hdr_path)


def apply_occlusion_filter(depth_img: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    return suppress_far_occlusion_bleed(
        depth_img,
        args.occlusion_filter_radius_px,
        args.occlusion_min_depth_gap_m,
        args.occlusion_min_depth_gap_ratio,
    )


def render_overlay(workspace_dir: Path, las_path: Path, out_path: Path, title_mode: str, args: argparse.Namespace) -> dict[str, int | str]:
    fit_data = read_json(workspace_dir / "fit.json")
    scene_data = read_json(workspace_dir / "scene.json")
    if not bool(fit_data.get("ready")):
        raise ValueError(f"Workspace fit is not ready: {workspace_dir}")

    cyl_path = resolve_local_artifact(workspace_dir, fit_data, "fitted_cyl", "fitted.cyl")
    cam = read_cam(str(cyl_path))
    gray = load_gray(workspace_dir, scene_data)
    height, width = gray.shape
    i_vals, j_vals, d = project_las(las_path, cam, fit_data, str(fit_data.get("mode")))
    inside = (i_vals >= 0) & (i_vals < width) & (j_vals >= 0) & (j_vals < height)
    i_vals = i_vals[inside]
    j_vals = j_vals[inside]
    d = d[inside]
    depth_img = rasterize(width, height, i_vals, j_vals, d)
    depth_img = apply_occlusion_filter(depth_img, args)
    title = auto_title(scene_data, workspace_dir) if title_mode == "auto" else None
    save_overlay(gray, depth_img, out_path, title)
    return {
        "projected_points": int(len(i_vals)),
        "valid_pixels": int(np.isfinite(depth_img).sum()),
    }


def render_lidar_labeling_overlay(
    scene,
    scene_dir: Path,
    hdr_path: Path,
    las_path: Path,
    out_path: Path,
    title_mode: str,
    args: argparse.Namespace,
) -> dict[str, int | str]:
    cyl_path = resolve_cyl(scene_dir, hdr_path)
    if cyl_path is None:
        raise FileNotFoundError(f"Could not resolve .cyl in {scene_dir}")
    cam = read_cam(str(cyl_path))

    import spectral as spy

    bsq = hdr_path.with_suffix(".bsq")
    img = spy.envi.open(str(hdr_path), str(bsq))
    cube = img.load()
    gray = cube.sum(axis=-1).astype(np.float64)
    gray -= gray.min()
    if gray.max() > 0:
        gray /= gray.max()
    height, width = gray.shape

    summary = scene.summary
    if "las_to_txt_rotation" in summary and "las_to_txt_translation" in summary:
        rot = np.asarray(summary["las_to_txt_rotation"], dtype=np.float64)
        trans = np.asarray(summary["las_to_txt_translation"], dtype=np.float64)
    else:
        txt_to_las_rot = np.asarray(summary["fitted_txt_to_las_rotation"], dtype=np.float64)
        txt_to_las_t = np.asarray(summary["fitted_t_las"], dtype=np.float64)
        rot = txt_to_las_rot.T
        trans = -(rot @ txt_to_las_t)

    import laspy

    las = laspy.read(las_path)
    xyz_las = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    xyz_txt = (rot @ xyz_las.T).T + trans.reshape(1, 3)
    pc = (cam.Rot @ xyz_txt.T).T + cam.t.reshape(1, 3)
    d = depth_range(pc)
    ij = project_vect_safe(xyz_txt, cam)
    i_vals = ij[:, 0]
    j_vals = ij[:, 1]
    valid = np.isfinite(i_vals) & np.isfinite(j_vals) & np.isfinite(d)
    i_vals = i_vals[valid].astype(np.float32)
    j_vals = j_vals[valid].astype(np.float32)
    d = d[valid].astype(np.float32)
    inside = (i_vals >= 0) & (i_vals < width) & (j_vals >= 0) & (j_vals < height)
    i_vals = i_vals[inside]
    j_vals = j_vals[inside]
    d = d[inside]
    depth_img = rasterize(width, height, i_vals, j_vals, d)
    depth_img = apply_occlusion_filter(depth_img, args)
    title = f"{scene.path_key} {scene.step_dir.split('_')[-1].title()}" if title_mode == "auto" else None
    save_overlay(gray, depth_img, out_path, title)
    return {
        "projected_points": int(len(i_vals)),
        "valid_pixels": int(np.isfinite(depth_img).sum()),
    }


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    scenes = discover_qc_scenes(results_root=results_root, data_root=data_root, cache_root=QC_ROOT / "cache")
    rows: list[dict[str, str]] = []

    for scene in scenes:
        scene_id = f"{scene.collection}/{scene.path_key}/{scene.step_dir}"
        workspace_dir = workspace_dir_for_scene(scene)
        scene_dir = resolve_scene_dir(scene.collection, scene.path_key, scene.step_dir, data_root)
        las_path = preprocessed_las_for_scene(scene, args.preprocess_suffix)

        base_row = {
            "scene": scene_id,
            "status": "",
            "workspace_dir": str(workspace_dir) if workspace_dir else "",
            "preprocessed_las": str(las_path) if las_path else "",
            "local_overlay": "",
            "disk_overlay": "",
            "disk_reference": "",
            "projected_points": "",
            "valid_pixels": "",
        }

        if scene_dir is None:
            rows.append({**base_row, "status": "skip_missing_scene_dir"})
            continue
        if las_path is None:
            rows.append({**base_row, "status": "skip_missing_preprocessed_las"})
            continue

        hdr_path = scene.reference_hdr_path or resolve_hsi_hdr(scene_dir, scene.collection, scene.path_key, scene.step_dir)
        if hdr_path is None:
            rows.append({**base_row, "status": "skip_missing_hdr"})
            continue

        ref_name, overlay_name = derive_output_names(hdr_path)
        disk_ref = scene_dir / ref_name
        disk_overlay = scene_dir / overlay_name
        if disk_overlay.exists() and not args.overwrite:
            rows.append(
                {
                    **base_row,
                    "status": "skip_overlay_exists",
                    "disk_overlay": str(disk_overlay),
                    "disk_reference": str(disk_ref),
                }
            )
            continue

        local_overlay = out_root / scene.collection / scene.path_key / scene.step_dir / overlay_name
        try:
            if workspace_dir is not None:
                stats = render_overlay(workspace_dir, las_path, local_overlay, args.title_mode, args)
            else:
                stats = render_lidar_labeling_overlay(scene, scene_dir, hdr_path, las_path, local_overlay, args.title_mode, args)
            ensure_reference_png(scene, scene_dir, disk_ref)
            shutil.copy2(local_overlay, disk_overlay)
        except Exception as exc:
            rows.append({**base_row, "status": f"error:{type(exc).__name__}:{exc}"})
            continue

        rows.append(
            {
                **base_row,
                "status": "staged",
                "local_overlay": str(local_overlay),
                "disk_overlay": str(disk_overlay),
                "disk_reference": str(disk_ref),
                "projected_points": str(stats["projected_points"]),
                "valid_pixels": str(stats["valid_pixels"]),
            }
        )

    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scene",
        "status",
        "workspace_dir",
        "preprocessed_las",
        "local_overlay",
        "disk_overlay",
        "disk_reference",
        "projected_points",
        "valid_pixels",
    ]
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    staged = sum(1 for row in rows if row["status"] == "staged")
    print(f"Scenes considered: {len(rows)}")
    print(f"Staged overlays: {staged}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
