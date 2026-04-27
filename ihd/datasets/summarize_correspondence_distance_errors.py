import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ihd.datasets.render_overlay_from_workspace import (
    load_gray,
    project_las,
    rasterize,
    read_json,
    resolve_local_artifact,
)
from ihd.datasets.cylindrical_camera import read_cam


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Compare picked 3D correspondence range against the local projected "
            "LiDAR depth label sampled near the corresponding picked 2D image point."
        )
    )
    ap.add_argument(
        "--workspace-roots",
        nargs="+",
        default=["analysis/annotation_workspace", "analysis/annotation_workspace_nocyl"],
        help="Annotation workspace roots to scan.",
    )
    ap.add_argument(
        "--out-dir",
        default="analysis/qc_review/correspondence_distance_errors",
        help="Output directory for per-point CSV, binned summary CSV, and histogram PNG.",
    )
    ap.add_argument(
        "--bins",
        default="auto-tertiles",
        help=(
            "Distance bin edges in meters, e.g. 0,25,50,inf, or auto-tertiles "
            "to choose near/mid/far bins from picked 3D distances."
        ),
    )
    ap.add_argument(
        "--preprocess-suffix",
        default="platform_sphere_r2p5",
        help="Preferred preprocessing suffix under analysis/lidar_preprocessing.",
    )
    ap.add_argument(
        "--sample-radius-px",
        type=int,
        default=5,
        help="Nearest valid projected-depth search radius around each picked 2D point.",
    )
    ap.add_argument(
        "--max-scenes",
        type=int,
        help="Optional debug limit for the number of workspace scenes to process.",
    )
    return ap.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def parse_explicit_bins(raw: str) -> list[float]:
    bins: list[float] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if token in {"inf", "+inf", "infinity"}:
            bins.append(math.inf)
        else:
            bins.append(float(token))
    if len(bins) < 2:
        raise ValueError("At least two bin edges are required")
    if any(bins[i] >= bins[i + 1] for i in range(len(bins) - 1)):
        raise ValueError(f"Bins must be strictly increasing: {bins}")
    return bins


def auto_tertile_bins(distances: list[float]) -> list[float]:
    if not distances:
        return [0.0, 25.0, 50.0, math.inf]
    q1, q2 = np.quantile(np.asarray(distances, dtype=np.float64), [1.0 / 3.0, 2.0 / 3.0])
    return [0.0, float(q1), float(q2), math.inf]


def bin_label(value: float, bins: list[float]) -> str:
    names = ["near", "middle", "far"]
    for idx, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        if lo <= value < hi:
            prefix = names[idx] if idx < len(names) else f"bin{idx + 1}"
            hi_txt = f"{hi:g}m" if math.isfinite(hi) else "inf"
            return f"{prefix}_{lo:g}-{hi_txt}"
    return "out_of_range"


def scene_parts(workspace_root: Path, fit_path: Path) -> tuple[str, str, str]:
    rel = fit_path.relative_to(workspace_root)
    if len(rel.parts) < 4:
        return "", "", ""
    return rel.parts[0], rel.parts[1], rel.parts[2]


def preferred_las_path(collection: str, path_key: str, step_dir: str, suffix: str, scene_data: dict[str, Any]) -> Path | None:
    pre_dir = Path("analysis/lidar_preprocessing") / collection / path_key / f"{step_dir}_{suffix}"
    candidates = sorted(pre_dir.glob("*_projection_clean.las"))
    if candidates:
        return candidates[0]
    raw = scene_data.get("source_paths", {}).get("projection_las")
    if raw and Path(raw).exists():
        return Path(raw)
    return None


def picked_distance_m(xyz: np.ndarray, cam) -> float:
    xyz_cam = cam.Rot @ xyz + cam.t
    return float(np.linalg.norm(xyz_cam))


def nearest_depth(depth_img: np.ndarray, uv: np.ndarray, radius_px: int) -> tuple[float, float, float, float]:
    x0 = int(round(float(uv[0])))
    y0 = int(round(float(uv[1])))
    height, width = depth_img.shape
    x_min = max(0, x0 - radius_px)
    x_max = min(width - 1, x0 + radius_px)
    y_min = max(0, y0 - radius_px)
    y_max = min(height - 1, y0 + radius_px)
    if x_min > x_max or y_min > y_max:
        return math.nan, math.nan, math.nan, math.nan

    patch = depth_img[y_min : y_max + 1, x_min : x_max + 1]
    yy, xx = np.nonzero(np.isfinite(patch))
    if len(xx) == 0:
        return math.nan, math.nan, math.nan, math.nan
    xs = xx + x_min
    ys = yy + y_min
    distances_px = np.hypot(xs - float(uv[0]), ys - float(uv[1]))
    best = int(np.argmin(distances_px))
    return float(patch[yy[best], xx[best]]), float(xs[best]), float(ys[best]), float(distances_px[best])


def discover_workspaces(workspace_roots: list[Path], max_scenes: int | None) -> list[tuple[Path, Path, Path]]:
    workspaces: list[tuple[Path, Path, Path]] = []
    for workspace_root in workspace_roots:
        if not workspace_root.exists():
            continue
        for fit_path in sorted(workspace_root.rglob("fit.json")):
            workspace_dir = fit_path.parent
            picks_path = workspace_dir / "picks.json"
            if picks_path.exists():
                workspaces.append((workspace_root, workspace_dir, fit_path))
                if max_scenes is not None and len(workspaces) >= max_scenes:
                    return workspaces
    return workspaces


def collect_unbinned_rows(args: argparse.Namespace) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    workspace_roots = [Path(root) for root in args.workspace_roots]
    for workspace_root, workspace_dir, fit_path in discover_workspaces(workspace_roots, args.max_scenes):
        fit = load_json(fit_path)
        if not fit.get("ready"):
            continue
        scene_json_path = workspace_dir / "scene.json"
        if not scene_json_path.exists():
            continue
        scene_data = read_json(scene_json_path)
        picks = load_json(workspace_dir / "picks.json").get("picks", [])
        valid_picks = [pick for pick in picks if pick.get("status") == "picked" and pick.get("las_xyz")]
        reference_uv = np.asarray(fit.get("fit_reference_uv", []), dtype=np.float64)
        n = min(len(reference_uv), len(valid_picks))
        if n == 0:
            continue

        collection, path_key, step_dir = scene_parts(workspace_root, fit_path)
        las_path = preferred_las_path(collection, path_key, step_dir, args.preprocess_suffix, scene_data)
        if las_path is None:
            continue

        try:
            cyl_path = resolve_local_artifact(workspace_dir, fit, "fitted_cyl", "fitted.cyl")
            cam = read_cam(str(cyl_path))
            gray = load_gray(workspace_dir, scene_data)
            height, width = gray.shape
            i_vals, j_vals, depth_vals = project_las(las_path, cam, fit, str(fit.get("mode")))
            depth_img = rasterize(width, height, i_vals, j_vals, depth_vals)
        except Exception as exc:
            rows.append(
                {
                    "collection": collection,
                    "path": path_key,
                    "step": step_dir,
                    "workspace_root": str(workspace_root),
                    "fit_path": str(fit_path),
                    "point_index": -1,
                    "status": f"scene_error:{type(exc).__name__}:{exc}",
                }
            )
            continue

        for idx in range(n):
            uv = reference_uv[idx]
            xyz = np.asarray(valid_picks[idx]["las_xyz"], dtype=np.float64)
            picked_range = picked_distance_m(xyz, cam)
            sampled_depth, sampled_x, sampled_y, nearest_px = nearest_depth(depth_img, uv, args.sample_radius_px)
            has_sample = math.isfinite(sampled_depth)
            depth_error = abs(sampled_depth - picked_range) if has_sample else math.nan
            rows.append(
                {
                    "collection": collection,
                    "path": path_key,
                    "step": step_dir,
                    "workspace_root": str(workspace_root),
                    "fit_path": str(fit_path),
                    "las_path": str(las_path),
                    "point_index": idx,
                    "status": "sampled" if has_sample else "missing_depth_near_pick",
                    "picked_u": float(uv[0]),
                    "picked_v": float(uv[1]),
                    "sample_radius_px": args.sample_radius_px,
                    "sampled_u": sampled_x,
                    "sampled_v": sampled_y,
                    "nearest_depth_pixel_distance_px": nearest_px,
                    "picked_range_m": picked_range,
                    "sampled_overlay_depth_m": sampled_depth,
                    "absolute_depth_error_m": depth_error,
                    "absolute_depth_error_percent_of_range": (
                        100.0 * depth_error / picked_range if has_sample and picked_range > 0 else math.nan
                    ),
                }
            )
    return rows


def assign_bins(rows: list[dict[str, str | float]], bins_arg: str) -> tuple[list[float], list[dict[str, str | float]]]:
    sampled_distances = [
        float(row["picked_range_m"])
        for row in rows
        if row.get("status") == "sampled" and isinstance(row.get("picked_range_m"), (float, int))
    ]
    bins = auto_tertile_bins(sampled_distances) if bins_arg == "auto-tertiles" else parse_explicit_bins(bins_arg)
    for row in rows:
        if row.get("status") == "sampled":
            row["distance_bin"] = bin_label(float(row["picked_range_m"]), bins)
        else:
            row["distance_bin"] = ""
    return bins, rows


def summarize(rows: list[dict[str, str | float]], bins: list[float]) -> list[dict[str, str | float | int]]:
    labels = [bin_label((lo + hi) / 2.0 if math.isfinite(hi) else lo + 1.0, bins) for lo, hi in zip(bins[:-1], bins[1:])]
    summary: list[dict[str, str | float | int]] = []
    for label in labels:
        bin_rows = [row for row in rows if row.get("status") == "sampled" and row.get("distance_bin") == label]
        if not bin_rows:
            summary.append({"distance_bin": label, "n": 0})
            continue
        ranges = np.asarray([float(row["picked_range_m"]) for row in bin_rows], dtype=np.float64)
        errors = np.asarray([float(row["absolute_depth_error_m"]) for row in bin_rows], dtype=np.float64)
        percents = np.asarray([float(row["absolute_depth_error_percent_of_range"]) for row in bin_rows], dtype=np.float64)
        nearest_px = np.asarray([float(row["nearest_depth_pixel_distance_px"]) for row in bin_rows], dtype=np.float64)
        summary.append(
            {
                "distance_bin": label,
                "n": int(len(bin_rows)),
                "picked_range_min_m": float(np.min(ranges)),
                "picked_range_median_m": float(np.median(ranges)),
                "picked_range_max_m": float(np.max(ranges)),
                "abs_depth_error_mean_m": float(np.mean(errors)),
                "abs_depth_error_median_m": float(np.median(errors)),
                "abs_depth_error_rmse_m": float(np.sqrt(np.mean(errors**2))),
                "abs_depth_error_p90_m": float(np.percentile(errors, 90)),
                "abs_depth_error_max_m": float(np.max(errors)),
                "abs_depth_error_median_percent_of_range": float(np.median(percents)),
                "abs_depth_error_p90_percent_of_range": float(np.percentile(percents, 90)),
                "nearest_depth_pixel_distance_median_px": float(np.median(nearest_px)),
                "nearest_depth_pixel_distance_p90_px": float(np.percentile(nearest_px, 90)),
            }
        )
    missing = sum(1 for row in rows if row.get("status") == "missing_depth_near_pick")
    scene_errors = sum(1 for row in rows if str(row.get("status", "")).startswith("scene_error"))
    summary.append({"distance_bin": "missing_depth_near_pick", "n": missing})
    summary.append({"distance_bin": "scene_errors", "n": scene_errors})
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_histogram(path: Path, rows: list[dict[str, str | float]], bins: list[float]) -> None:
    labels = [bin_label((lo + hi) / 2.0 if math.isfinite(hi) else lo + 1.0, bins) for lo, hi in zip(bins[:-1], bins[1:])]
    fig, axes = plt.subplots(len(labels), 1, figsize=(8, max(2.6, 2.3 * len(labels))), sharex=True)
    if len(labels) == 1:
        axes = [axes]
    for ax, label in zip(axes, labels):
        values = [
            float(row["absolute_depth_error_m"])
            for row in rows
            if row.get("status") == "sampled" and row.get("distance_bin") == label
        ]
        ax.hist(values, bins=24, color="#2f6f73", edgecolor="white")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
        ax.text(0.98, 0.82, f"n={len(values)}", transform=ax.transAxes, ha="right")
    axes[-1].set_xlabel("Absolute local depth disagreement near picked 2D point (meters)")
    fig.suptitle("Picked 3D range vs. local projected LiDAR depth label")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_overview_plot(path: Path, rows: list[dict[str, str | float]]) -> None:
    sampled = [row for row in rows if row.get("status") == "sampled"]
    if not sampled:
        return
    ranges = np.asarray([float(row["picked_range_m"]) for row in sampled], dtype=np.float64)
    errors = np.asarray([float(row["absolute_depth_error_m"]) for row in sampled], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.scatter(ranges, errors, s=8, alpha=0.45, color="#2f6f73")
    ax.set_xlabel("Picked 3D point range (meters)")
    ax.set_ylabel("Absolute local depth disagreement (meters)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.sample_radius_px < 0:
        raise ValueError("--sample-radius-px must be non-negative")
    out_dir = Path(args.out_dir)
    rows = collect_unbinned_rows(args)
    bins, rows = assign_bins(rows, args.bins)
    summary = summarize(rows, bins)
    write_csv(out_dir / "per_correspondence_local_depth_errors.csv", rows)
    write_csv(out_dir / "local_depth_error_distance_bin_summary.csv", summary)
    save_histogram(out_dir / "local_depth_error_histograms.png", rows, bins)
    save_overview_plot(out_dir / "local_depth_error_vs_range.png", rows)
    sampled = sum(1 for row in rows if row.get("status") == "sampled")
    print(f"Rows: {len(rows)}")
    print(f"Sampled correspondences: {sampled}")
    print(f"Bins: {bins}")
    print(f"Per-point CSV: {out_dir / 'per_correspondence_local_depth_errors.csv'}")
    print(f"Summary CSV: {out_dir / 'local_depth_error_distance_bin_summary.csv'}")
    print(f"Histogram: {out_dir / 'local_depth_error_histograms.png'}")


if __name__ == "__main__":
    main()
