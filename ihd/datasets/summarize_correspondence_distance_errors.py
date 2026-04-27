import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ihd.datasets.cylindrical_camera import read_cam


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Summarize picked-correspondence reprojection residuals by 3D distance. "
            "This audits the sparse calibration correspondences, not dense projected depth labels."
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
        default="0,25,50,inf",
        help="Comma-separated distance bin edges in meters. Use inf for the last open edge.",
    )
    return ap.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def parse_bins(raw: str) -> list[float]:
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


def bin_label(value: float, bins: list[float]) -> str:
    for lo, hi in zip(bins[:-1], bins[1:]):
        if lo <= value < hi:
            return f"{lo:g}-{hi:g}m" if math.isfinite(hi) else f"{lo:g}+m"
    return "out_of_range"


def resolve_fitted_cyl(workspace_dir: Path, fit: dict[str, Any]) -> Path | None:
    candidates = []
    if fit.get("fitted_cyl"):
        candidates.append(workspace_dir / str(fit["fitted_cyl"]))
        candidates.append(Path(str(fit["fitted_cyl"])))
    candidates.append(workspace_dir / "fitted.cyl")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def scene_parts(workspace_root: Path, fit_path: Path) -> tuple[str, str, str]:
    rel = fit_path.relative_to(workspace_root)
    if len(rel.parts) < 4:
        return "", "", ""
    return rel.parts[0], rel.parts[1], rel.parts[2]


def collect_rows(workspace_roots: list[Path], bins: list[float]) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for workspace_root in workspace_roots:
        if not workspace_root.exists():
            continue
        for fit_path in sorted(workspace_root.rglob("fit.json")):
            workspace_dir = fit_path.parent
            picks_path = workspace_dir / "picks.json"
            if not picks_path.exists():
                continue
            fit = read_json(fit_path)
            if not fit.get("ready"):
                continue
            reference_uv = np.asarray(fit.get("fit_reference_uv", []), dtype=np.float64)
            projected_uv = np.asarray(fit.get("fit_projected_uv", []), dtype=np.float64)
            picks = read_json(picks_path).get("picks", [])
            valid_picks = [pick for pick in picks if pick.get("status") == "picked" and pick.get("las_xyz")]
            n = min(len(reference_uv), len(projected_uv), len(valid_picks))
            if n == 0:
                continue

            cyl_path = resolve_fitted_cyl(workspace_dir, fit)
            cam = read_cam(str(cyl_path)) if cyl_path is not None else None
            collection, path_key, step_dir = scene_parts(workspace_root, fit_path)

            for idx in range(n):
                ref = reference_uv[idx]
                proj = projected_uv[idx]
                xyz = np.asarray(valid_picks[idx]["las_xyz"], dtype=np.float64)
                if cam is not None:
                    xyz_cam = cam.Rot @ xyz + cam.t
                    distance_m = float(np.linalg.norm(xyz_cam))
                    distance_frame = "camera"
                    du_angle_rad = float((proj[0] - ref[0]) * cam.y)
                    ref_v_angle = math.atan2(float(ref[1] - cam.j0), cam.f)
                    proj_v_angle = math.atan2(float(proj[1] - cam.j0), cam.f)
                    dv_angle_rad = float(proj_v_angle - ref_v_angle)
                else:
                    distance_m = float(np.linalg.norm(xyz))
                    distance_frame = "point_frame"
                    du_angle_rad = math.nan
                    dv_angle_rad = math.nan
                du = float(proj[0] - ref[0])
                dv = float(proj[1] - ref[1])
                residual_px = float(np.hypot(du, dv))
                angular_residual_rad = float(np.hypot(du_angle_rad, dv_angle_rad))
                metric_residual_m = float(distance_m * angular_residual_rad)
                rows.append(
                    {
                        "collection": collection,
                        "path": path_key,
                        "step": step_dir,
                        "workspace_root": str(workspace_root),
                        "fit_path": str(fit_path),
                        "point_index": idx,
                        "distance_m": distance_m,
                        "distance_frame": distance_frame,
                        "distance_bin": bin_label(distance_m, bins),
                        "du_px": du,
                        "dv_px": dv,
                        "residual_px": residual_px,
                        "du_angle_rad": du_angle_rad,
                        "dv_angle_rad": dv_angle_rad,
                        "angular_residual_rad": angular_residual_rad,
                        "metric_residual_m": metric_residual_m,
                        "metric_residual_percent_of_range": (
                            100.0 * metric_residual_m / distance_m if distance_m > 0 else math.nan
                        ),
                    }
                )
    return rows


def summarize(rows: list[dict[str, str | float]], bins: list[float]) -> list[dict[str, str | float | int]]:
    summary = []
    labels = [f"{lo:g}-{hi:g}m" if math.isfinite(hi) else f"{lo:g}+m" for lo, hi in zip(bins[:-1], bins[1:])]
    for label in labels:
        bin_rows = [row for row in rows if row["distance_bin"] == label]
        values = np.asarray([float(row["residual_px"]) for row in bin_rows], dtype=np.float64)
        metric_values = np.asarray([float(row["metric_residual_m"]) for row in bin_rows], dtype=np.float64)
        percent_values = np.asarray(
            [float(row["metric_residual_percent_of_range"]) for row in bin_rows],
            dtype=np.float64,
        )
        distances = np.asarray([float(row["distance_m"]) for row in bin_rows], dtype=np.float64)
        if len(bin_rows) == 0:
            summary.append({"distance_bin": label, "n": 0})
            continue
        metric_values = metric_values[np.isfinite(metric_values)]
        percent_values = percent_values[np.isfinite(percent_values)]
        summary.append(
            {
                "distance_bin": label,
                "n": int(len(values)),
                "distance_min_m": float(np.min(distances)),
                "distance_median_m": float(np.median(distances)),
                "distance_max_m": float(np.max(distances)),
                "residual_mean_px": float(np.mean(values)),
                "residual_median_px": float(np.median(values)),
                "residual_rmse_px": float(np.sqrt(np.mean(values**2))),
                "residual_p90_px": float(np.percentile(values, 90)),
                "residual_max_px": float(np.max(values)),
                "metric_mean_m": float(np.mean(metric_values)),
                "metric_median_m": float(np.median(metric_values)),
                "metric_rmse_m": float(np.sqrt(np.mean(metric_values**2))),
                "metric_p90_m": float(np.percentile(metric_values, 90)),
                "metric_max_m": float(np.max(metric_values)),
                "metric_median_percent_of_range": float(np.median(percent_values)),
                "metric_p90_percent_of_range": float(np.percentile(percent_values, 90)),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_histogram(path: Path, rows: list[dict[str, str | float]], bins: list[float]) -> None:
    labels = [f"{lo:g}-{hi:g}m" if math.isfinite(hi) else f"{lo:g}+m" for lo, hi in zip(bins[:-1], bins[1:])]
    fig, axes = plt.subplots(len(labels), 1, figsize=(8, max(2.4, 2.2 * len(labels))), sharex=True)
    if len(labels) == 1:
        axes = [axes]
    for ax, label in zip(axes, labels):
        values = [float(row["residual_px"]) for row in rows if row["distance_bin"] == label]
        ax.hist(values, bins=24, color="#2f6f73", edgecolor="white")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
        ax.text(0.98, 0.82, f"n={len(values)}", transform=ax.transAxes, ha="right")
    axes[-1].set_xlabel("Correspondence reprojection residual (pixels)")
    fig.suptitle("Sparse correspondence residuals by 3D distance bin")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_metric_histogram(path: Path, rows: list[dict[str, str | float]], bins: list[float]) -> None:
    labels = [f"{lo:g}-{hi:g}m" if math.isfinite(hi) else f"{lo:g}+m" for lo, hi in zip(bins[:-1], bins[1:])]
    fig, axes = plt.subplots(len(labels), 1, figsize=(8, max(2.4, 2.2 * len(labels))), sharex=True)
    if len(labels) == 1:
        axes = [axes]
    for ax, label in zip(axes, labels):
        values = [
            float(row["metric_residual_m"])
            for row in rows
            if row["distance_bin"] == label and math.isfinite(float(row["metric_residual_m"]))
        ]
        ax.hist(values, bins=24, color="#8a5a22", edgecolor="white")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
        ax.text(0.98, 0.82, f"n={len(values)}", transform=ax.transAxes, ha="right")
    axes[-1].set_xlabel("Approx. metric reprojection residual at point range (meters)")
    fig.suptitle("Sparse correspondence metric residuals by 3D distance bin")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    bins = parse_bins(args.bins)
    out_dir = Path(args.out_dir)
    rows = collect_rows([Path(root) for root in args.workspace_roots], bins)
    summary = summarize(rows, bins)
    write_csv(out_dir / "per_correspondence_errors.csv", rows)
    write_csv(out_dir / "distance_bin_summary.csv", summary)
    save_histogram(out_dir / "distance_bin_histograms.png", rows, bins)
    save_metric_histogram(out_dir / "distance_bin_metric_histograms.png", rows, bins)
    print(f"Correspondences: {len(rows)}")
    print(f"Per-point CSV: {out_dir / 'per_correspondence_errors.csv'}")
    print(f"Summary CSV: {out_dir / 'distance_bin_summary.csv'}")
    print(f"Histogram: {out_dir / 'distance_bin_histograms.png'}")
    print(f"Metric histogram: {out_dir / 'distance_bin_metric_histograms.png'}")


if __name__ == "__main__":
    main()
