import argparse
import csv
import json
import math
import time
from pathlib import Path

import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spectral as spy

from calibration_lidar_cylindrical import read_corresp
from cylindrical_camera import project_vect_safe, read_cam
from ihd.datasets.depth_rasterization import depth_range, rasterize


BASELINE_TXT_TO_LAS_R = np.array([
    [0.0, -1.0, 0.0],
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0],
], dtype=np.float64)
BASELINE_LAS_TO_TXT_R = BASELINE_TXT_TO_LAS_R.T
BASELINE_T_LAS = np.array([34.18, 1.75, 0.0], dtype=np.float64)
BASELINE_T_TXT = -(BASELINE_LAS_TO_TXT_R @ BASELINE_T_LAS)
MIN_MANUAL_LAS_POINTS_REQUIRED = 6
CYL_VERIFY_RMSE_TOTAL_THRESHOLD_PX = 10.0
FIT_RMSE_TOTAL_THRESHOLD_PX = 10.0
ACCEPTANCE_RULE_VERSION = "v1"


def parse_args():
    ap = argparse.ArgumentParser(
        description="Fit a rigid txt->las transform from manual pairs and compare image residuals to the fixed baseline."
    )
    ap.add_argument("--corresp", required=True, help="Correspondence .txt file")
    ap.add_argument("--manual-las-csv", required=True, help="CSV with x,y,z manual LAS points in the same order as corresp")
    ap.add_argument("--cyl", required=True, help="Scene .cyl file")
    ap.add_argument("--hsi-hdr", required=True, help="Scene .hdr file")
    ap.add_argument("--las", required=True, help="Scene .las file")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--scene-label", help="Short scene label for titles and summaries")
    ap.add_argument("--annotation-minutes", type=float, default=0.0, help="Manual annotation time in minutes")
    ap.add_argument("--verdict", default="pending", help="Qualitative verdict: good, usable with caution, bad, or pending")
    ap.add_argument("--chunk", type=int, default=2_000_000)
    return ap.parse_args()


def read_manual_las(path: Path) -> np.ndarray:
    rows = []
    with path.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if len(row) == 1:
                row = row[0].split(",")
            if len(row) >= 4:
                vals = [float(x) for x in row[-3:]]
            else:
                vals = [float(x) for x in row[:3]]
            rows.append(vals)
    return np.asarray(rows, dtype=np.float64)


def fit_rigid_transform(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    H = src_centered.T @ dst_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = dst_mean - (R @ src_mean)
    return R, t


def project_manual_points(las_points: np.ndarray, uv_gt: np.ndarray, cam, R_las_to_txt: np.ndarray, t_txt: np.ndarray):
    pts_txt = (R_las_to_txt @ las_points.T).T + t_txt.reshape(1, 3)
    uv_pred = project_vect_safe(pts_txt, cam)
    residual = uv_pred - uv_gt
    return pts_txt, uv_pred, residual


def load_gray(hdr_path: Path) -> np.ndarray:
    bsq = hdr_path.with_suffix(".bsq")
    img = spy.envi.open(str(hdr_path), str(bsq))
    cube = img.load()
    gray = cube.sum(axis=-1).astype(np.float64)
    gray -= gray.min()
    if gray.max() > 0:
        gray /= gray.max()
    return gray


def save_manual_projection_plot(
    gray: np.ndarray,
    uv_gt: np.ndarray,
    uv_base: np.ndarray,
    uv_fit: np.ndarray,
    out_path: Path,
    scene_label: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 4), dpi=160)
    ax.imshow(gray, cmap="gray", interpolation="nearest")
    ax.scatter(uv_gt[:, 0], uv_gt[:, 1], s=44, facecolors="none", edgecolors="red", linewidths=1.5, label="corr 2D")
    ax.scatter(uv_base[:, 0], uv_base[:, 1], s=28, marker="x", c="cyan", linewidths=1.2, label="baseline")
    ax.scatter(uv_fit[:, 0], uv_fit[:, 1], s=28, marker="+", c="yellow", linewidths=1.2, label="rigid fit")
    for gt, pred in zip(uv_gt, uv_fit):
        ax.plot([gt[0], pred[0]], [gt[1], pred[1]], color="yellow", alpha=0.5, linewidth=0.8)
    ax.set_xlim(0, gray.shape[1] - 1)
    ax.set_ylim(gray.shape[0] - 1, 0)
    ax.set_title(f"{scene_label}: manual LiDAR points projected with baseline vs fitted rigid transform")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_cyl_verification_plot(
    gray: np.ndarray,
    uv_gt: np.ndarray,
    uv_cyl: np.ndarray,
    out_path: Path,
    scene_label: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 4), dpi=160)
    ax.imshow(gray, cmap="gray", interpolation="nearest")
    ax.scatter(uv_gt[:, 0], uv_gt[:, 1], s=44, facecolors="none", edgecolors="red", linewidths=1.5, label="corr 2D")
    ax.scatter(uv_cyl[:, 0], uv_cyl[:, 1], s=28, marker="x", c="cyan", linewidths=1.2, label="txt XYZ via .cyl")
    for gt, pred in zip(uv_gt, uv_cyl):
        ax.plot([gt[0], pred[0]], [gt[1], pred[1]], color="yellow", alpha=0.5, linewidth=0.8)
    ax.set_xlim(0, gray.shape[1] - 1)
    ax.set_ylim(gray.shape[0] - 1, 0)
    ax.set_title(f"{scene_label}: .cyl verification against .txt correspondences")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_residual_csv(out_path: Path, uv_gt: np.ndarray, uv_base: np.ndarray, uv_fit: np.ndarray, res_base: np.ndarray, res_fit: np.ndarray):
    lines = ["idx,gt_u,gt_v,base_u,base_v,fit_u,fit_v,base_du,base_dv,fit_du,fit_dv"]
    for idx, (g, b, f, rb, rf) in enumerate(zip(uv_gt, uv_base, uv_fit, res_base, res_fit)):
        lines.append(f"{idx},{g[0]},{g[1]},{b[0]},{b[1]},{f[0]},{f[1]},{rb[0]},{rb[1]},{rf[0]},{rf[1]}")
    out_path.write_text("\n".join(lines) + "\n")


def metrics_from_residual(residual: np.ndarray) -> dict[str, float]:
    return {
        "rmse_u": float(np.sqrt(np.mean(residual[:, 0] ** 2))),
        "rmse_v": float(np.sqrt(np.mean(residual[:, 1] ** 2))),
        "rmse_total": float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))),
        "mean_abs_du": float(np.mean(np.abs(residual[:, 0]))),
        "mean_abs_dv": float(np.mean(np.abs(residual[:, 1]))),
    }


def build_summary_data(
    scene_label: str,
    R_fit: np.ndarray,
    t_fit: np.ndarray,
    res_base: np.ndarray,
    res_fit: np.ndarray,
    res_cyl: np.ndarray,
    num_txt_points: int,
    num_manual_las_points: int,
    num_las_points: int,
    num_projected_las_points: int,
    annotation_minutes: float,
    processing_seconds: float,
    verdict: str,
) -> dict:
    cyl_metrics = metrics_from_residual(res_cyl)
    base_metrics = metrics_from_residual(res_base)
    fit_metrics = metrics_from_residual(res_fit)
    total_minutes = annotation_minutes + (processing_seconds / 60.0)
    manual_points_pass = num_manual_las_points >= MIN_MANUAL_LAS_POINTS_REQUIRED
    cyl_verify_pass = cyl_metrics["rmse_total"] <= CYL_VERIFY_RMSE_TOTAL_THRESHOLD_PX
    fit_pass = fit_metrics["rmse_total"] <= FIT_RMSE_TOTAL_THRESHOLD_PX
    auto_accept_pass = manual_points_pass and cyl_verify_pass and fit_pass
    return {
        "scene_label": scene_label,
        "acceptance_rule_version": ACCEPTANCE_RULE_VERSION,
        "min_manual_las_points_required": MIN_MANUAL_LAS_POINTS_REQUIRED,
        "cyl_verify_rmse_total_threshold_px": CYL_VERIFY_RMSE_TOTAL_THRESHOLD_PX,
        "fit_rmse_total_threshold_px": FIT_RMSE_TOTAL_THRESHOLD_PX,
        "num_txt_points": int(num_txt_points),
        "num_manual_las_points": int(num_manual_las_points),
        "num_las_points": int(num_las_points),
        "num_projected_las_points": int(num_projected_las_points),
        "baseline_txt_to_las_rotation": BASELINE_TXT_TO_LAS_R.tolist(),
        "baseline_t_las": BASELINE_T_LAS.tolist(),
        "fitted_txt_to_las_rotation": R_fit.tolist(),
        "fitted_t_las": t_fit.tolist(),
        "cyl_verify_mean_abs_du": cyl_metrics["mean_abs_du"],
        "cyl_verify_mean_abs_dv": cyl_metrics["mean_abs_dv"],
        "cyl_verify_rmse_u": cyl_metrics["rmse_u"],
        "cyl_verify_rmse_v": cyl_metrics["rmse_v"],
        "cyl_verify_rmse_total": cyl_metrics["rmse_total"],
        "baseline_rmse_u": base_metrics["rmse_u"],
        "baseline_rmse_v": base_metrics["rmse_v"],
        "baseline_rmse_total": base_metrics["rmse_total"],
        "fit_rmse_u": fit_metrics["rmse_u"],
        "fit_rmse_v": fit_metrics["rmse_v"],
        "fit_rmse_total": fit_metrics["rmse_total"],
        "baseline_mean_abs_du": base_metrics["mean_abs_du"],
        "baseline_mean_abs_dv": base_metrics["mean_abs_dv"],
        "fit_mean_abs_du": fit_metrics["mean_abs_du"],
        "fit_mean_abs_dv": fit_metrics["mean_abs_dv"],
        "annotation_minutes": annotation_minutes,
        "processing_seconds": processing_seconds,
        "processing_minutes": processing_seconds / 60.0,
        "total_minutes": total_minutes,
        "manual_points_pass": manual_points_pass,
        "cyl_verify_pass": cyl_verify_pass,
        "fit_pass": fit_pass,
        "auto_accept_pass": auto_accept_pass,
        "verdict": verdict,
    }


def save_summary_json(out_path: Path, summary: dict) -> None:
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def save_overlay(gray: np.ndarray, depth_img: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask_valid = np.isfinite(depth_img)
    d_min = float(np.nanmin(depth_img[mask_valid]))
    d_max = float(np.nanmax(depth_img[mask_valid]))
    if d_max <= d_min:
        d_max = d_min + 1e-6
    H, W = gray.shape
    dpi = 100
    cb_px = 20
    fig = plt.figure(figsize=(W / dpi, (H + cb_px) / dpi), dpi=dpi)
    ax_img = fig.add_axes([0.0, 0.0, 1.0, H / (H + cb_px)])
    ax_cb = fig.add_axes([0.0, H / (H + cb_px), 1.0, cb_px / (H + cb_px)])
    ax_img.imshow(gray, cmap="gray", interpolation="nearest")
    yv, xv = np.nonzero(mask_valid)
    ax_img.scatter(xv, yv, c=depth_img[mask_valid], s=1, cmap="viridis_r", vmin=d_min, vmax=d_max, marker="s", linewidths=0)
    ax_img.set_xlim(0, W)
    ax_img.set_ylim(H, 0)
    ax_img.axis("off")
    gradient = np.linspace(d_min, d_max, max(2, W), dtype=np.float32)[None, :]
    ax_cb.imshow(gradient, aspect="auto", cmap="viridis_r", vmin=d_min, vmax=d_max, extent=[d_min, d_max, 0, 1])
    ax_cb.set_xlim(d_min, d_max)
    ax_cb.set_yticks([])
    span = d_max - d_min
    inset_data = span * (25 / W)
    ax_cb.set_xticks([d_min + inset_data, d_max - inset_data])
    ax_cb.set_xticklabels([f"{d_min:.0f} m", f"{d_max:.0f} m"])
    ax_cb.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, length=0, pad=2)
    for spine in ax_cb.spines.values():
        spine.set_visible(False)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def project_full_las(cam, las_path: Path, R_las_to_txt: np.ndarray, t_txt: np.ndarray, chunk_size: int):
    all_i = []
    all_j = []
    all_d = []
    with laspy.open(las_path) as lf:
        total = lf.header.point_count
        print(f"LAS points: {total}")
        for chunk in lf.chunk_iterator(chunk_size):
            xyz_las = np.column_stack((chunk.x, chunk.y, chunk.z)).astype(np.float64)
            xyz_txt = (R_las_to_txt @ xyz_las.T).T + t_txt.reshape(1, 3)
            Pc = (cam.Rot @ xyz_txt.T).T + cam.t.reshape(1, 3)
            d = depth_range(Pc)
            ij = project_vect_safe(xyz_txt, cam)
            i_vals = ij[:, 0]
            j_vals = ij[:, 1]
            finite = np.isfinite(i_vals) & np.isfinite(j_vals)
            if not np.any(finite):
                continue
            i_vals = i_vals[finite]
            j_vals = j_vals[finite]
            d = d[finite]
            inside = (i_vals >= 0) & (i_vals < W) & (j_vals >= 0) & (j_vals < H)
            if not np.any(inside):
                continue
            all_i.append(i_vals[inside].astype(np.float32))
            all_j.append(j_vals[inside].astype(np.float32))
            all_d.append(d[inside].astype(np.float32))
    i_all = np.concatenate(all_i) if all_i else np.empty(0, dtype=np.float32)
    j_all = np.concatenate(all_j) if all_j else np.empty(0, dtype=np.float32)
    d_all = np.concatenate(all_d) if all_d else np.empty(0, dtype=np.float32)
    return i_all, j_all, d_all, total


def main():
    args = parse_args()
    start = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_label = args.scene_label or out_dir.name

    corr_i, corr_j, corr_xyz = read_corresp(args.corresp)
    manual_las = read_manual_las(Path(args.manual_las_csv))
    if len(manual_las) != len(corr_xyz):
        raise ValueError("Manual LAS point count must match correspondence count.")

    cam = read_cam(args.cyl)
    gray = load_gray(Path(args.hsi_hdr))
    global H, W
    H, W = gray.shape

    R_fit, t_fit = fit_rigid_transform(corr_xyz, manual_las)
    R_fit_inv = R_fit.T
    t_fit_inv = -(R_fit_inv @ t_fit)

    uv_gt = np.column_stack((corr_i, corr_j))
    uv_cyl = project_vect_safe(corr_xyz, cam)
    res_cyl = uv_cyl - uv_gt
    _, uv_base, res_base = project_manual_points(manual_las, uv_gt, cam, BASELINE_LAS_TO_TXT_R, BASELINE_T_TXT)
    _, uv_fit, res_fit = project_manual_points(manual_las, uv_gt, cam, R_fit_inv, t_fit_inv)

    save_cyl_verification_plot(gray, uv_gt, uv_cyl, out_dir / "cyl_verification_overlay.png", scene_label)
    save_manual_projection_plot(gray, uv_gt, uv_base, uv_fit, out_dir / "manual_projection_comparison.png", scene_label)
    save_residual_csv(out_dir / "manual_projection_residuals.csv", uv_gt, uv_base, uv_fit, res_base, res_fit)

    i_all, j_all, d_all, num_las_points = project_full_las(cam, Path(args.las), R_fit_inv, t_fit_inv, args.chunk)
    depth_img = rasterize(W, H, i_all, j_all, d_all, reduce="min")
    save_overlay(gray, depth_img, out_dir / "fitted_rigid_overlay.png")
    np.savez_compressed(
        out_dir / "fitted_rigid_projection.npz",
        i=i_all,
        j=j_all,
        depth=d_all,
        width=W,
        height=H,
        fitted_txt_to_las_rotation=R_fit.astype(np.float32),
        fitted_txt_to_las_translation=t_fit.astype(np.float32),
        las_to_txt_rotation=R_fit_inv.astype(np.float32),
        las_to_txt_translation=t_fit_inv.astype(np.float32),
    )
    processing_seconds = time.perf_counter() - start
    summary = build_summary_data(
        scene_label,
        R_fit,
        t_fit,
        res_base,
        res_fit,
        res_cyl,
        len(corr_xyz),
        len(manual_las),
        num_las_points,
        len(i_all),
        args.annotation_minutes,
        processing_seconds,
        args.verdict,
    )
    save_summary_json(out_dir / "summary.json", summary)
    print("Baseline vs fitted residuals saved.")
    print(f"Baseline mean abs (du,dv): {np.mean(np.abs(res_base[:,0])):.3f}, {np.mean(np.abs(res_base[:,1])):.3f}")
    print(f"Fitted   mean abs (du,dv): {np.mean(np.abs(res_fit[:,0])):.3f}, {np.mean(np.abs(res_fit[:,1])):.3f}")
    print(f".cyl verify mean abs (du,dv): {np.mean(np.abs(res_cyl[:,0])):.3f}, {np.mean(np.abs(res_cyl[:,1])):.3f}")
    print(f"Annotation minutes: {args.annotation_minutes:.3f}")
    print(f"Processing seconds: {processing_seconds:.3f}")
    print(f"Saved fitted overlay and projection to {out_dir}")


if __name__ == "__main__":
    main()
