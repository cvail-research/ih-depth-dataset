import argparse
import csv
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


def parse_args():
    ap = argparse.ArgumentParser(
        description="Fit rigid transforms from manual LAS/TXT pairs and compare transfer across scenes."
    )
    ap.add_argument("--source-corresp", required=True)
    ap.add_argument("--source-manual-las", required=True)
    ap.add_argument("--target-corresp", required=True)
    ap.add_argument("--target-manual-las", required=True)
    ap.add_argument("--target-cyl", required=True)
    ap.add_argument("--target-hsi-hdr", required=True)
    ap.add_argument("--target-las", required=True)
    ap.add_argument("--out-dir", required=True)
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
            rows.append([float(x) for x in row[:3]])
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


def invert_transform(R_txt_to_las: np.ndarray, t_txt_to_las: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R_las_to_txt = R_txt_to_las.T
    t_las_to_txt = -(R_las_to_txt @ t_txt_to_las)
    return R_las_to_txt, t_las_to_txt


def project_manual_points(
    las_points: np.ndarray,
    uv_gt: np.ndarray,
    cam,
    R_las_to_txt: np.ndarray,
    t_las_to_txt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pts_txt = (R_las_to_txt @ las_points.T).T + t_las_to_txt.reshape(1, 3)
    uv_pred = project_vect_safe(pts_txt, cam)
    residual = uv_pred - uv_gt
    return uv_pred, residual


def load_gray(hdr_path: Path) -> np.ndarray:
    bsq = hdr_path.with_suffix(".bsq")
    img = spy.envi.open(str(hdr_path), str(bsq))
    cube = img.load()
    gray = cube.sum(axis=-1).astype(np.float64)
    gray -= gray.min()
    if gray.max() > 0:
        gray /= gray.max()
    return gray


def save_manual_plot(
    gray: np.ndarray,
    uv_gt: np.ndarray,
    uv_transfer: np.ndarray,
    uv_target: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 4), dpi=160)
    ax.imshow(gray, cmap="gray", interpolation="nearest")
    ax.scatter(uv_gt[:, 0], uv_gt[:, 1], s=46, facecolors="none", edgecolors="red", linewidths=1.5, label="corr 2D")
    ax.scatter(uv_transfer[:, 0], uv_transfer[:, 1], s=28, marker="x", c="cyan", linewidths=1.2, label="Step5 fit on Step6")
    ax.scatter(uv_target[:, 0], uv_target[:, 1], s=28, marker="+", c="yellow", linewidths=1.2, label="Step6 own fit")
    ax.set_xlim(0, gray.shape[1] - 1)
    ax.set_ylim(gray.shape[0] - 1, 0)
    ax.set_title("Step6 manual LiDAR points: transferred Step5 fit vs Step6 own fit")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_overlay(gray: np.ndarray, depth_img: np.ndarray, out_path: Path) -> None:
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


def project_full_las(
    las_path: Path,
    cam,
    R_las_to_txt: np.ndarray,
    t_las_to_txt: np.ndarray,
    width: int,
    height: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_i = []
    all_j = []
    all_d = []
    with laspy.open(las_path) as lf:
        for chunk in lf.chunk_iterator(chunk_size):
            xyz_las = np.column_stack((chunk.x, chunk.y, chunk.z)).astype(np.float64)
            xyz_txt = (R_las_to_txt @ xyz_las.T).T + t_las_to_txt.reshape(1, 3)
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
            inside = (i_vals >= 0) & (i_vals < width) & (j_vals >= 0) & (j_vals < height)
            if not np.any(inside):
                continue
            all_i.append(i_vals[inside].astype(np.float32))
            all_j.append(j_vals[inside].astype(np.float32))
            all_d.append(d[inside].astype(np.float32))
    i_all = np.concatenate(all_i) if all_i else np.empty(0, dtype=np.float32)
    j_all = np.concatenate(all_j) if all_j else np.empty(0, dtype=np.float32)
    d_all = np.concatenate(all_d) if all_d else np.empty(0, dtype=np.float32)
    return i_all, j_all, d_all


def metrics(residual: np.ndarray) -> dict[str, float]:
    return {
        "mean_abs_du": float(np.mean(np.abs(residual[:, 0]))),
        "mean_abs_dv": float(np.mean(np.abs(residual[:, 1]))),
        "rmse_u": float(np.sqrt(np.mean(residual[:, 0] ** 2))),
        "rmse_v": float(np.sqrt(np.mean(residual[:, 1] ** 2))),
        "rmse_total": float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))),
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, _, src_txt = read_corresp(args.source_corresp)
    src_las = read_manual_las(Path(args.source_manual_las))
    tgt_i, tgt_j, tgt_txt = read_corresp(args.target_corresp)
    tgt_las = read_manual_las(Path(args.target_manual_las))

    if len(src_txt) != len(src_las):
        raise ValueError("Source manual LAS point count must match source correspondences.")
    if len(tgt_txt) != len(tgt_las):
        raise ValueError("Target manual LAS point count must match target correspondences.")

    R_src, t_src = fit_rigid_transform(src_txt, src_las)
    R_tgt, t_tgt = fit_rigid_transform(tgt_txt, tgt_las)
    R_src_inv, t_src_inv = invert_transform(R_src, t_src)
    R_tgt_inv, t_tgt_inv = invert_transform(R_tgt, t_tgt)

    cam = read_cam(args.target_cyl)
    gray = load_gray(Path(args.target_hsi_hdr))
    H, W = gray.shape
    uv_gt = np.column_stack((tgt_i, tgt_j))

    uv_transfer, res_transfer = project_manual_points(tgt_las, uv_gt, cam, R_src_inv, t_src_inv)
    uv_target, res_target = project_manual_points(tgt_las, uv_gt, cam, R_tgt_inv, t_tgt_inv)

    save_manual_plot(gray, uv_gt, uv_transfer, uv_target, out_dir / "manual_projection_comparison.png")

    with (out_dir / "manual_projection_residuals.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "gt_u", "gt_v", "transfer_u", "transfer_v", "target_u", "target_v", "transfer_du", "transfer_dv", "target_du", "target_dv"])
        for idx, (gt, tr, tg, rr, rt) in enumerate(zip(uv_gt, uv_transfer, uv_target, res_transfer, res_target)):
            writer.writerow([idx, gt[0], gt[1], tr[0], tr[1], tg[0], tg[1], rr[0], rr[1], rt[0], rt[1]])

    m_transfer = metrics(res_transfer)
    m_target = metrics(res_target)
    summary_lines = [
        "source_step_fit_txt_to_las_rotation:",
        *[" ".join(f"{v:.8f}" for v in row) for row in R_src],
        f"source_step_fit_txt_to_las_translation: {t_src[0]:.8f} {t_src[1]:.8f} {t_src[2]:.8f}",
        "target_step_fit_txt_to_las_rotation:",
        *[" ".join(f"{v:.8f}" for v in row) for row in R_tgt],
        f"target_step_fit_txt_to_las_translation: {t_tgt[0]:.8f} {t_tgt[1]:.8f} {t_tgt[2]:.8f}",
        f"transfer_mean_abs_du: {m_transfer['mean_abs_du']:.6f}",
        f"transfer_mean_abs_dv: {m_transfer['mean_abs_dv']:.6f}",
        f"transfer_rmse_u: {m_transfer['rmse_u']:.6f}",
        f"transfer_rmse_v: {m_transfer['rmse_v']:.6f}",
        f"transfer_rmse_total: {m_transfer['rmse_total']:.6f}",
        f"target_mean_abs_du: {m_target['mean_abs_du']:.6f}",
        f"target_mean_abs_dv: {m_target['mean_abs_dv']:.6f}",
        f"target_rmse_u: {m_target['rmse_u']:.6f}",
        f"target_rmse_v: {m_target['rmse_v']:.6f}",
        f"target_rmse_total: {m_target['rmse_total']:.6f}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")

    i_transfer, j_transfer, d_transfer = project_full_las(Path(args.target_las), cam, R_src_inv, t_src_inv, W, H, args.chunk)
    depth_transfer = rasterize(W, H, i_transfer, j_transfer, d_transfer, reduce="min")
    save_overlay(gray, depth_transfer, out_dir / "transfer_overlay.png")

    i_target, j_target, d_target = project_full_las(Path(args.target_las), cam, R_tgt_inv, t_tgt_inv, W, H, args.chunk)
    depth_target = rasterize(W, H, i_target, j_target, d_target, reduce="min")
    save_overlay(gray, depth_target, out_dir / "target_fit_overlay.png")

    np.savez_compressed(
        out_dir / "transfer_projection.npz",
        i=i_transfer,
        j=j_transfer,
        depth=d_transfer,
        width=W,
        height=H,
        source_rotation_txt_to_las=R_src.astype(np.float32),
        source_translation_txt_to_las=t_src.astype(np.float32),
    )
    np.savez_compressed(
        out_dir / "target_fit_projection.npz",
        i=i_target,
        j=j_target,
        depth=d_target,
        width=W,
        height=H,
        target_rotation_txt_to_las=R_tgt.astype(np.float32),
        target_translation_txt_to_las=t_tgt.astype(np.float32),
    )

    print("Saved transfer comparison outputs.")
    print(f"Transfer mean abs (du,dv): {m_transfer['mean_abs_du']:.3f}, {m_transfer['mean_abs_dv']:.3f}")
    print(f"Target   mean abs (du,dv): {m_target['mean_abs_du']:.3f}, {m_target['mean_abs_dv']:.3f}")


if __name__ == "__main__":
    main()
