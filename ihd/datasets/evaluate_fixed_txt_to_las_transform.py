import argparse
import math
from pathlib import Path

import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spectral as spy
from scipy.spatial import cKDTree

from calibration_lidar_cylindrical import read_corresp
from cylindrical_camera import project_vect_safe, read_cam
from register_lidar_cylindrical import depth_range, rasterize


def parse_vec(text: str, n: int) -> np.ndarray:
    vals = [float(x) for x in text.split(",")]
    if len(vals) != n:
        raise ValueError(f"Expected {n} comma-separated values, got {len(vals)}")
    return np.asarray(vals, dtype=np.float64)


def sample_las_points(las_path: Path, max_points: int, chunk_size: int) -> tuple[np.ndarray, int]:
    sampled = []
    with laspy.open(las_path) as lf:
        total = lf.header.point_count
        stride = max(1, total // max_points)
        global_idx = 0
        for chunk in lf.chunk_iterator(chunk_size):
            xyz = np.column_stack((chunk.x, chunk.y, chunk.z)).astype(np.float64)
            idx = np.arange(global_idx, global_idx + len(xyz))
            keep = (idx % stride) == 0
            if np.any(keep):
                sampled.append(xyz[keep])
            global_idx += len(xyz)
    pts = np.concatenate(sampled, axis=0) if sampled else np.empty((0, 3), dtype=np.float64)
    if len(pts) > max_points:
        pts = pts[:max_points]
    return pts, total


def load_gray(hdr_path: Path) -> np.ndarray:
    bsq = hdr_path.with_suffix(".bsq")
    img = spy.envi.open(str(hdr_path), str(bsq))
    cube = img.load()
    gray = cube.sum(axis=-1).astype(np.float64)
    gray -= gray.min()
    if gray.max() > 0:
        gray /= gray.max()
    return gray


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


def main():
    ap = argparse.ArgumentParser(description="Evaluate a fixed txt->las rigid transform on a scene.")
    ap.add_argument("--corresp", required=True)
    ap.add_argument("--las", required=True)
    ap.add_argument("--cyl", required=True)
    ap.add_argument("--hsi-hdr", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rotation", required=True, help="9 comma-separated values row-major for txt->las rotation")
    ap.add_argument("--translation", required=True, help="3 comma-separated values for txt->las translation")
    ap.add_argument("--las-sample-max", type=int, default=250000)
    ap.add_argument("--plot-sample-max", type=int, default=20000)
    ap.add_argument("--chunk", type=int, default=2_000_000)
    args = ap.parse_args()

    R_txt_to_las = parse_vec(args.rotation, 9).reshape(3, 3)
    t_txt_to_las = parse_vec(args.translation, 3)
    R_las_to_txt = R_txt_to_las.T
    t_las_to_txt = -(R_las_to_txt @ t_txt_to_las)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corr_i, corr_j, corr_xyz = read_corresp(args.corresp)
    transformed_txt = (R_txt_to_las @ corr_xyz.T).T + t_txt_to_las.reshape(1, 3)
    las_sample, total_las = sample_las_points(Path(args.las), args.las_sample_max, args.chunk)
    tree = cKDTree(las_sample)
    d_nn, _ = tree.query(transformed_txt, k=1)

    cam = read_cam(args.cyl)
    gray = load_gray(Path(args.hsi_hdr))
    H, W = gray.shape

    all_i = []
    all_j = []
    all_d = []
    with laspy.open(args.las) as lf:
        for chunk in lf.chunk_iterator(args.chunk):
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
            inside = (i_vals >= 0) & (i_vals < W) & (j_vals >= 0) & (j_vals < H)
            if not np.any(inside):
                continue
            all_i.append(i_vals[inside].astype(np.float32))
            all_j.append(j_vals[inside].astype(np.float32))
            all_d.append(d[inside].astype(np.float32))

    i_all = np.concatenate(all_i) if all_i else np.empty(0, dtype=np.float32)
    j_all = np.concatenate(all_j) if all_j else np.empty(0, dtype=np.float32)
    d_all = np.concatenate(all_d) if all_d else np.empty(0, dtype=np.float32)
    depth_img = rasterize(W, H, i_all, j_all, d_all, reduce="min")
    valid = np.isfinite(depth_img)

    save_overlay(gray, depth_img, out_dir / "projected_overlay.png")
    np.savez_compressed(
        out_dir / "projection.npz",
        i=i_all,
        j=j_all,
        depth=d_all,
        width=W,
        height=H,
        rotation_txt_to_las=R_txt_to_las.astype(np.float32),
        translation_txt_to_las=t_txt_to_las.astype(np.float32),
    )

    lines = [
        "rotation_txt_to_las:",
        *[" ".join(f"{v:.8f}" for v in row) for row in R_txt_to_las],
        f"translation_txt_to_las: {t_txt_to_las[0]:.8f} {t_txt_to_las[1]:.8f} {t_txt_to_las[2]:.8f}",
        f"num_correspondences: {len(corr_xyz)}",
        f"lidar_total_points: {total_las}",
        f"lidar_sampled_for_nn: {len(las_sample)}",
        f"nn_distance_mean: {float(np.mean(d_nn)):.6f}",
        f"nn_distance_median: {float(np.median(d_nn)):.6f}",
        f"nn_distance_max: {float(np.max(d_nn)):.6f}",
        f"projected_points_retained: {len(i_all)}",
        f"valid_pixel_coverage: {float(valid.sum() / (W * H)):.6f}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"Saved outputs to {out_dir}")
    print(f"NN mean distance: {float(np.mean(d_nn)):.6f}")
    print(f"Valid pixel coverage: {float(valid.sum() / (W * H)):.6%}")


if __name__ == "__main__":
    main()
