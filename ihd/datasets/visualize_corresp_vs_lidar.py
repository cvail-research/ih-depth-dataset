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


def parse_args():
    ap = argparse.ArgumentParser(
        description="Visualize correspondence points in image space and against a sampled LiDAR cloud."
    )
    ap.add_argument("--corresp", required=True, help="Correspondence file with i j X Y Z rows")
    ap.add_argument("--cyl", required=True, help="Camera .cyl file used to reproject XYZ")
    ap.add_argument("--hsi-hdr", required=True, help="ENVI .hdr file for the scene image")
    ap.add_argument("--las", required=True, help="LiDAR .las file for the same scene")
    ap.add_argument("--out-dir", required=True, help="Directory for outputs")
    ap.add_argument("--las-sample-max", type=int, default=20000, help="Maximum LAS points to show in 3D")
    ap.add_argument("--las-chunk", type=int, default=1000000, help="Chunk size for LAS streaming")
    return ap.parse_args()


def load_gray_image(hdr_path: Path) -> np.ndarray:
    bsq_path = hdr_path.with_suffix(".bsq")
    img = spy.envi.open(str(hdr_path), str(bsq_path))
    cube = img.load()
    gray = cube.sum(axis=-1).astype(np.float64)
    gray -= gray.min()
    if gray.max() > 0:
        gray /= gray.max()
    return gray


def sample_las_points(las_path: Path, max_points: int, chunk_size: int) -> tuple[np.ndarray, int]:
    sampled = []
    with laspy.open(las_path) as lf:
        total = lf.header.point_count
        stride = max(1, total // max_points)
        global_idx = 0
        for chunk in lf.chunk_iterator(chunk_size):
            xyz = np.column_stack((chunk.x, chunk.y, chunk.z)).astype(np.float32)
            idx = np.arange(global_idx, global_idx + len(xyz))
            keep = (idx % stride) == 0
            if np.any(keep):
                sampled.append(xyz[keep])
            global_idx += len(xyz)
    if sampled:
        pts = np.concatenate(sampled, axis=0)
    else:
        pts = np.empty((0, 3), dtype=np.float32)
    if len(pts) > max_points:
        pts = pts[:max_points]
    return pts, total


def save_projection_csv(
    out_csv: Path,
    corr_i: np.ndarray,
    corr_j: np.ndarray,
    proj_ij: np.ndarray,
    corr_xyz: np.ndarray,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "corr_i", "corr_j", "proj_i", "proj_j", "error_px", "x", "y", "z"])
        for idx, (gt_i, gt_j, pred, xyz) in enumerate(zip(corr_i, corr_j, proj_ij, corr_xyz)):
            err = float(np.linalg.norm(pred - np.array([gt_i, gt_j], dtype=np.float64)))
            writer.writerow([idx, gt_i, gt_j, pred[0], pred[1], err, xyz[0], xyz[1], xyz[2]])


def plot_correspondences(
    gray: np.ndarray,
    corr_i: np.ndarray,
    corr_j: np.ndarray,
    proj_ij: np.ndarray,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 4), dpi=140)
    ax.imshow(gray, cmap="gray", interpolation="nearest")
    ax.scatter(corr_i, corr_j, s=42, facecolors="none", edgecolors="red", linewidths=1.4, label="corr 2D")
    ax.scatter(proj_ij[:, 0], proj_ij[:, 1], s=28, marker="x", c="cyan", linewidths=1.2, label="corr XYZ via .cyl")
    for gt_i, gt_j, pred in zip(corr_i, corr_j, proj_ij):
        ax.plot([gt_i, pred[0]], [gt_j, pred[1]], color="yellow", alpha=0.65, linewidth=0.9)
    ax.set_title("Correspondence image points vs reprojection of correspondence XYZ")
    ax.set_xlim(0, gray.shape[1] - 1)
    ax.set_ylim(gray.shape[0] - 1, 0)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_indexed_correspondences(
    gray: np.ndarray,
    corr_i: np.ndarray,
    corr_j: np.ndarray,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 4), dpi=160)
    ax.imshow(gray, cmap="gray", interpolation="nearest")
    ax.scatter(corr_i, corr_j, s=42, facecolors="none", edgecolors="red", linewidths=1.5)
    for idx, (ii, jj) in enumerate(zip(corr_i, corr_j)):
        ax.text(
            ii + 8,
            jj - 6,
            str(idx),
            color="yellow",
            fontsize=9,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.55, edgecolor="none"),
        )
    ax.set_title("Correspondence points with zero-based indices")
    ax.set_xlim(0, gray.shape[1] - 1)
    ax.set_ylim(gray.shape[0] - 1, 0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_lidar_vs_corresp(
    las_xyz: np.ndarray,
    corr_xyz: np.ndarray,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    if len(las_xyz):
        ax.scatter(
            las_xyz[:, 0],
            las_xyz[:, 1],
            las_xyz[:, 2],
            s=0.8,
            c="lightgray",
            alpha=0.18,
            depthshade=False,
            label="LiDAR sample",
        )
    ax.scatter(
        corr_xyz[:, 0],
        corr_xyz[:, 1],
        corr_xyz[:, 2],
        s=32,
        c="red",
        depthshade=False,
        label="Correspondence XYZ",
    )
    for idx, xyz in enumerate(corr_xyz):
        ax.text(xyz[0], xyz[1], xyz[2], str(idx), color="darkred", fontsize=7)
    ax.set_title("Correspondence XYZ vs sampled LiDAR cloud")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_summary(out_path: Path, corr_i: np.ndarray, corr_j: np.ndarray, proj_ij: np.ndarray, las_total: int, las_sampled: int) -> None:
    errors = np.linalg.norm(proj_ij - np.column_stack((corr_i, corr_j)), axis=1)
    lines = [
        f"num_correspondences: {len(corr_i)}",
        f"lidar_total_points: {las_total}",
        f"lidar_sampled_points: {las_sampled}",
        f"reprojection_rmse_px: {float(np.sqrt(np.mean(errors ** 2))):.6f}",
        f"reprojection_median_px: {float(np.median(errors)):.6f}",
        f"reprojection_max_px: {float(np.max(errors)):.6f}",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corr_i, corr_j, corr_xyz = read_corresp(args.corresp)
    cam = read_cam(args.cyl)
    proj_ij = project_vect_safe(corr_xyz, cam)
    gray = load_gray_image(Path(args.hsi_hdr))
    las_xyz, las_total = sample_las_points(Path(args.las), args.las_sample_max, args.las_chunk)

    save_projection_csv(out_dir / "corresp_reprojection.csv", corr_i, corr_j, proj_ij, corr_xyz)
    plot_indexed_correspondences(gray, corr_i, corr_j, out_dir / "corresp_indexed_on_image.png")
    plot_correspondences(gray, corr_i, corr_j, proj_ij, out_dir / "corresp_on_image.png")
    plot_lidar_vs_corresp(las_xyz, corr_xyz, out_dir / "corresp_vs_lidar_3d.png")
    save_summary(out_dir / "summary.txt", corr_i, corr_j, proj_ij, las_total, len(las_xyz))

    print(f"Saved outputs to {out_dir}")
    print(f"Correspondences: {len(corr_i)}")
    print(f"LiDAR sample: {len(las_xyz)} / {las_total}")


if __name__ == "__main__":
    main()
