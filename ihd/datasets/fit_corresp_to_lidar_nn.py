import argparse
from pathlib import Path

import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from calibration_lidar_cylindrical import read_corresp


def parse_args():
    ap = argparse.ArgumentParser(
        description="Match correspondence XYZ to nearest LiDAR points and fit a rigid transform."
    )
    ap.add_argument("--corresp", required=True, help="Correspondence file with i j X Y Z rows")
    ap.add_argument("--las", required=True, help="LiDAR .las file")
    ap.add_argument("--out-dir", required=True, help="Directory for outputs")
    ap.add_argument("--las-sample-max", type=int, default=250000, help="Maximum LAS points to load for NN search")
    ap.add_argument("--plot-sample-max", type=int, default=20000, help="Maximum LAS points to show in plots")
    ap.add_argument("--las-chunk", type=int, default=1000000, help="Chunk size for LAS streaming")
    return ap.parse_args()


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
    if sampled:
        pts = np.concatenate(sampled, axis=0)
    else:
        pts = np.empty((0, 3), dtype=np.float64)
    if len(pts) > max_points:
        pts = pts[:max_points]
    return pts, total


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


def apply_transform(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ points.T).T + t.reshape(1, 3)


def save_summary(
    out_path: Path,
    corr_xyz: np.ndarray,
    nn_xyz: np.ndarray,
    d_before: np.ndarray,
    d_after: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    total_las: int,
    sampled_las: int,
) -> None:
    lines = [
        f"num_correspondences: {len(corr_xyz)}",
        f"lidar_total_points: {total_las}",
        f"lidar_sampled_for_nn: {sampled_las}",
        f"nn_distance_mean_before: {float(np.mean(d_before)):.6f}",
        f"nn_distance_median_before: {float(np.median(d_before)):.6f}",
        f"nn_distance_max_before: {float(np.max(d_before)):.6f}",
        f"residual_mean_after: {float(np.mean(d_after)):.6f}",
        f"residual_median_after: {float(np.median(d_after)):.6f}",
        f"residual_max_after: {float(np.max(d_after)):.6f}",
        "rotation_matrix:",
        *[" ".join(f"{v:.8f}" for v in row) for row in R],
        "translation:",
        " ".join(f"{v:.8f}" for v in t),
    ]
    out_path.write_text("\n".join(lines) + "\n")


def save_matches_csv(out_path: Path, corr_xyz: np.ndarray, nn_xyz: np.ndarray, d_before: np.ndarray, transformed: np.ndarray, d_after: np.ndarray) -> None:
    lines = ["idx,corr_x,corr_y,corr_z,nn_x,nn_y,nn_z,nn_dist_before,fit_x,fit_y,fit_z,residual_after"]
    for idx, (c, n, db, tf, da) in enumerate(zip(corr_xyz, nn_xyz, d_before, transformed, d_after)):
        lines.append(
            f"{idx},{c[0]},{c[1]},{c[2]},{n[0]},{n[1]},{n[2]},{db},{tf[0]},{tf[1]},{tf[2]},{da}"
        )
    out_path.write_text("\n".join(lines) + "\n")


def plot_alignment(out_path: Path, lidar_xyz: np.ndarray, corr_xyz: np.ndarray, nn_xyz: np.ndarray, transformed: np.ndarray) -> None:
    fig = plt.figure(figsize=(14, 6), dpi=160)

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(lidar_xyz[:, 0], lidar_xyz[:, 1], lidar_xyz[:, 2], s=0.6, c="lightgray", alpha=0.16, depthshade=False, label="LiDAR sample")
    ax1.scatter(corr_xyz[:, 0], corr_xyz[:, 1], corr_xyz[:, 2], s=34, c="red", depthshade=False, label="Corr XYZ")
    ax1.scatter(nn_xyz[:, 0], nn_xyz[:, 1], nn_xyz[:, 2], s=28, c="royalblue", marker="x", depthshade=False, label="Nearest LiDAR")
    for c, n in zip(corr_xyz, nn_xyz):
        ax1.plot([c[0], n[0]], [c[1], n[1]], [c[2], n[2]], color="orange", alpha=0.7, linewidth=1.0)
    ax1.set_title("Before rigid fit")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(lidar_xyz[:, 0], lidar_xyz[:, 1], lidar_xyz[:, 2], s=0.6, c="lightgray", alpha=0.16, depthshade=False, label="LiDAR sample")
    ax2.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2], s=34, c="green", depthshade=False, label="Corr XYZ after fit")
    ax2.scatter(nn_xyz[:, 0], nn_xyz[:, 1], nn_xyz[:, 2], s=28, c="royalblue", marker="x", depthshade=False, label="Matched LiDAR")
    for tf, n in zip(transformed, nn_xyz):
        ax2.plot([tf[0], n[0]], [tf[1], n[1]], [tf[2], n[2]], color="purple", alpha=0.7, linewidth=1.0)
    ax2.set_title("After rigid fit")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, _, corr_xyz = read_corresp(args.corresp)
    las_xyz, total_las = sample_las_points(Path(args.las), args.las_sample_max, args.las_chunk)
    if len(las_xyz) == 0:
        raise ValueError("No LiDAR points sampled.")

    tree = cKDTree(las_xyz)
    d_before, nn_idx = tree.query(corr_xyz, k=1)
    nn_xyz = las_xyz[nn_idx]

    R, t = fit_rigid_transform(corr_xyz, nn_xyz)
    transformed = apply_transform(corr_xyz, R, t)
    d_after = np.linalg.norm(transformed - nn_xyz, axis=1)

    plot_xyz = las_xyz if len(las_xyz) <= args.plot_sample_max else las_xyz[:args.plot_sample_max]
    plot_alignment(out_dir / "corr_nn_rigid_fit.png", plot_xyz, corr_xyz, nn_xyz, transformed)
    save_summary(out_dir / "summary.txt", corr_xyz, nn_xyz, d_before, d_after, R, t, total_las, len(las_xyz))
    save_matches_csv(out_dir / "matches.csv", corr_xyz, nn_xyz, d_before, transformed, d_after)

    print(f"Saved outputs to {out_dir}")
    print(f"Correspondences: {len(corr_xyz)}")
    print(f"LiDAR sample for NN: {len(las_xyz)} / {total_las}")
    print(f"Mean NN distance before fit: {float(np.mean(d_before)):.6f}")
    print(f"Mean residual after fit: {float(np.mean(d_after)):.6f}")


if __name__ == "__main__":
    main()
