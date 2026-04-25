import argparse
from pathlib import Path

import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from calibration_lidar_cylindrical import read_corresp


GLOBAL_R = np.array([
    [0.0, -1.0, 0.0],
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0],
], dtype=np.float64)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Apply a fixed txt->las rotation and compare against the LiDAR cloud."
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
    pts = np.concatenate(sampled, axis=0) if sampled else np.empty((0, 3), dtype=np.float64)
    if len(pts) > max_points:
        pts = pts[:max_points]
    return pts, total


def save_summary(out_path: Path, d_before: np.ndarray, d_after: np.ndarray, total_las: int, sampled_las: int) -> None:
    lines = [
        f"rotation_matrix:",
        *[" ".join(f"{v:.8f}" for v in row) for row in GLOBAL_R],
        f"lidar_total_points: {total_las}",
        f"lidar_sampled_for_nn: {sampled_las}",
        f"nn_distance_mean_before: {float(np.mean(d_before)):.6f}",
        f"nn_distance_median_before: {float(np.median(d_before)):.6f}",
        f"nn_distance_max_before: {float(np.max(d_before)):.6f}",
        f"nn_distance_mean_after_rotation: {float(np.mean(d_after)):.6f}",
        f"nn_distance_median_after_rotation: {float(np.median(d_after)):.6f}",
        f"nn_distance_max_after_rotation: {float(np.max(d_after)):.6f}",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def save_matches_csv(out_path: Path, corr_xyz: np.ndarray, rotated_xyz: np.ndarray, d_before: np.ndarray, d_after: np.ndarray) -> None:
    lines = ["idx,corr_x,corr_y,corr_z,rot_x,rot_y,rot_z,nn_dist_before,nn_dist_after_rotation"]
    for idx, (c, r, db, da) in enumerate(zip(corr_xyz, rotated_xyz, d_before, d_after)):
        lines.append(f"{idx},{c[0]},{c[1]},{c[2]},{r[0]},{r[1]},{r[2]},{db},{da}")
    out_path.write_text("\n".join(lines) + "\n")


def plot_alignment(out_path: Path, lidar_xyz: np.ndarray, corr_xyz: np.ndarray, rotated_xyz: np.ndarray) -> None:
    fig = plt.figure(figsize=(14, 6), dpi=160)

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(lidar_xyz[:, 0], lidar_xyz[:, 1], lidar_xyz[:, 2], s=0.6, c="lightgray", alpha=0.16, depthshade=False, label="LiDAR sample")
    ax1.scatter(corr_xyz[:, 0], corr_xyz[:, 1], corr_xyz[:, 2], s=34, c="red", depthshade=False, label=".txt XYZ")
    ax1.set_title("Before fixed rotation")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(lidar_xyz[:, 0], lidar_xyz[:, 1], lidar_xyz[:, 2], s=0.6, c="lightgray", alpha=0.16, depthshade=False, label="LiDAR sample")
    ax2.scatter(rotated_xyz[:, 0], rotated_xyz[:, 1], rotated_xyz[:, 2], s=34, c="green", depthshade=False, label="R * .txt XYZ")
    ax2.set_title("After fixed rotation")
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

    rotated_xyz = (GLOBAL_R @ corr_xyz.T).T

    tree = cKDTree(las_xyz)
    d_before, _ = tree.query(corr_xyz, k=1)
    d_after, _ = tree.query(rotated_xyz, k=1)

    plot_xyz = las_xyz if len(las_xyz) <= args.plot_sample_max else las_xyz[:args.plot_sample_max]
    plot_alignment(out_dir / "txt_to_las_fixed_rotation.png", plot_xyz, corr_xyz, rotated_xyz)
    save_summary(out_dir / "summary.txt", d_before, d_after, total_las, len(las_xyz))
    save_matches_csv(out_dir / "matches.csv", corr_xyz, rotated_xyz, d_before, d_after)

    print(f"Saved outputs to {out_dir}")
    print(f"Correspondences: {len(corr_xyz)}")
    print(f"LiDAR sample for NN: {len(las_xyz)} / {total_las}")
    print(f"Mean NN distance before: {float(np.mean(d_before)):.6f}")
    print(f"Mean NN distance after rotation: {float(np.mean(d_after)):.6f}")


if __name__ == "__main__":
    main()
