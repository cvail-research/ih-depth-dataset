import argparse
import copy
import json
import time
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import cKDTree

DEFAULT_PROFILE_NAME = "projection_sor50_2p0_voxel0p03"
DEFAULT_PROJECTION_USE_SOR = True
DEFAULT_PROJECTION_SOR_K = 50
DEFAULT_PROJECTION_SOR_STD_RATIO = 2.0
DEFAULT_PROJECTION_VOXEL = 0.03
DEFAULT_SOR_QUERY_BATCH_SIZE = 200_000
DEFAULT_PLATFORM_CENTER_X = 0.0
DEFAULT_PLATFORM_CENTER_Y = 0.0
DEFAULT_PLATFORM_RADIUS = None
DEFAULT_PLATFORM_Z_MIN = None
DEFAULT_PLATFORM_Z_MAX = None


def parse_exclusion_sphere(raw: str) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "Exclusion spheres must be formatted as x,y,z,radius"
        )
    try:
        x, y, z, radius = (float(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Exclusion sphere values must be numeric: x,y,z,radius"
        ) from exc
    if radius <= 0:
        raise argparse.ArgumentTypeError("Exclusion sphere radius must be positive")
    return x, y, z, radius


def parse_exclusion_box(raw: str) -> tuple[float, float, float, float, float, float]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            "Exclusion boxes must be formatted as x_min,x_max,y_min,y_max,z_min,z_max"
        )
    try:
        x_min, x_max, y_min, y_max, z_min, z_max = (float(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Exclusion box values must be numeric: x_min,x_max,y_min,y_max,z_min,z_max"
        ) from exc
    if x_min > x_max or y_min > y_max or z_min > z_max:
        raise argparse.ArgumentTypeError("Exclusion box min values must be <= max values")
    return x_min, x_max, y_min, y_max, z_min, z_max


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Create one cleaned display/projection LAS cloud from a raw scene LAS using optional "
            "range/z cropping, statistical outlier removal, and voxel representative-point downsampling."
        )
    )
    ap.add_argument("--las", required=True, help="Input raw LAS file")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--scene-label", help="Short label used in metadata")
    ap.add_argument(
        "--profile-name",
        default=DEFAULT_PROFILE_NAME,
        help="Name for the preprocessing profile stored in metadata",
    )
    ap.add_argument(
        "--sor-k",
        type=int,
        default=DEFAULT_PROJECTION_SOR_K,
        help="Default SOR neighbor count",
    )
    ap.add_argument(
        "--sor-std-ratio",
        type=float,
        default=DEFAULT_PROJECTION_SOR_STD_RATIO,
        help="Default SOR standard deviation ratio",
    )
    ap.add_argument(
        "--projection-sor-k",
        type=int,
        help="Optional separate SOR neighbor count for the projection cloud",
    )
    ap.add_argument(
        "--projection-sor-std-ratio",
        type=float,
        help="Optional separate SOR std ratio for the projection cloud",
    )
    ap.add_argument(
        "--projection-use-sor",
        action="store_true",
        help="Apply SOR to the projection cloud too.",
    )
    ap.add_argument(
        "--no-projection-use-sor",
        action="store_true",
        help="Disable projection-cloud SOR even if the default profile enables it.",
    )
    ap.add_argument(
        "--projection-voxel",
        type=float,
        default=DEFAULT_PROJECTION_VOXEL,
        help="Voxel size in meters for the cleaned display/projection cloud",
    )
    ap.add_argument("--range-min", type=float, help="Optional minimum Euclidean range in meters")
    ap.add_argument("--range-max", type=float, help="Optional maximum Euclidean range in meters")
    ap.add_argument("--z-min", type=float, help="Optional minimum z in meters")
    ap.add_argument("--z-max", type=float, help="Optional maximum z in meters")
    ap.add_argument(
        "--platform-center-x",
        type=float,
        default=DEFAULT_PLATFORM_CENTER_X,
        help="Scanner-frame x coordinate of the platform exclusion center.",
    )
    ap.add_argument(
        "--platform-center-y",
        type=float,
        default=DEFAULT_PLATFORM_CENTER_Y,
        help="Scanner-frame y coordinate of the platform exclusion center.",
    )
    ap.add_argument(
        "--platform-radius",
        type=float,
        default=DEFAULT_PLATFORM_RADIUS,
        help="Optional exclusion radius in meters around the LiDAR platform center.",
    )
    ap.add_argument(
        "--platform-z-min",
        type=float,
        default=DEFAULT_PLATFORM_Z_MIN,
        help="Optional minimum z for platform exclusion in meters.",
    )
    ap.add_argument(
        "--platform-z-max",
        type=float,
        default=DEFAULT_PLATFORM_Z_MAX,
        help="Optional maximum z for platform exclusion in meters.",
    )
    ap.add_argument(
        "--exclude-sphere",
        action="append",
        type=parse_exclusion_sphere,
        default=[],
        metavar="X,Y,Z,R",
        help=(
            "Optional scanner-frame spherical exclusion. Repeat for multiple "
            "platform components."
        ),
    )
    ap.add_argument(
        "--exclude-box",
        action="append",
        type=parse_exclusion_box,
        default=[],
        metavar="XMIN,XMAX,YMIN,YMAX,ZMIN,ZMAX",
        help=(
            "Optional scanner-frame box exclusion. Repeat for reusable platform "
            "or rig volumes."
        ),
    )
    return ap.parse_args()


def build_mask_from_optional_bounds(
    xyz: np.ndarray,
    range_min: float | None,
    range_max: float | None,
    z_min: float | None,
    z_max: float | None,
) -> np.ndarray:
    mask = np.all(np.isfinite(xyz), axis=1)
    if range_min is not None or range_max is not None:
        ranges = np.linalg.norm(xyz, axis=1)
        if range_min is not None:
            mask &= ranges >= range_min
        if range_max is not None:
            mask &= ranges <= range_max
    if z_min is not None:
        mask &= xyz[:, 2] >= z_min
    if z_max is not None:
        mask &= xyz[:, 2] <= z_max
    return mask


def build_platform_exclusion_mask(
    xyz: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float | None,
    z_min: float | None,
    z_max: float | None,
) -> np.ndarray:
    if radius is None and z_min is None and z_max is None:
        return np.zeros((xyz.shape[0],), dtype=bool)

    mask = np.ones((xyz.shape[0],), dtype=bool)
    if radius is not None:
        dx = xyz[:, 0] - center_x
        dy = xyz[:, 1] - center_y
        mask &= (dx * dx + dy * dy) <= (radius * radius)
    if z_min is not None:
        mask &= xyz[:, 2] >= z_min
    if z_max is not None:
        mask &= xyz[:, 2] <= z_max
    return mask


def build_sphere_exclusion_mask(
    xyz: np.ndarray,
    spheres: list[tuple[float, float, float, float]],
) -> np.ndarray:
    mask = np.zeros((xyz.shape[0],), dtype=bool)
    for cx, cy, cz, radius in spheres:
        center = np.array([cx, cy, cz], dtype=xyz.dtype)
        diff = xyz - center.reshape(1, 3)
        mask |= np.einsum("ij,ij->i", diff, diff) <= (radius * radius)
    return mask


def build_box_exclusion_mask(
    xyz: np.ndarray,
    boxes: list[tuple[float, float, float, float, float, float]],
) -> np.ndarray:
    mask = np.zeros((xyz.shape[0],), dtype=bool)
    for x_min, x_max, y_min, y_max, z_min, z_max in boxes:
        mask |= (
            (xyz[:, 0] >= x_min)
            & (xyz[:, 0] <= x_max)
            & (xyz[:, 1] >= y_min)
            & (xyz[:, 1] <= y_max)
            & (xyz[:, 2] >= z_min)
            & (xyz[:, 2] <= z_max)
        )
    return mask


def sor_inlier_mask(
    xyz: np.ndarray,
    k: int,
    std_ratio: float,
    batch_size: int = DEFAULT_SOR_QUERY_BATCH_SIZE,
) -> tuple[np.ndarray, dict[str, float]]:
    if xyz.shape[0] == 0:
        return np.zeros((0,), dtype=bool), {
            "sor_mean_neighbor_distance": float("nan"),
            "sor_std_neighbor_distance": float("nan"),
            "sor_threshold": float("nan"),
        }
    if xyz.shape[0] <= 2:
        return np.ones((xyz.shape[0],), dtype=bool), {
            "sor_mean_neighbor_distance": 0.0,
            "sor_std_neighbor_distance": 0.0,
            "sor_threshold": 0.0,
        }

    k_eff = min(k + 1, xyz.shape[0])
    tree = cKDTree(xyz)
    mean_neighbor_dist = np.empty((xyz.shape[0],), dtype=np.float32)
    for start in range(0, xyz.shape[0], batch_size):
        stop = min(start + batch_size, xyz.shape[0])
        dists, _ = tree.query(xyz[start:stop], k=k_eff, workers=1)
        if dists.ndim == 1:
            dists = dists[:, None]
        mean_neighbor_dist[start:stop] = dists[:, 1:].mean(axis=1, dtype=np.float32)
    mu = float(mean_neighbor_dist.mean())
    sigma = float(mean_neighbor_dist.std())
    threshold = mu + std_ratio * sigma
    keep = mean_neighbor_dist <= threshold
    return keep, {
        "sor_mean_neighbor_distance": mu,
        "sor_std_neighbor_distance": sigma,
        "sor_threshold": float(threshold),
    }


def voxel_representative_indices(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    if xyz.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    if voxel_size <= 0:
        return np.arange(xyz.shape[0], dtype=np.int64)

    xyz0 = xyz.min(axis=0, keepdims=True)
    grid = np.floor((xyz - xyz0) / voxel_size).astype(np.int64)
    grid_view = np.ascontiguousarray(grid).view(
        np.dtype((np.void, grid.dtype.itemsize * grid.shape[1]))
    )
    _, first_idx = np.unique(grid_view, return_index=True)
    first_idx.sort()
    return first_idx.astype(np.int64)


def write_subset_las(src_las: laspy.LasData, indices: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = copy.deepcopy(src_las.header)
    # Some IH LAS files contain non-ASCII VLR metadata that laspy cannot
    # re-serialize with its default LAS header writer. The workspace only needs
    # geometry, so keep the point format/scales and drop ancillary metadata.
    header.vlrs.clear()
    if header.evlrs is not None:
        header.evlrs.clear()
    out_las = laspy.LasData(header)
    out_las.points = src_las.points[indices].copy()
    out_las.write(out_path)


def main():
    args = parse_args()
    start = time.time()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    las_path = Path(args.las)
    scene_label = args.scene_label or las_path.stem
    projection_use_sor = DEFAULT_PROJECTION_USE_SOR
    if args.projection_use_sor:
        projection_use_sor = True
    if args.no_projection_use_sor:
        projection_use_sor = False
    projection_sor_k = (
        args.projection_sor_k if args.projection_sor_k is not None else DEFAULT_PROJECTION_SOR_K
    )
    projection_sor_std_ratio = (
        args.projection_sor_std_ratio
        if args.projection_sor_std_ratio is not None
        else DEFAULT_PROJECTION_SOR_STD_RATIO
    )

    las = laspy.read(las_path)
    xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float32, copy=False)

    mask_bounds = build_mask_from_optional_bounds(
        xyz,
        range_min=args.range_min,
        range_max=args.range_max,
        z_min=args.z_min,
        z_max=args.z_max,
    )
    platform_exclusion_mask = build_platform_exclusion_mask(
        xyz,
        center_x=args.platform_center_x,
        center_y=args.platform_center_y,
        radius=args.platform_radius,
        z_min=args.platform_z_min,
        z_max=args.platform_z_max,
    )
    sphere_exclusion_mask = build_sphere_exclusion_mask(xyz, args.exclude_sphere)
    box_exclusion_mask = build_box_exclusion_mask(xyz, args.exclude_box)
    combined_exclusion_mask = platform_exclusion_mask | sphere_exclusion_mask | box_exclusion_mask
    keep_mask = mask_bounds & (~combined_exclusion_mask)
    bound_idx = np.flatnonzero(keep_mask)
    xyz_bounds = xyz[bound_idx]

    if projection_use_sor:
        projection_mask_sor, projection_sor_stats = sor_inlier_mask(
            xyz_bounds,
            k=projection_sor_k,
            std_ratio=projection_sor_std_ratio,
        )
        projection_clean_idx = bound_idx[projection_mask_sor]
        xyz_projection_clean = xyz[projection_clean_idx]
    else:
        projection_clean_idx = bound_idx
        xyz_projection_clean = xyz_bounds
        projection_sor_stats = {
            "projection_sor_mean_neighbor_distance": float("nan"),
            "projection_sor_std_neighbor_distance": float("nan"),
            "projection_sor_threshold": float("nan"),
        }

    projection_rel_idx = voxel_representative_indices(xyz_projection_clean, args.projection_voxel)
    projection_idx = projection_clean_idx[projection_rel_idx]

    projection_out = out_dir / f"{las_path.stem}_projection_clean.las"
    write_subset_las(las, projection_idx, projection_out)

    processing_seconds = time.time() - start
    summary = {
        "scene_label": scene_label,
        "input_las": str(las_path),
        "projection_las": str(projection_out),
        "profile_name": args.profile_name,
        "num_input_points": int(xyz.shape[0]),
        "num_after_bounds": int(bound_idx.shape[0]),
        "num_removed_platform_points": int(platform_exclusion_mask.sum()),
        "num_removed_exclusion_sphere_points": int(sphere_exclusion_mask.sum()),
        "num_removed_exclusion_box_points": int(box_exclusion_mask.sum()),
        "num_removed_total_exclusion_points": int(combined_exclusion_mask.sum()),
        "num_after_projection_cleaning": int(projection_clean_idx.shape[0]),
        "num_projection_points_final": int(projection_idx.shape[0]),
        "projection_fraction_of_input": float(projection_idx.shape[0] / max(1, xyz.shape[0])),
        "processing_seconds": float(processing_seconds),
        "preprocessing": {
            "range_min": args.range_min,
            "range_max": args.range_max,
            "z_min": args.z_min,
            "z_max": args.z_max,
            "platform_center_x": args.platform_center_x,
            "platform_center_y": args.platform_center_y,
            "platform_radius": args.platform_radius,
            "platform_z_min": args.platform_z_min,
            "platform_z_max": args.platform_z_max,
            "exclude_spheres": [
                {
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "radius": float(radius),
                }
                for x, y, z, radius in args.exclude_sphere
            ],
            "exclude_boxes": [
                {
                    "x_min": float(x_min),
                    "x_max": float(x_max),
                    "y_min": float(y_min),
                    "y_max": float(y_max),
                    "z_min": float(z_min),
                    "z_max": float(z_max),
                }
                for x_min, x_max, y_min, y_max, z_min, z_max in args.exclude_box
            ],
            "sor_k": int(args.sor_k),
            "sor_std_ratio": float(args.sor_std_ratio),
            "projection_use_sor": bool(projection_use_sor),
            "projection_sor_k": int(projection_sor_k),
            "projection_sor_std_ratio": float(projection_sor_std_ratio),
            "projection_voxel": float(args.projection_voxel),
        },
        **projection_sor_stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
