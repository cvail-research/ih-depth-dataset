from typing import Literal

import cv2
import numpy as np


Reduction = Literal["min", "median", "mean"]


def depth_range(points_cam: np.ndarray) -> np.ndarray:
    """Euclidean range from the camera origin for points in camera coordinates."""
    return np.linalg.norm(points_cam, axis=1)


def rasterize_projected_points(
    width: int,
    height: int,
    i: np.ndarray,
    j: np.ndarray,
    depth_m: np.ndarray,
    reduce: Reduction = "min",
) -> np.ndarray:
    """Rasterize projected point ranges into an image-sized depth map.

    The default `min` reduction is the canonical z-buffer behavior for this
    project: when multiple LiDAR points fall into the same image pixel, keep the
    closest range.
    """
    img = np.full((height, width), np.nan, dtype=np.float32)
    if len(depth_m) == 0:
        return img

    ui = np.floor(i).astype(np.int32)
    vj = np.floor(j).astype(np.int32)
    valid = (ui >= 0) & (ui < width) & (vj >= 0) & (vj < height) & np.isfinite(depth_m)
    if not np.any(valid):
        return img

    ui = ui[valid]
    vj = vj[valid]
    depth_m = depth_m[valid]
    pix = vj * width + ui

    if reduce == "min":
        order = np.lexsort((depth_m, pix))
        pix_sorted = pix[order]
        depth_sorted = depth_m[order]
        keep = np.ones_like(pix_sorted, dtype=bool)
        keep[1:] = pix_sorted[1:] != pix_sorted[:-1]
        y = (pix_sorted[keep] // width).astype(int)
        x = (pix_sorted[keep] % width).astype(int)
        img[y, x] = depth_sorted[keep]
        return img

    order = np.argsort(pix)
    pix_sorted = pix[order]
    depth_sorted = depth_m[order]
    unique, idx_start = np.unique(pix_sorted, return_index=True)
    vals = np.empty(len(unique), dtype=np.float32)
    for idx, _pix in enumerate(unique):
        start = idx_start[idx]
        end = idx_start[idx + 1] if idx + 1 < len(unique) else len(pix_sorted)
        segment = depth_sorted[start:end]
        if reduce == "median":
            vals[idx] = np.median(segment)
        elif reduce == "mean":
            vals[idx] = float(np.mean(segment))
        else:
            raise ValueError(f"Unsupported rasterization reduction: {reduce}")
    y = (unique // width).astype(int)
    x = (unique % width).astype(int)
    img[y, x] = vals
    return img


def suppress_far_occlusion_bleed(
    depth_img: np.ndarray,
    radius_px: int,
    min_depth_gap_m: float,
    min_depth_gap_ratio: float,
) -> np.ndarray:
    """Remove far returns that sit next to a much closer projected return.

    Exact-pixel z-buffering keeps the closest return only when points land in
    the same pixel. Sparse LiDAR projections can still leave far background
    returns immediately adjacent to foreground returns, which looks like depth
    bleeding at object boundaries. This filter suppresses only those far
    pixels; it does not densify the label map.
    """
    if radius_px <= 0:
        return depth_img
    if depth_img.size == 0:
        return depth_img

    finite = np.isfinite(depth_img)
    if not np.any(finite):
        return depth_img

    work = np.where(finite, depth_img, np.inf).astype(np.float32, copy=False)
    kernel_size = 2 * radius_px + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    local_min = cv2.erode(work, kernel)
    has_neighbor = np.isfinite(local_min)
    absolute_gap = depth_img - local_min
    relative_gap = absolute_gap / np.maximum(local_min, 1e-6)
    suppress = (
        finite
        & has_neighbor
        & (absolute_gap > float(min_depth_gap_m))
        & (relative_gap > float(min_depth_gap_ratio))
    )
    if not np.any(suppress):
        return depth_img
    filtered = depth_img.copy()
    filtered[suppress] = np.nan
    return filtered


def rasterize(
    width: int,
    height: int,
    i: np.ndarray,
    j: np.ndarray,
    depth_m: np.ndarray,
    reduce: Reduction = "min",
) -> np.ndarray:
    """Backward-compatible alias for rasterize_projected_points."""
    return rasterize_projected_points(width, height, i, j, depth_m, reduce=reduce)
