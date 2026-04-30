import argparse
import json
from pathlib import Path

import cv2
import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spectral as spy

from ihd.datasets.cylindrical_camera import project_vect_safe, read_cam


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Render a LiDAR depth overlay from an annotation workspace fit and a chosen LAS file."
    )
    ap.add_argument("--workspace-dir", required=True, help="Annotation workspace scene directory")
    ap.add_argument("--las", required=True, help="Preprocessed LAS to project")
    ap.add_argument("--out", required=True, help="Output overlay PNG")
    ap.add_argument("--npz-out", help="Optional projected point/depth NPZ")
    ap.add_argument(
        "--occlusion-filter-radius-px",
        type=int,
        default=0,
        help="Suppress far depth pixels if a much closer pixel exists within this radius. Disabled at 0.",
    )
    ap.add_argument(
        "--occlusion-min-depth-gap-m",
        type=float,
        default=1.0,
        help="Minimum absolute range gap required for occlusion suppression.",
    )
    ap.add_argument(
        "--occlusion-min-depth-gap-ratio",
        type=float,
        default=0.05,
        help="Minimum relative range gap required for occlusion suppression, e.g. 0.05 = 5%%.",
    )
    ap.add_argument(
        "--title-mode",
        choices=["none", "auto", "custom"],
        default="none",
        help="Whether to render a title strip above the overlay.",
    )
    ap.add_argument("--title-text", help="Title text when --title-mode=custom")
    return ap.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def resolve_local_artifact(workspace_dir: Path, fit_data: dict, key: str, fallback_name: str) -> Path:
    local = workspace_dir / fallback_name
    if local.exists():
        return local
    raw = fit_data.get(key)
    if raw and Path(raw).exists():
        return Path(raw)
    raise FileNotFoundError(f"Could not resolve {key} for {workspace_dir}")


def load_gray(workspace_dir: Path, scene_data: dict) -> np.ndarray:
    preview = workspace_dir / "image_preview.png"
    if preview.exists():
        img = cv2.imread(str(preview), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image preview: {preview}")
        return img.astype(np.float64) / 255.0

    hdr_raw = scene_data.get("source_paths", {}).get("hsi_hdr")
    if not hdr_raw:
        raise FileNotFoundError(f"No image_preview.png or hsi_hdr found for {workspace_dir}")
    hdr = Path(hdr_raw)
    bsq = hdr.with_suffix(".bsq")
    img = spy.envi.open(str(hdr), str(bsq))
    cube = img.load()
    gray = cube.sum(axis=-1).astype(np.float64)
    gray -= gray.min()
    if gray.max() > 0:
        gray /= gray.max()
    return gray


def depth_range(points_cam: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points_cam, axis=1)


def rasterize(width: int, height: int, i: np.ndarray, j: np.ndarray, d: np.ndarray) -> np.ndarray:
    img = np.full((height, width), np.nan, dtype=np.float32)
    if len(d) == 0:
        return img
    ui = np.floor(i).astype(np.int32)
    vj = np.floor(j).astype(np.int32)
    valid = (ui >= 0) & (ui < width) & (vj >= 0) & (vj < height) & np.isfinite(d)
    if not np.any(valid):
        return img
    ui = ui[valid]
    vj = vj[valid]
    d = d[valid]
    pix = vj * width + ui
    order = np.lexsort((d, pix))
    pix = pix[order]
    d = d[order]
    keep = np.ones_like(pix, dtype=bool)
    keep[1:] = pix[1:] != pix[:-1]
    y = (pix[keep] // width).astype(int)
    x = (pix[keep] % width).astype(int)
    img[y, x] = d[keep]
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


def save_overlay(gray: np.ndarray, depth_img: np.ndarray, out_path: Path, title: str | None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask_valid = np.isfinite(depth_img)
    if not np.any(mask_valid):
        raise ValueError("No valid depth pixels for overlay.")

    d_min = float(np.nanmin(depth_img[mask_valid]))
    d_max = float(np.nanmax(depth_img[mask_valid]))
    if d_max <= d_min:
        d_max = d_min + 1e-6

    height, width = gray.shape
    dpi = 100
    cb_px = 22
    title_px = 18 if title else 0
    total_h = height + cb_px + title_px
    fig = plt.figure(figsize=(width / dpi, total_h / dpi), dpi=dpi)

    if title:
        ax_title = fig.add_axes([0.0, (height + cb_px) / total_h, 1.0, title_px / total_h])
        ax_title.axis("off")
        ax_title.text(0.5, 0.5, title, ha="center", va="center", fontsize=9)

    ax_img = fig.add_axes([0.0, 0.0, 1.0, height / total_h])
    ax_cb = fig.add_axes([0.0, height / total_h, 1.0, cb_px / total_h])

    ax_img.imshow(gray, cmap="gray", interpolation="nearest")
    yv, xv = np.nonzero(mask_valid)
    ax_img.scatter(
        xv,
        yv,
        c=depth_img[mask_valid],
        s=1,
        cmap="viridis_r",
        vmin=d_min,
        vmax=d_max,
        marker="s",
        linewidths=0,
    )
    ax_img.set_xlim(0, width)
    ax_img.set_ylim(height, 0)
    ax_img.axis("off")

    gradient = np.linspace(d_min, d_max, max(2, width), dtype=np.float32)[None, :]
    ax_cb.imshow(
        gradient,
        aspect="auto",
        cmap="viridis_r",
        vmin=d_min,
        vmax=d_max,
        extent=[d_min, d_max, 0, 1],
    )
    ax_cb.set_xlim(d_min, d_max)
    ax_cb.set_yticks([])
    span = d_max - d_min
    inset = max(span * 0.015, 1e-6)
    ax_cb.set_xticks([d_min + inset, d_max - inset])
    labels = ax_cb.set_xticklabels([f"{int(round(d_min))} m", f"{int(round(d_max))} m"])
    if len(labels) == 2:
        labels[0].set_ha("left")
        labels[1].set_ha("right")
    ax_cb.tick_params(
        axis="x",
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
        length=0,
        pad=2,
        labelsize=8,
    )
    for spine in ax_cb.spines.values():
        spine.set_visible(False)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def auto_title(scene_data: dict, workspace_dir: Path) -> str:
    path_key = scene_data.get("path_key")
    step = scene_data.get("step")
    if path_key and step is not None:
        return f"{path_key} Step{int(step)}"
    return workspace_dir.name


def project_las(las_path: Path, cam, fit_data: dict, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    las = laspy.read(las_path)
    xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64)

    if mode == "existing_cyl":
        rot = np.asarray(fit_data["las_to_txt_rotation"], dtype=np.float64)
        trans = np.asarray(fit_data["las_to_txt_translation"], dtype=np.float64)
        xyz_project = (rot @ xyz.T).T + trans.reshape(1, 3)
    elif mode == "generated_cyl":
        rot = fit_data.get("las_to_reference_rotation")
        trans = fit_data.get("las_to_reference_translation")
        if rot is not None and trans is not None:
            rot = np.asarray(rot, dtype=np.float64)
            trans = np.asarray(trans, dtype=np.float64)
            xyz_project = (rot @ xyz.T).T + trans.reshape(1, 3)
        else:
            xyz_project = xyz
    else:
        raise ValueError(f"Unsupported fit mode: {mode}")

    pc = (cam.Rot @ xyz_project.T).T + cam.t.reshape(1, 3)
    d = depth_range(pc)
    ij = project_vect_safe(xyz_project, cam)
    i_vals = ij[:, 0]
    j_vals = ij[:, 1]
    valid = np.isfinite(i_vals) & np.isfinite(j_vals) & np.isfinite(d)
    return i_vals[valid].astype(np.float32), j_vals[valid].astype(np.float32), d[valid].astype(np.float32)


def main() -> None:
    args = parse_args()
    workspace_dir = Path(args.workspace_dir)
    fit_data = read_json(workspace_dir / "fit.json")
    scene_data = read_json(workspace_dir / "scene.json")
    mode = fit_data.get("mode")
    if not bool(fit_data.get("ready")):
        raise ValueError(f"Workspace fit is not ready: {workspace_dir}")

    cyl_path = resolve_local_artifact(workspace_dir, fit_data, "fitted_cyl", "fitted.cyl")
    cam = read_cam(str(cyl_path))
    gray = load_gray(workspace_dir, scene_data)
    height, width = gray.shape

    i_vals, j_vals, d = project_las(Path(args.las), cam, fit_data, mode)
    inside = (i_vals >= 0) & (i_vals < width) & (j_vals >= 0) & (j_vals < height)
    i_vals = i_vals[inside]
    j_vals = j_vals[inside]
    d = d[inside]
    depth_img = rasterize(width, height, i_vals, j_vals, d)
    depth_img = suppress_far_occlusion_bleed(
        depth_img,
        args.occlusion_filter_radius_px,
        args.occlusion_min_depth_gap_m,
        args.occlusion_min_depth_gap_ratio,
    )

    title = None
    if args.title_mode == "auto":
        title = auto_title(scene_data, workspace_dir)
    elif args.title_mode == "custom":
        if not args.title_text:
            raise ValueError("--title-text is required when --title-mode=custom")
        title = args.title_text

    save_overlay(gray, depth_img, Path(args.out), title)
    if args.npz_out:
        npz_path = Path(args.npz_out)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_path,
            i=i_vals,
            j=j_vals,
            depth=d,
            depth_img=depth_img.astype(np.float32),
            occlusion_filter_radius_px=args.occlusion_filter_radius_px,
            occlusion_min_depth_gap_m=args.occlusion_min_depth_gap_m,
            occlusion_min_depth_gap_ratio=args.occlusion_min_depth_gap_ratio,
            width=width,
            height=height,
            workspace_dir=str(workspace_dir),
            las=str(args.las),
            mode=mode,
        )
    print(f"Saved overlay: {args.out}")
    print(f"Projected points retained: {len(i_vals)}")
    print(f"Valid pixels: {int(np.isfinite(depth_img).sum())} / {width * height}")


if __name__ == "__main__":
    main()
