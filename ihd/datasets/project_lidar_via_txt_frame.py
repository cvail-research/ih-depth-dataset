import argparse
import math
from pathlib import Path

import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spectral as spy

from cylindrical_camera import project_vect_safe, read_cam
from register_lidar_cylindrical import depth_range, rasterize


TXT_TO_LAS_R = np.array([
    [0.0, -1.0, 0.0],
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0],
], dtype=np.float64)

LAS_TO_TXT_R = TXT_TO_LAS_R.T


def parse_args():
    ap = argparse.ArgumentParser(
        description="Project LiDAR after mapping it into the txt/.cyl frame with a fixed inverse rotation."
    )
    ap.add_argument("--cyl", required=True, help="Calibrated camera (.cyl)")
    ap.add_argument("--las", required=True, help="LiDAR point cloud (.las)")
    ap.add_argument("--hsi-hdr", required=True, help="ENVI .hdr file")
    ap.add_argument("--overlay-png", required=True, help="Output overlay PNG")
    ap.add_argument("--export-npz", required=True, help="Output projection NPZ")
    ap.add_argument("--chunk", type=int, default=2_000_000, help="LAS chunk size")
    ap.add_argument("--reduce", choices=["min", "median", "mean"], default="min")
    ap.add_argument("--tx", type=float, default=0.0, help="Optional translation in txt frame x after rotation")
    ap.add_argument("--ty", type=float, default=0.0, help="Optional translation in txt frame y after rotation")
    ap.add_argument("--tz", type=float, default=0.0, help="Optional translation in txt frame z after rotation")
    return ap.parse_args()


def load_gray(hdr_path: Path):
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
    if not np.any(mask_valid):
        raise ValueError("No valid depth pixels for overlay.")

    d_min = float(np.nanmin(depth_img[mask_valid]))
    d_max = float(np.nanmax(depth_img[mask_valid]))
    if d_max <= d_min:
        d_max = d_min + 1e-6

    H, W = gray.shape
    dpi = 100
    cb_px = 20
    fig_w_in = W / dpi
    fig_h_in = (H + cb_px) / dpi
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)

    img_h_rel = H / (H + cb_px)
    cb_h_rel = cb_px / (H + cb_px)
    ax_img = fig.add_axes([0.0, 0.0, 1.0, img_h_rel])
    ax_cb = fig.add_axes([0.0, img_h_rel, 1.0, cb_h_rel])

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
    ax_img.set_xlim(0, W)
    ax_img.set_ylim(H, 0)
    ax_img.axis("off")

    gradient = np.linspace(d_min, d_max, max(2, W), dtype=np.float32)[None, :]
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
    inset_px = 25
    span = d_max - d_min
    inset_data = span * (inset_px / W)
    ticks = [d_min + inset_data, d_max - inset_data]
    ax_cb.set_xticks(ticks)
    ax_cb.set_xticklabels([f"{d_min:.0f} m", f"{d_max:.0f} m"])
    ax_cb.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, length=0, pad=2)
    for spine in ax_cb.spines.values():
        spine.set_visible(False)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def main():
    args = parse_args()
    cam = read_cam(args.cyl)
    hdr = Path(args.hsi_hdr)
    gray = load_gray(hdr)
    H, W = gray.shape

    print(f"HSI dims: W={W} H={H}")
    print(f"Native cylindrical width (2π/y): {(2 * math.pi) / cam.y:.1f} px")

    t_txt = np.array([args.tx, args.ty, args.tz], dtype=np.float64)
    all_i = []
    all_j = []
    all_d = []
    all_xyz_txt = []

    with laspy.open(args.las) as lf:
        total = lf.header.point_count
        print(f"LAS points: {total}")
        for chunk in lf.chunk_iterator(args.chunk):
            xyz_las = np.column_stack((chunk.x, chunk.y, chunk.z)).astype(np.float64)
            xyz_txt = (LAS_TO_TXT_R @ xyz_las.T).T + t_txt.reshape(1, 3)
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
            xyz_txt = xyz_txt[finite]

            inside = (i_vals >= 0) & (i_vals < W) & (j_vals >= 0) & (j_vals < H)
            if not np.any(inside):
                continue
            all_i.append(i_vals[inside].astype(np.float32))
            all_j.append(j_vals[inside].astype(np.float32))
            all_d.append(d[inside].astype(np.float32))
            all_xyz_txt.append(xyz_txt[inside].astype(np.float32))

    i_all = np.concatenate(all_i) if all_i else np.empty(0, dtype=np.float32)
    j_all = np.concatenate(all_j) if all_j else np.empty(0, dtype=np.float32)
    d_all = np.concatenate(all_d) if all_d else np.empty(0, dtype=np.float32)
    xyz_all = np.concatenate(all_xyz_txt) if all_xyz_txt else np.empty((0, 3), dtype=np.float32)

    print(f"Projected points retained: {len(i_all)}")
    if len(i_all):
        print(f"i range: [{i_all.min():.1f},{i_all.max():.1f}]")
        print(f"j range: [{j_all.min():.1f},{j_all.max():.1f}]")

    depth_img = rasterize(W, H, i_all, j_all, d_all, reduce=args.reduce)
    valid = np.isfinite(depth_img)
    print(f"Valid pixels: {valid.sum()} / {W * H} ({valid.sum() / (W * H):.2%})")

    save_overlay(gray, depth_img, Path(args.overlay_png))
    out_npz = Path(args.export_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        i=i_all,
        j=j_all,
        depth=d_all,
        xyz_txt=xyz_all,
        depth_img=depth_img.astype(np.float32),
        las_to_txt_rotation=LAS_TO_TXT_R.astype(np.float32),
        txt_translation=t_txt.astype(np.float32),
        width=W,
        height=H,
    )
    print(f"Saved overlay: {args.overlay_png}")
    print(f"Saved NPZ: {out_npz}")


if __name__ == "__main__":
    main()
