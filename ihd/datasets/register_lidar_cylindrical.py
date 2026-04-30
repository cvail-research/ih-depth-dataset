import argparse, sys, math
from pathlib import Path
import numpy as np, laspy, spectral as spy
from cylindrical_camera import read_cam, project_vect_safe
import shutil
from ihd.datasets.depth_rasterization import depth_range, rasterize


def parse_args():
    ap = argparse.ArgumentParser(
        "Project LiDAR into cylindrical image and (optionally) rasterize.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--cyl", required=True, help="Calibrated camera (.cyl)")
    ap.add_argument("--las", required=True, help="LiDAR point cloud (.las/.laz)")
    ap.add_argument("--hsi-hdr", required=True, help="ENVI .hdr (derives W,H and grayscale)")
    ap.add_argument("--chunk", type=int, default=2_000_000, help="LiDAR chunk size (points)")
    ap.add_argument("--reduce", choices=["min","median","mean"], default="min",
                    help="Per-pixel reduction when multiple points land in same pixel")
    ap.add_argument("--depth-map", help="Output 16-bit PNG depth map (range * 256)")
    ap.add_argument("--overlay-png", help="PNG overlay of depth on grayscale HSI")
    ap.add_argument("--export-npz", help="Save projected (i,j,depth,xyz) + depth image (if created)")
    ap.add_argument("--export-label", help="Save minimal label NPZ (depth float32 + mask uint8)")
    ap.add_argument("--copy-las", help="Copy input LAS/LAZ to this file or directory (for dataset packaging)")
    ap.add_argument("--copy-cyl", help="Copy input cylindrical camera file (.cyl) to this file or directory")
    ap.add_argument("--progress", action="store_true", help="Show progress bar")
    ap.add_argument("--stats-only", action="store_true", help="Only print stats (still creates outputs if specified)")
    ap.add_argument("--depth-min", type=float,
                    help="Optional min depth (meters) for depth_map color scaling (default = data min)")
    ap.add_argument("--depth-max", type=float,
                    help="Optional max depth (meters) for depth_map color scaling (default = data max)")
    return ap.parse_args()


def main():
    args = parse_args()
    cam = read_cam(args.cyl)

    # Load HSI (width/height + grayscale)
    hdr = Path(args.hsi_hdr); bsq = hdr.with_suffix('.bsq')
    if not hdr.exists() or not bsq.exists():
        print("HSI files missing.", file=sys.stderr); return 2
    img = spy.envi.open(str(hdr), str(bsq))
    cube = img.load()
    gray = cube.sum(axis=-1).astype(np.float64)
    gray -= gray.min()
    if gray.max()>0: gray /= gray.max()
    H,W = gray.shape
    print(f"HSI dims: W={W} H={H}")

    # Native cylindrical width (diagnostic)
    native_w = (2*math.pi)/cam.y
    print(f"Native cylindrical width (2π/y): {native_w:.1f} px (image uses first {W} columns)")

    # Stream project
    all_i=[]; all_j=[]; all_d=[]; all_xyz=[]
    use_pb = args.progress
    if use_pb:
        try:
            from tqdm import tqdm
        except ImportError:
            print("tqdm not installed; disable --progress.", file=sys.stderr)
            use_pb=False
    with laspy.open(args.las) as lf:
        total = lf.header.point_count
        it = lf.chunk_iterator(args.chunk)
        if use_pb: 
            import math as _m
            it = tqdm(it, total=max(1,_m.ceil(total/args.chunk)), desc="Projecting")
        for ch in it:
            xyz = np.vstack([ch.x, ch.y, ch.z]).T.astype(np.float64)
            Pc = (cam.Rot @ xyz.T).T + cam.t.reshape(1,3)
            d = depth_range(Pc)
            ij = project_vect_safe(xyz, cam)
            i_vals = ij[:,0]; j_vals = ij[:,1]
            finite = np.isfinite(i_vals) & np.isfinite(j_vals)
            i_vals=i_vals[finite]; j_vals=j_vals[finite]; d=d[finite]; xyz=xyz[finite]
            # Keep only those falling into current HSI window
            inside = (i_vals>=0)&(i_vals<W)&(j_vals>=0)&(j_vals<H)
            i_vals=i_vals[inside]; j_vals=j_vals[inside]; d=d[inside]; xyz=xyz[inside]
            all_i.append(i_vals.astype(np.float32))
            all_j.append(j_vals.astype(np.float32))
            all_d.append(d.astype(np.float32))
            all_xyz.append(xyz.astype(np.float32))

    i_all = np.concatenate(all_i) if all_i else np.empty(0,np.float32)
    j_all = np.concatenate(all_j) if all_j else np.empty(0,np.float32)
    d_all = np.concatenate(all_d) if all_d else np.empty(0,np.float32)
    xyz_all = np.concatenate(all_xyz) if all_xyz else np.empty((0,3),np.float32)

    print(f"Projected points retained: {len(i_all)}")
    if len(i_all):
        span_i = i_all.max()-i_all.min()
        print(f"i range: [{i_all.min():.1f},{i_all.max():.1f}] span={span_i:.1f}")
        print(f"j range: [{j_all.min():.1f},{j_all.max():.1f}]")
        coverage_est = len(np.unique(np.floor(j_all)*W + np.floor(i_all)))/(W*H)
        print(f"Approx pixel coverage (before reduction): {coverage_est:.2%}")

    # Decide if we need a raster (requested by any output that uses it)
    depth_img = None
    need_raster = bool(args.depth_map or args.overlay_png or args.export_npz or args.export_label)
    if need_raster:
        print("Rasterizing (triggered by requested outputs)...")
        depth_img = rasterize(W, H, i_all, j_all, d_all, reduce=args.reduce)
        valid = np.isfinite(depth_img)
        cov = valid.sum()/(W*H)
        print(f"Valid pixels: {valid.sum()} / {W*H} ({cov:.2%})")
        if args.depth_map:
            try:
                import matplotlib
                import matplotlib.pyplot as plt
                import numpy as _np

                Path(args.depth_map).parent.mkdir(parents=True, exist_ok=True)

                mask_valid = np.isfinite(depth_img)
                if not _np.any(mask_valid):
                    print("No valid depth pixels; skipping depth_map.", file=sys.stderr)
                else:
                    data_min = float(_np.nanmin(depth_img[mask_valid]))
                    data_max = float(_np.nanmax(depth_img[mask_valid]))

                    # Determine clipping range (user override or auto)
                    d_min_clip = args.depth_min if args.depth_min is not None else data_min
                    d_max_clip = args.depth_max if args.depth_max is not None else data_max

                    # Handle inverted / degenerate ranges
                    if d_max_clip <= d_min_clip:
                        d_max_clip = d_min_clip + 1e-6

                    range_source = []
                    range_source.append("user-min" if args.depth_min is not None else "auto-min")
                    range_source.append("user-max" if args.depth_max is not None else "auto-max")
                    range_source = "/".join(range_source)

                    depth_plot = depth_img.copy()
                    depth_plot[~mask_valid] = _np.nan
                    depth_display = _np.clip(depth_plot, d_min_clip, d_max_clip)

                    # Colormap
                    cmap = matplotlib.colormaps['viridis_r'].copy()
                    cmap.set_bad(color='black')

                    # Layout: allow user image to fill figure; colorbar 95% width
                    dpi = 100
                    cb_px = 20
                    fig_w_in = W / dpi
                    fig_h_in = (H + cb_px) / dpi
                    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)

                    img_h_rel = H / (H + cb_px)
                    cb_h_rel = cb_px / (H + cb_px)
                    cb_width_factor = 1
                    cb_x0 = (1.0 - cb_width_factor) / 2.0
                    cb_y0 = img_h_rel
                    ax_img = fig.add_axes([0.0, 0.0, 1.0, img_h_rel])
                    ax_cb  = fig.add_axes([cb_x0, cb_y0, cb_width_factor, cb_h_rel])

                    ax_img.imshow(
                        depth_display,
                        cmap=cmap,
                        vmin=d_min_clip,
                        vmax=d_max_clip,
                        interpolation='nearest'
                    )
                    ax_img.axis('off')

                    grad_w = max(2, int(W * cb_width_factor))
                    gradient = _np.linspace(d_min_clip, d_max_clip, grad_w, dtype=_np.float32)[None, :]
                    ax_cb.imshow(
                        gradient,
                        aspect='auto',
                        cmap=cmap,
                        vmin=d_min_clip,
                        vmax=d_max_clip,
                        extent=[d_min_clip, d_max_clip, 0, 1]
                    )
                    ax_cb.set_xlim(d_min_clip, d_max_clip)
                    ax_cb.set_yticks([])

                    # Inset tick labels slightly so they are not flush with the image edge (avoids cropping without padding).
                    inset_px = 25  # horizontal inset in pixels
                    span = (d_max_clip - d_min_clip)
                    inset_data = span * (inset_px / W)  # convert pixel inset to data units
                    tick_positions = [d_min_clip + inset_data, d_max_clip - inset_data]
                    ax_cb.set_xticks(tick_positions)
                    ax_cb.set_xticklabels([f"{d_min_clip:.0f} m", f"{d_max_clip:.0f} m"])

                    ax_cb.tick_params(
                        axis='x',
                        which='both',
                        top=True,
                        bottom=False,
                        labeltop=True,
                        labelbottom=False,
                        direction='out',
                        pad=2,
                        length=0
                    )
                    for spine in ax_cb.spines.values():
                        spine.set_visible(False)

                    # Tight layout without adding side borders
                    fig.savefig(args.depth_map, dpi=dpi, bbox_inches='tight', pad_inches=0.01)
                    plt.close(fig)

                    print(f"Saved depth map [{d_min_clip:.3f},{d_max_clip:.3f}] m ({range_source}) : {args.depth_map}")
                    print(f"Data range before clip: min={data_min:.3f} max={data_max:.3f} valid_px={mask_valid.sum()}")

            except ImportError as _e:
                print("matplotlib not installed; cannot create depth map with colorbar.", file=sys.stderr)
                print("ImportError:", _e, file=sys.stderr)

    if args.overlay_png:
        if depth_img is None:
            print("WARNING: overlay requested but no raster was generated.", file=sys.stderr)
        else:
            try:
                import matplotlib
                import matplotlib.pyplot as plt
                import numpy as _np
                Path(args.overlay_png).parent.mkdir(parents=True, exist_ok=True)

                mask_valid = np.isfinite(depth_img)
                if not _np.any(mask_valid):
                    print("No valid depth pixels; skipping overlay.", file=sys.stderr)
                else:
                    # Determine scaling (use user clip if provided for consistency)
                    d_min_ov = args.depth_min if args.depth_min is not None else float(_np.nanmin(depth_img[mask_valid]))
                    d_max_ov = args.depth_max if args.depth_max is not None else float(_np.nanmax(depth_img[mask_valid]))
                    if d_max_ov <= d_min_ov:
                        d_max_ov = d_min_ov + 1e-6

                    depth_overlay = _np.clip(depth_img, d_min_ov, d_max_ov)

                    # Figure layout: image + thin horizontal gradient colorbar (same style as depth_map)
                    dpi = 100
                    cb_px = 20                # thin bar height
                    fig_w_in = W / dpi
                    fig_h_in = (H + cb_px) / dpi
                    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)

                    img_h_rel = H / (H + cb_px)
                    cb_h_rel  = cb_px / (H + cb_px)
                    cb_width_factor = 1.0
                    cb_x0 = (1.0 - cb_width_factor) / 2.0
                    cb_y0 = img_h_rel

                    ax_img = fig.add_axes([0.0, 0.0, 1.0, img_h_rel])
                    ax_cb  = fig.add_axes([cb_x0, cb_y0, cb_width_factor, cb_h_rel])

                    # Base grayscale
                    ax_img.imshow(gray, cmap='gray', interpolation='nearest')
                    # Overlay depth scatter (use small square markers)
                    yv, xv = _np.nonzero(mask_valid)
                    sc = ax_img.scatter(
                        xv, yv,
                        c=depth_overlay[mask_valid],
                        s=1,
                        cmap='viridis_r',
                        vmin=d_min_ov,
                        vmax=d_max_ov,
                        marker='s',
                        linewidths=0
                    )
                    ax_img.set_xlim(0, W)
                    ax_img.set_ylim(H, 0)
                    ax_img.axis('off')

                    # Build horizontal gradient colorbar
                    grad_w = max(2, int(W * cb_width_factor))
                    gradient = _np.linspace(d_min_ov, d_max_ov, grad_w, dtype=_np.float32)[None, :]
                    cmap = matplotlib.colormaps['viridis_r']
                    ax_cb.imshow(
                        gradient,
                        aspect='auto',
                        cmap=cmap,
                        vmin=d_min_ov,
                        vmax=d_max_ov,
                        extent=[d_min_ov, d_max_ov, 0, 1]
                    )
                    ax_cb.set_xlim(d_min_ov, d_max_ov)
                    ax_cb.set_yticks([])

                    # Two tick labels only, slightly inset so no extra padding needed
                    inset_px = 25
                    span = (d_max_ov - d_min_ov)
                    inset_data = span * (inset_px / W)
                    ticks = [d_min_ov + inset_data, d_max_ov - inset_data]
                    ax_cb.set_xticks(ticks)
                    ax_cb.set_xticklabels([f"{d_min_ov:.0f} m", f"{d_max_ov:.0f} m"])

                    ax_cb.tick_params(
                        axis='x',
                        which='both',
                        top=True,
                        bottom=False,
                        labeltop=True,
                        labelbottom=False,
                        length=0,
                        width=0,
                        pad=2
                    )
                    # Hide tick lines explicitly
                    for t in ax_cb.xaxis.get_major_ticks():
                        t.tick1line.set_visible(False)
                        t.tick2line.set_visible(False)

                    for spine in ax_cb.spines.values():
                        spine.set_visible(False)

                    fig.savefig(args.overlay_png, dpi=dpi, bbox_inches='tight', pad_inches=0.01)
                    plt.close(fig)
                    print(f"Saved overlay (depth range [{d_min_ov:.3f},{d_max_ov:.3f}]): {args.overlay_png}")
            except ImportError:
                print("matplotlib not installed; skipping overlay.", file=sys.stderr)

    if args.export_npz:
        out = Path(args.export_npz)
        out.parent.mkdir(parents=True, exist_ok=True)
        save = dict(i=i_all, j=j_all, depth=d_all, xyz=xyz_all, width=W, height=H)
        if depth_img is not None:
            save["depth_img"]=depth_img.astype(np.float32)
        np.savez_compressed(out, **save)
        print("Saved NPZ:", out)

    if args.export_label:
        if depth_img is None:
            print("ERROR: --export-label requested but rasterization failed/was skipped.", file=sys.stderr)
        else:
            lbl_path = Path(args.export_label)
            lbl_path.parent.mkdir(parents=True, exist_ok=True)
            mask = np.isfinite(depth_img).astype(np.uint8)
            np.savez_compressed(lbl_path,
                                depth=depth_img.astype(np.float32),
                                mask=mask)
            print(f"Saved label NPZ (depth, mask): {lbl_path}")
        
    if args.copy_las:
        src = Path(args.las)
        dst_input = Path(args.copy_las)
        if dst_input.suffix.lower() in (".las", ".laz"):
            dst = dst_input
        else:
            dst_input.mkdir(parents=True, exist_ok=True)
            dst = dst_input / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
            print(f"Copied LAS to: {dst}")
        except Exception as e:
            print(f"Failed to copy LAS: {e}", file=sys.stderr)

    if args.copy_cyl:
        src = Path(args.cyl)
        dst_input = Path(args.copy_cyl)
        if dst_input.suffix.lower() == ".cyl":
            dst = dst_input
        else:
            dst_input.mkdir(parents=True, exist_ok=True)
            dst = dst_input / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
            print(f"Copied CYL to: {dst}")
        except Exception as e:
            print(f"Failed to copy CYL: {e}", file=sys.stderr)

    return 0

def project_lidar_stream(cam, las_path:Path, W:int, H:int, chunk:int=2_000_000, progress:bool=False):
    """
    Stream LiDAR points, project with project_vect_safe, keep in-image points.
    Returns dict(i,j,depth,xyz)
    """
    all_i=[]; all_j=[]; all_d=[]; all_xyz=[]
    use_pb = progress
    if use_pb:
        try:
            from tqdm import tqdm
        except ImportError:
            use_pb=False
    with laspy.open(las_path) as lf:
        total = lf.header.point_count
        it = lf.chunk_iterator(chunk)
        if use_pb:
            import math as _m
            from tqdm import tqdm
            it = tqdm(it, total=max(1,_m.ceil(total/chunk)), desc="Projecting")
        for ch in it:
            xyz = np.vstack([ch.x, ch.y, ch.z]).T.astype(np.float64)
            Pc = (cam.Rot @ xyz.T).T + cam.t.reshape(1,3)
            d = depth_range(Pc)
            ij = project_vect_safe(xyz, cam)
            i_vals = ij[:,0]; j_vals = ij[:,1]
            finite = np.isfinite(i_vals) & np.isfinite(j_vals)
            if not np.any(finite):
                continue
            i_vals=i_vals[finite]; j_vals=j_vals[finite]; d=d[finite]; xyz=xyz[finite]
            inside = (i_vals>=0)&(i_vals<W)&(j_vals>=0)&(j_vals<H)
            if not np.any(inside):
                continue
            all_i.append(i_vals[inside].astype(np.float32))
            all_j.append(j_vals[inside].astype(np.float32))
            all_d.append(d[inside].astype(np.float32))
            all_xyz.append(xyz[inside].astype(np.float32))
    i_all = np.concatenate(all_i) if all_i else np.empty(0,np.float32)
    j_all = np.concatenate(all_j) if all_j else np.empty(0,np.float32)
    d_all = np.concatenate(all_d) if all_d else np.empty(0,np.float32)
    xyz_all = np.concatenate(all_xyz) if all_xyz else np.empty((0,3),np.float32)
    return dict(i=i_all, j=j_all, depth=d_all, xyz=xyz_all)

__all__ = [
    "depth_range","rasterize","project_lidar_stream","parse_args","main"
]

if __name__ == "__main__":
    sys.exit(main())
