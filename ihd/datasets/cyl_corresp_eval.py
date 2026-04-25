import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

from cylindrical_camera import read_cam, project_vect, camera

def _clone_camera(src: camera, Rot=None, t=None) -> camera:
    """Create a new camera instance copying all fields, with optional updated Rot/t."""
    return camera(
        src.R, src.w, src.y, src.f, src.j0,
        Rot if Rot is not None else src.Rot,
        t if t is not None else src.t,
    )

@dataclass
class Correspondence:
    i: float
    j: float
    xyz: np.ndarray  # shape (3,)


# -------- Parsing --------

def read_cyl_camera(path: str) -> camera:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Cyl camera file not found: {p}")
    return read_cam(str(p))


def read_corresp(path: str) -> List[Correspondence]:
    """
    Supported formats (examples):
    ""
    10
    941 114 0.660000 41.360001 -2.980000
    ...
    ""
    or
    10
    i j X Y Z
    ...

    Each data line: i j X Y Z (pixel indices i,j in cylindrical image; world XYZ).
    Non-numeric header lines are skipped until a pure integer (count) line is found.
    Quote-only lines ("", ")", etc.) are ignored.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Correspondence file not found: {p}")

    with p.open("r") as f:
        raw_lines = [ln.strip() for ln in f]

    # Remove blank / quote-only lines
    lines = [ln for ln in raw_lines if ln and ln not in {'""', '"'}]

    # Find line with count
    idx = 0
    while idx < len(lines):
        token = lines[idx].split()[0]
        if token.isdigit():
            break
        idx += 1
    if idx >= len(lines):
        raise ValueError("Could not find correspondence count line.")
    try:
        n = int(lines[idx].split()[0])
    except ValueError as e:
        raise ValueError(f"Invalid count line: {lines[idx]}") from e

    data_lines = lines[idx + 1: idx + 1 + n]
    if len(data_lines) != n:
        raise ValueError(f"Expected {n} correspondence lines, found {len(data_lines)}.")

    corrs: List[Correspondence] = []
    for li, dl in enumerate(data_lines, start=1):
        parts = dl.split()
        if len(parts) != 5:
            raise ValueError(f"Bad correspondence line #{li}: '{dl}' (need 5 tokens)")
        try:
            i, j = float(parts[0]), float(parts[1])
            X, Y, Z = map(float, parts[2:])
        except ValueError as e:
            raise ValueError(f"Non-numeric value in line #{li}: '{dl}'") from e
        corrs.append(Correspondence(i=i, j=j, xyz=np.array([X, Y, Z], dtype=float)))
    return corrs


# -------- Projection / Residuals --------

def project_points(cam: camera, corrs: List[Correspondence]) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.stack([c.xyz for c in corrs], axis=0)  # (N,3)
    ij_pred = project_vect(pts, cam)  # (N,2)
    ij_gt = np.stack([[c.i, c.j] for c in corrs], axis=0)
    return ij_pred, ij_gt


def residuals(ij_pred: np.ndarray, ij_gt: np.ndarray) -> Dict[str, float]:
    diff = ij_pred - ij_gt
    di = diff[:, 0]
    dj = diff[:, 1]
    per_pix_err = np.sqrt(di * di + dj * dj)
    stats = {
        "n": int(len(diff)),
        "mean_abs_i": float(np.mean(np.abs(di))),
        "mean_abs_j": float(np.mean(np.abs(dj))),
        "rmse_i": float(np.sqrt(np.mean(di * di))),
        "rmse_j": float(np.sqrt(np.mean(dj * dj))),
        "mean_pixel_err": float(np.mean(per_pix_err)),
        "rmse_pixel_err": float(np.sqrt(np.mean(per_pix_err * per_pix_err))),
        "max_abs_i": float(np.max(np.abs(di))),
        "max_abs_j": float(np.max(np.abs(dj))),
        "max_pixel_err": float(np.max(per_pix_err)),
        "median_pixel_err": float(np.median(per_pix_err)),
        "p90_pixel_err": float(np.percentile(per_pix_err, 90.0)),
    }
    return stats


# -------- Refinement --------

def refine(cam: camera, corrs: List[Correspondence], mode: str = "t", max_iters: int = 200) -> Tuple[camera, Dict[str, float]]:
    """
    mode:
      - 'none': no refinement
      - 't': refine translation only
      - 'rt': refine rotation (axis-angle) + translation
    Uses scipy.optimize if available; falls back to simple coordinate search.
    """
    if mode == "none":
        ij_pred, ij_gt = project_points(cam, corrs)
        return cam, residuals(ij_pred, ij_gt)

    try:
        from scipy.optimize import least_squares
    except ImportError:
        print("SciPy not available; falling back to naive coordinate descent.", file=sys.stderr)
        return refine_naive(cam, corrs, mode)

    # Parameterization:
    #   t (3)
    #   optional rotation as axis-angle omega (3) (applied as exp(omega^) * R0)
    R0 = np.array(cam.Rot, dtype=float)
    t0 = np.array(cam.t, dtype=float).reshape(3,)

    if mode == "t":
        x0 = t0
    elif mode == "rt":
        x0 = np.zeros(6, dtype=float)  # omega(3)=0, t(3)=current
        x0[3:] = t0
    else:
        raise ValueError(f"Unknown refine mode: {mode}")

    def rodrigues(omega: np.ndarray) -> np.ndarray:
        theta = np.linalg.norm(omega)
        if theta < 1e-12:
            return np.eye(3)
        k = omega / theta
        K = np.array([[0, -k[2], k[1]],
                      [k[2], 0, -k[0]],
                      [-k[1], k[0], 0]], dtype=float)
        return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)

    pts = np.stack([c.xyz for c in corrs], axis=0)
    ij_gt = np.stack([[c.i, c.j] for c in corrs], axis=0)

    def residual_vec(x):
        if mode == "t":
            t = x
            R = R0
        else:  # rt
            omega = x[:3]
            t = x[3:]
            R = rodrigues(omega) @ R0
        # Construct a temporary camera
        tmp = _clone_camera(cam, Rot=R, t=t)
        ij_pred = project_vect(pts, tmp)
        return (ij_pred - ij_gt).ravel()

    res = least_squares(residual_vec, x0, max_nfev=max_iters, verbose=0)
    x_opt = res.x
    if mode == "t":
        t_opt = x_opt
        R_opt = R0
    else:
        omega_opt = x_opt[:3]
        t_opt = x_opt[3:]
        R_opt = rodrigues(omega_opt) @ R0

    cam_ref = _clone_camera(cam, Rot=R_opt, t=t_opt)
    cam_ref.Rot = R_opt
    cam_ref.t = t_opt
    cam_ref.R = cam.R
    cam_ref.w = cam.w
    cam_ref.f = cam.f
    cam_ref.j0 = cam.j0
    cam_ref.y = cam.y

    ij_pred, _ = project_points(cam_ref, corrs)
    stats = residuals(ij_pred, ij_gt)
    stats["optimizer_nfev"] = int(res.nfev)
    stats["optimizer_cost"] = float(res.cost)
    return cam_ref, stats


def refine_naive(cam: camera, corrs: List[Correspondence], mode: str) -> Tuple[camera, Dict[str, float]]:
    from copy import deepcopy
    best = deepcopy(cam)
    ij_pred, ij_gt = project_points(best, corrs)
    best_loss = float(np.sum((ij_pred - ij_gt) ** 2))

    # Only translation naive search implemented
    if mode not in ("t", "rt"):
        return best, residuals(ij_pred, ij_gt)

    step = 0.05
    for _ in range(200):
        improved = False
        for axis in range(3):
            for delta in (+step, -step):
                trial = deepcopy(best)
                trial.t[axis] += delta
                ij_pred2, _ = project_points(trial, corrs)
                loss = float(np.sum((ij_pred2 - ij_gt) ** 2))
                if loss < best_loss:
                    best, best_loss, improved = trial, loss, True
        if not improved:
            step *= 0.5
        if step < 1e-4:
            break

    ij_pred, _ = project_points(best, corrs)
    return best, residuals(ij_pred, ij_gt)


# -------- CLI --------

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaluate cylindrical camera against i,j ↔ XYZ correspondences.")
    ap.add_argument("--cyl", required=True, help="Path to .cyl file")
    ap.add_argument("--corresp", required=True, help="Path to correspondence file")
    ap.add_argument("--json-out", help="Optional path to write stats JSON")
    ap.add_argument("--show-points", action="store_true", help="Print per-point residuals")
    ap.add_argument("--refine", choices=["none", "t", "rt"], default="none", help="Refinement mode")
    ap.add_argument("--max-rmse", type=float, help="Fail (exit 2) if rmse_pixel_err exceeds this")
    ap.add_argument("--save-refined-cyl", help="If refining, write out a new .cyl file here")
    ap.add_argument("--max-iters", type=int, default=200, help="Max iterations for optimizer")
    return ap


def format_stats(stats: Dict[str, float]) -> str:
    return (
        f"n={stats['n']} | mean_px={stats['mean_pixel_err']:.3f} | rmse_px={stats['rmse_pixel_err']:.3f} | "
        f"median_px={stats['median_pixel_err']:.3f} | p90_px={stats['p90_pixel_err']:.3f} | "
        f"max_px={stats['max_pixel_err']:.3f}"
    )


def write_cyl(cam: camera, path: str):
    """
    Write a .cyl file in the same ordering you described:
      3 lines R (3x3)
      1 line t
      radius
      principal angle (w)
      focal length (f)
      principal row (j0)
      pixel angle width (y)
    """
    with open(path, "w") as f:
        R = np.asarray(cam.Rot, dtype=float)
        for r in R:
            f.write(" ".join(f"{v:.6f}" for v in r) + "\n")
        t = np.asarray(cam.t, dtype=float).reshape(3,)
        f.write(" ".join(f"{v:.6f}" for v in t) + "\n")
        f.write(f"{cam.R:.6f}\n")
        f.write(f"{cam.w:.6f}\n")
        f.write(f"{cam.f:.6f}\n")
        f.write(f"{cam.j0:.6f}\n")
        f.write(f"{cam.y:.9f}\n")


def main(argv=None) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)

    cam = read_cyl_camera(args.cyl)
    corrs = read_corresp(args.corresp)

    # Initial evaluation
    ij_pred, ij_gt = project_points(cam, corrs)
    base_stats = residuals(ij_pred, ij_gt)

    print("Initial:", format_stats(base_stats))

    final_cam = cam
    final_stats = base_stats

    if args.refine != "none":
        final_cam, final_stats = refine(cam, corrs, mode=args.refine, max_iters=args.max_iters)
        print(f"Refined ({args.refine}):", format_stats(final_stats))

    if args.show_points:
        diff = ij_pred - ij_gt if args.refine == "none" else project_points(final_cam, corrs)[0] - ij_gt
        print("\nPer-point residuals (i_gt j_gt -> i_pred j_pred | di dj | d_norm):")
        ij_final = ij_pred if args.refine == "none" else project_points(final_cam, corrs)[0]
        for (gt, pr) in zip(ij_gt, ij_final):
            di = pr[0] - gt[0]
            dj = pr[1] - gt[1]
            dn = math.sqrt(di * di + dj * dj)
            print(f"{gt[0]:8.3f} {gt[1]:8.3f} -> {pr[0]:8.3f} {pr[1]:8.3f} | {di:7.3f} {dj:7.3f} | {dn:7.3f}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(final_stats, f, indent=2)
        print(f"Wrote stats JSON: {out_path}")

    if args.save_refined_cyl and args.refine != "none":
        write_cyl(final_cam, args.save_refined_cyl)
        print(f"Wrote refined .cyl: {args.save_refined_cyl}")

    if args.max_rmse is not None and final_stats["rmse_pixel_err"] > args.max_rmse:
        print(f"RMSE {final_stats['rmse_pixel_err']:.3f} exceeds threshold {args.max_rmse:.3f}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())