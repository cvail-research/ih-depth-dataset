import argparse
import math
import sys
from pathlib import Path
from typing import List

import numpy as np

try:
    from ihd.datasets.cylindrical_camera import camera, read_cam, project_vect_safe
except ImportError:
    from cylindrical_camera import camera, read_cam, project_vect_safe

try:
    from scipy.optimize import least_squares
except ImportError:
    print("Please sync project dependencies with uv in the active env: uv sync --active", file=sys.stderr)
    sys.exit(1)

# ---------------- Shared functions with notebook ----------------

def calibrate_single(corr_i, corr_j, corr_xyz, cam_init, *, opt_mode:str,
                     image_width=None, image_height=None, max_iters=400):
    """
    opt_mode: 'extr' | 'extr+w' | 'all'
    Returns (optimized_cam, stats)
    """
    if opt_mode == "all":
        mask = ParamMask(opt_extr=True, opt_R=True, opt_w=True, opt_y=True, opt_f=True, opt_j0=True)
    elif opt_mode == "extr+w":
        mask = ParamMask(opt_extr=True, opt_w=True)
    elif opt_mode == "extr":
        mask = ParamMask(opt_extr=True)
    else:
        raise ValueError("Unknown opt_mode")
    cam_opt, stats = calibrate(
        corr_i, corr_j, corr_xyz, cam_init, mask,
        max_iters=max_iters,
        wrap_horizontal=True,
        width_hint=image_width,
        height_hint=image_height
    )
    return cam_opt, stats

def run_calibration_pipeline(corr_path:str, init_cyl:str, opt_modes, image_width:int=None,
                             image_height:int=None, max_iters=400):
    """
    opt_modes: iterable like ['extr','extr+w','all']
    Returns list of dict(name, cam, stats)
    """
    corr_i, corr_j, corr_xyz = read_corresp(corr_path)
    base_cam = read_cam(init_cyl)
    out = []
    for mode in opt_modes:
        cam_opt, stats = calibrate_single(
            corr_i, corr_j, corr_xyz, base_cam,
            opt_mode=mode, image_width=image_width,
            image_height=image_height, max_iters=max_iters
        )
        out.append(dict(name=mode, cam=cam_opt, stats=stats))
    return out


# ---------------- Correspondence I/O ----------------

def read_corresp(path: str):
    """
    Supported formats:

    1) With count line:
       N
       i j X Y Z
       ...

    2) Without count line (manual annotations):
       (optional header: i j X Y Z | u v x y z ...)
       i j X Y Z
       i j X Y Z
       ...

    Header line is ignored if its first token is non-numeric.
    Returns (i_array, j_array, xyz_array)
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)

    with p.open("r") as f:
        raw = [ln.strip() for ln in f if ln.strip() and ln.strip() not in {'""', '"'}]

    if not raw:
        raise ValueError("Empty correspondence file.")

    # Detect and remove header (first token non-numeric)
    def is_numeric_token(tok: str) -> bool:
        try:
            float(tok)
            return True
        except ValueError:
            return False

    lines = raw[:]
    if lines and not is_numeric_token(lines[0].split()[0]):
        # header candidate: require >=5 tokens
        if len(lines[0].split()) >= 5:
            lines = lines[1:]

    if not lines:
        raise ValueError("No data lines after removing header.")

    # Check for count line (single integer token whose value equals remaining numeric lines with 5 tokens)
    first_tokens = lines[0].split()
    has_count = False
    count = None
    if len(first_tokens) == 1 and first_tokens[0].isdigit():
        try:
            count_candidate = int(first_tokens[0])
            # Count following numeric 5-token lines
            numeric_lines = []
            for ln in lines[1:]:
                parts = ln.split()
                if len(parts) == 5 and all(is_numeric_token(p) for p in parts):
                    numeric_lines.append(ln)
                else:
                    break
            if len(numeric_lines) == count_candidate:
                has_count = True
                count = count_candidate
                data_lines = numeric_lines
                # Ignore any trailing lines beyond the count block
            else:
                # Treat as no count; fall through
                pass
        except Exception:
            pass

    if not has_count:
        # Use all lines that have exactly 5 numeric tokens
        data_lines = []
        for ln in lines:
            parts = ln.split()
            if len(parts) == 5 and all(is_numeric_token(p) for p in parts):
                data_lines.append(ln)

    if not data_lines:
        raise ValueError("No valid correspondence rows (need lines with 5 numeric values).")

    i_list = []
    j_list = []
    xyz_list = []
    for row in data_lines:
        a, b, x, y, z = row.split()
        try:
            i_list.append(float(a))
            j_list.append(float(b))
            xyz_list.append([float(x), float(y), float(z)])
        except ValueError as e:
            raise ValueError(f"Non-numeric row: {row}") from e

    return (np.array(i_list, dtype=np.float64),
            np.array(j_list, dtype=np.float64),
            np.array(xyz_list, dtype=np.float64))


# ---------------- Rotation utilities ----------------

def rodrigues(omega: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(omega)
    if theta < 1e-12:
        return np.eye(3)
    k = omega / theta
    kx, ky, kz = k
    K = np.array([[0, -kz, ky],
                  [kz, 0, -kx],
                  [-ky, kx, 0]], dtype=np.float64)
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


# ---------------- Projection (with optional Jacobian later) ----------------

# def project_vect_safe(P: np.ndarray, cam: camera):
#     """
#     Same as project_vect but more numerically robust (clips arcsin domain).
#     """
#     P_0 = cam.Rot.dot(P.T).T + cam.t
#     X0 = P_0[:, 0]; Y0 = P_0[:, 1]; Z0 = P_0[:, 2]
#     hyp = np.sqrt(X0 * X0 + Z0 * Z0)
#     denom = np.clip(hyp, 1e-12, None)
#     arg = (cam.R * math.sin(cam.w)) / denom
#     arg = np.clip(arg, -1.0, 1.0)
#     i_angle = np.arctan2(X0, Z0) - cam.w + np.arcsin(arg)
#     i_angle %= (2 * np.pi)
#     i = i_angle / cam.y
#     # vertical
#     inner = hyp * hyp - (cam.R * cam.R) * (math.sin(cam.w) ** 2)
#     inner = np.clip(inner, 1e-12, None)
#     B = np.arctan2(Y0, (np.sqrt(inner) - cam.R * math.cos(cam.w)))
#     j = cam.f * np.tan(B) + cam.j0
#     return np.stack([i, j], axis=1)


# ---------------- Parameter handling ----------------

class ParamMask:
    def __init__(self,
                 opt_extr=True,
                 opt_R=False,
                 opt_w=False,
                 opt_y=False,
                 opt_f=False,
                 opt_j0=False):
        self.opt_extr = opt_extr
        self.opt_R = opt_R
        self.opt_w = opt_w
        self.opt_y = opt_y
        self.opt_f = opt_f
        self.opt_j0 = opt_j0

    def size(self):
        n = 0
        if self.opt_extr:
            n += 6  # 3 rot vec + 3 t
        if self.opt_R: n += 1
        if self.opt_w: n += 1
        if self.opt_y: n += 1
        if self.opt_f: n += 1
        if self.opt_j0: n += 1
        return n


def pack_params(cam: camera, mask: ParamMask) -> np.ndarray:
    params: List[float] = []
    if mask.opt_extr:
        # Extract small rotation vector from current Rot (approx via log map)
        R = cam.Rot
        angle = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
        if angle < 1e-12:
            omega = np.zeros(3)
        else:
            w = (1 / (2 * math.sin(angle))) * np.array([
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1]
            ])
            omega = angle * w
        params.extend(omega.tolist())
        params.extend(cam.t.tolist())
    if mask.opt_R: params.append(cam.R)
    if mask.opt_w: params.append(cam.w)
    if mask.opt_y: params.append(cam.y)
    if mask.opt_f: params.append(cam.f)
    if mask.opt_j0: params.append(cam.j0)
    return np.array(params, dtype=np.float64)


def unpack_params(x: np.ndarray, cam0: camera, mask: ParamMask) -> camera:
    idx = 0
    Rot = cam0.Rot.copy()
    t = cam0.t.copy()
    Rval = cam0.R
    w = cam0.w
    y = cam0.y
    f = cam0.f
    j0 = cam0.j0

    if mask.opt_extr:
        omega = x[idx:idx + 3]; idx += 3
        Rot = rodrigues(omega) @ Rot
        t = x[idx:idx + 3]; idx += 3
    if mask.opt_R:
        Rval = x[idx]; idx += 1
    if mask.opt_w:
        w = x[idx]; idx += 1
    if mask.opt_y:
        y = x[idx]; idx += 1
    if mask.opt_f:
        f = x[idx]; idx += 1
    if mask.opt_j0:
        j0 = x[idx]; idx += 1
    return camera(Rval, w, y, f, j0, Rot, t)


# ---------------- Calibration core ----------------

def wrap_diff(di: np.ndarray, cam: camera):
    """Wrap horizontal residuals into half native cylindrical width."""
    Wcyl = (2 * math.pi) / cam.y
    return ((di + 0.5 * Wcyl) % Wcyl) - 0.5 * Wcyl

def calibrate(corr_i, corr_j, corr_xyz, cam_init: camera, mask: ParamMask,
              max_iters=300, wrap_horizontal=True,
              width_hint=None, height_hint=None):
    """
    Simplified calibrate:
    - Always linear least squares
    - Optional normalization if width/height hints provided
    - Internal soft bounds (not exposed)
    - Optional horizontal wrap handling
    """
    x0 = pack_params(cam_init, mask)

    # Weights (optional normalization)
    if width_hint and height_hint:
        wi = 1.0 / max(width_hint, 1)
        wj = 1.0 / max(height_hint, 1)
    else:
        wi = 1.0
        wj = 1.0

    def residuals_vec(x):
        cam_cur = unpack_params(x, cam_init, mask)
        ij = project_vect_safe(corr_xyz, cam_cur)
        di = ij[:, 0] - corr_i
        if wrap_horizontal:
            di = wrap_diff(di, cam_cur)
        dj = ij[:, 1] - corr_j
        return np.concatenate([di * wi, dj * wj], axis=0)

    # Minimal internal bounds (only if those params active)
    lo = []
    hi = []
    if mask.opt_extr:
        lo += [-np.inf]*3 + [-np.inf]*3
        hi += [ np.inf]*3 + [ np.inf]*3
    if mask.opt_R:
        R0 = cam_init.R
        lo.append(0.1 * R0 if R0 > 0 else 1e-3); hi.append(10 * max(R0, 1e-3))
    if mask.opt_w:
        lo.append(-math.pi); hi.append(math.pi)
    if mask.opt_y:
        y0 = cam_init.y
        lo.append(0.2 * y0); hi.append(5.0 * y0)
    if mask.opt_f:
        f0 = cam_init.f
        lo.append(0.2 * f0); hi.append(5.0 * f0)
    if mask.opt_j0:
        # If height hint available, constrain near image band
        if height_hint:
            lo.append(-0.25 * height_hint)
            hi.append(1.25 * height_hint)
        else:
            lo.append(cam_init.j0 - 5000)
            hi.append(cam_init.j0 + 5000)
    if lo:
        bounds = (np.array(lo, dtype=np.float64), np.array(hi, dtype=np.float64))
    else:
        bounds = (-np.inf, np.inf)

    result = least_squares(
        residuals_vec,
        x0,
        max_nfev=max_iters,
        bounds=bounds
    )
    cam_opt = unpack_params(result.x, cam_init, mask)
    ij_opt = project_vect_safe(corr_xyz, cam_opt)
    di = ij_opt[:, 0] - corr_i
    if wrap_horizontal:
        di = wrap_diff(di, cam_opt)
    dj = ij_opt[:, 1] - corr_j
    stats = dict(
        rmse_i=float(np.sqrt(np.mean(di*di))),
        rmse_j=float(np.sqrt(np.mean(dj*dj))),
        rmse_total=float(np.sqrt(np.mean(di*di + dj*dj))),
        mean_abs_i=float(np.mean(np.abs(di))),
        mean_abs_j=float(np.mean(np.abs(dj))),
        max_abs_i=float(np.max(np.abs(di))),
        max_abs_j=float(np.max(np.abs(dj))),
        n=len(corr_i),
        success=bool(result.success),
        message=result.message,
        nfev=int(result.nfev),
        weighted=bool(width_hint and height_hint),
        wrapped=wrap_horizontal
    )
    return cam_opt, stats


# ---------------- Helpers ----------------

def write_cyl(cam: camera, path: str):
    with open(path, "w") as f:
        for r in cam.Rot:
            f.write(" ".join(f"{v:.6f}" for v in r) + "\n")
        f.write(" ".join(f"{v:.6f}" for v in cam.t) + "\n")
        f.write(f"{cam.R:.6f}\n")
        f.write(f"{cam.w:.6f}\n")
        f.write(f"{cam.f:.6f}\n")
        f.write(f"{cam.j0:.6f}\n")
        f.write(f"{cam.y:.9f}\n")


def build_initial_camera(args, corr_xyz, corr_i, corr_j):
    if args.init_cyl:
        return read_cam(args.init_cyl)

    # Need intrinsics provided
    missing = [name for name in ["R", "w", "y", "f", "j0"] if getattr(args, name) is None]
    if missing:
        raise ValueError(f"Missing intrinsic(s) {missing}; provide --init-cyl or all of --R --w --y --f --j0")

    Rot = np.eye(3, dtype=np.float64)
    t = np.zeros(3, dtype=np.float64)

    return camera(args.R, args.w, args.y, args.f, args.j0, Rot, t)


def parse_args():
    ap = argparse.ArgumentParser(
        "Simple cylindrical camera calibration (start small).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--corresp", required=True, help="(i j X Y Z) correspondence file")
    ap.add_argument("--out-cyl", required=True, help="Output calibrated .cyl path")
    ap.add_argument("--init-cyl", required=True, help="Initial .cyl (required in simplified mode)")
    ap.add_argument("--show-points", action="store_true", help="Print per-point residuals after optimization")

    # Minimal future-growth toggles
    ap.add_argument("--opt-w", action="store_true",
                    help="Also refine principal angle w (besides extrinsics).")
    ap.add_argument("--opt-all", action="store_true",
                    help="Refine all intrinsics (R, w, y, f, j0). Overrides --opt-w.")

    # Optional image size hints (enable normalization of residuals)
    ap.add_argument("--image-width", type=int, help="Image width for normalization (optional)")
    ap.add_argument("--image-height", type=int, help="Image height for normalization (optional)")

    ap.add_argument("--max-iters", type=int, default=400, help="Max optimizer iterations")
    return ap.parse_args()


def main():
    args = parse_args()

    corr_i, corr_j, corr_xyz = read_corresp(args.corresp)
    print(f"Loaded {len(corr_i)} correspondences.")

    # Require init cyl for simplicity
    cam0 = read_cam(args.init_cyl)
    print("Initial camera:")
    print(f"  R={cam0.R} w={cam0.w} y={cam0.y} f={cam0.f} j0={cam0.j0}")
    print(f"  Rot=\n{cam0.Rot}")
    print(f"  t={cam0.t}")

    # Parameter selection
    if args.opt_all:
        mask = ParamMask(opt_extr=True, opt_R=True, opt_w=True, opt_y=True, opt_f=True, opt_j0=True)
        print("Optimizing: extrinsics + ALL intrinsics")
    elif args.opt_w:
        mask = ParamMask(opt_extr=True, opt_w=True)
        print("Optimizing: extrinsics + w")
    else:
        mask = ParamMask(opt_extr=True)
        print("Optimizing: extrinsics only")

    # Initial diagnostics
    ij0 = project_vect_safe(corr_xyz, cam0)
    di0 = ij0[:,0] - corr_i
    dj0 = ij0[:,1] - corr_j
    print(f"Initial residuals: rmse_i={np.sqrt(np.mean(di0*di0)):.2f} rmse_j={np.sqrt(np.mean(dj0*dj0)):.2f}")

    cam_opt, stats = calibrate(
        corr_i, corr_j, corr_xyz, cam0, mask,
        max_iters=args.max_iters,
        wrap_horizontal=True,
        width_hint=args.image_width,
        height_hint=args.image_height
    )

    print("Calibration stats:")
    for k,v in stats.items():
        print(f"  {k}: {v}")

    print("Optimized camera:")
    print(f"  R={cam_opt.R} w={cam_opt.w} y={cam_opt.y} f={cam_opt.f} j0={cam_opt.j0}")
    print(f"  Rot=\n{cam_opt.Rot}")
    print(f"  t={cam_opt.t}")

    write_cyl(cam_opt, args.out_cyl)
    print(f"Wrote calibrated camera: {args.out_cyl}")

    if args.show_points:
        ij_opt = project_vect_safe(corr_xyz, cam_opt)
        print("Per-point residuals (wrapped i):")
        for (ig, jg, ip, jp) in zip(corr_i, corr_j, ij_opt[:,0], ij_opt[:,1]):
            di = ip - ig
            di = wrap_diff(np.array([di]), cam_opt)[0]
            dj = jp - jg
            dn = math.sqrt(di*di + dj*dj)
            print(f"{ig:7.2f} {jg:7.2f} -> {ip:7.2f} {jp:7.2f} | {di:+7.2f} {dj:+7.2f} | {dn:6.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
