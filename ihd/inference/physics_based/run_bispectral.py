"""run_bispectral.py

Bispectral baseline for distance (d) estimation on LWIR HSI.
Direct translation of bispectral_estimation.m.

Typical usage:
  python ihd/inference/physics_based/precompute_attenuation.py
  python ihd/inference/physics_based/run_bispectral.py --hsi-hdr <scene.hdr> --lidar-mat ihd/inference/physics_based/data/lidar.mat
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ihd.evaluation.model_io import save_depth_prediction
from ihd.inference.physics_based.utils.io_utils import load_lidar, load_scene
from ihd.inference.physics_based.utils.physics import blackbody, estimate_T_air
from ihd.inference.physics_based.utils.vis import save_distance_png, save_error_png


def _pick_bands(lambda_um: np.ndarray, attenuation: np.ndarray, lambda_min: float, lambda_max: float) -> tuple[int, int]:
	mask = (lambda_um >= lambda_min) & (lambda_um <= lambda_max)
	idxs = np.where(mask)[0]
	if len(idxs) < 2:
		raise ValueError("Spectral range too small to pick 2 bands")

	local_att = attenuation[idxs]
	idx1 = int(idxs[np.argmax(local_att)])  # band with highest attenuation
	idx2 = int(idxs[np.argmin(local_att)])  # band with lowest attenuation
	if idx1 == idx2:
		raise ValueError("Failed to pick two distinct bands")
	return idx1, idx2


def bispectral_distance(
	lambda_um: np.ndarray,
	measurements: np.ndarray,
	attenuation: np.ndarray,
	idx1: int,
	idx2: int,
	T_air: float,
) -> np.ndarray:
	"""Return d_hat (H, W) in meters, with NaN for invalid pixels."""
	L1 = measurements[:, :, idx1]
	L2 = measurements[:, :, idx2]

	bb1 = blackbody(np.array([lambda_um[idx1]]), T_air)[0]
	bb2 = blackbody(np.array([lambda_um[idx2]]), T_air)[0]

	denom = (L2 - bb2)
	numer = (L1 - bb1)

	# transmittance_hat = (L1 - bb(λ1,T_air)) / (L2 - bb(λ2,T_air))
	with np.errstate(divide="ignore", invalid="ignore"):
		transmittance_hat = numer / denom
		delta_att = attenuation[idx1] - attenuation[idx2]
		d_hat = (-10.0 * np.log10(transmittance_hat)) / delta_att

	# Filter non-physical values
	invalid = ~np.isfinite(d_hat) | (d_hat < 0)
	d_hat = d_hat.astype(np.float64)
	d_hat[invalid] = np.nan
	return d_hat


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description=__doc__)
	here = Path(__file__).resolve().parent
	default_data = here / "data"
	p.add_argument("--hsi-hdr", type=Path, required=True, help="Path to the scene .hdr")
	p.add_argument("--lidar-mat", type=Path, default=None, help="Path to lidar.mat (optional, only for error visualization)")
	p.add_argument("--data-dir", type=Path, default=default_data, help="Data directory (default: ihd/inference/physics_based/data)")
	p.add_argument("--out-dir", type=Path, default=here / "outputs", help="Output directory")
	p.add_argument("--t-air", type=float, default=None, help="Set T_air manually (K). If omitted, it is estimated")
	p.add_argument("--lambda-min", type=float, default=8.5, help="Range for automatic band selection")
	p.add_argument("--lambda-max", type=float, default=12.0, help="Range for automatic band selection")
	p.add_argument("--idx1", type=int, default=None, help="Band index 1 (absorption)")
	p.add_argument("--idx2", type=int, default=None, help="Band index 2 (clear)")
	p.add_argument("--save-npy", action="store_true", help="Also save legacy d_hat as .npy")
	p.add_argument("--save-fig", action="store_true", help="Save a PNG visualization of d_hat")
	return p.parse_args()


def main() -> None:
	args = parse_args()
	meas, lambda_um, attenuation, _downwelling, sensor = load_scene(
		str(args.hsi_hdr),
		str(args.data_dir / "precomputed"),
		str(args.data_dir),
	)

	if args.t_air is None:
		T_air, _ = estimate_T_air(meas, lambda_um, attenuation, lambda_min=args.lambda_min, lambda_max=args.lambda_max)
	else:
		T_air = float(args.t_air)
		print(f"  T_air (manual): {T_air:.2f} K")

	if args.idx1 is None or args.idx2 is None:
		idx1, idx2 = _pick_bands(lambda_um, attenuation, args.lambda_min, args.lambda_max)
	else:
		idx1, idx2 = int(args.idx1), int(args.idx2)
	print(
		f"  Bands: idx1={idx1} (λ={lambda_um[idx1]:.3f}µm, atten={attenuation[idx1]:.5f}), "
		f"idx2={idx2} (λ={lambda_um[idx2]:.3f}µm, atten={attenuation[idx2]:.5f})"
	)

	d_hat = bispectral_distance(lambda_um, meas, attenuation, idx1, idx2, T_air)

	out_dir = args.out_dir
	out_dir.mkdir(parents=True, exist_ok=True)
	scene_name = args.hsi_hdr.stem
	method = f"bispectral_{sensor}"

	metadata = {
		"model_slug": "bispectral",
		"method_name": method,
		"sensor": sensor,
		"t_air_k": float(T_air),
		"idx1": int(idx1),
		"idx2": int(idx2),
		"lambda_idx1_um": float(lambda_um[idx1]),
		"lambda_idx2_um": float(lambda_um[idx2]),
	}
	out_npz = save_depth_prediction(
		d_hat.astype(np.float32),
		out_dir,
		model_name=method,
		hdr_path=args.hsi_hdr,
		metadata=metadata,
		save_visualization=bool(args.save_fig),
	)
	print(f"  Saved: {out_npz}")

	if args.save_npy:
		out_npy = out_dir / f"{scene_name}_{method}_d.npy"
		np.save(out_npy, d_hat.astype(np.float32))
		print(f"  Saved: {out_npy}")

	if args.save_fig:
		out_png = out_dir / f"{scene_name}_{method}_d.png"
		save_distance_png(d_hat, out_png, title=f"{scene_name} | {method}")
		print(f"  Saved: {out_png}")

	if args.lidar_mat is not None:
		gt = load_lidar(str(args.lidar_mat))
		if gt.shape != d_hat.shape:
			raise ValueError(f"Shape mismatch: pred={d_hat.shape} vs gt={gt.shape}")
		if args.save_fig:
			out_err_png = out_dir / f"{scene_name}_{method}_error.png"
			save_error_png(d_hat, gt, out_err_png, title=f"{scene_name} | {method} | error")
			print(f"  Saved: {out_err_png}")


if __name__ == "__main__":
	main()
