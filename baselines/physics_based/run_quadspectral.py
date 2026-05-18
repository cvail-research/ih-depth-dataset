"""run_quadspectral.py

Quadspectral baseline for distance (d) estimation on LWIR HSI.
Direct translation of quadspectral_estimation.m.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from baselines.physics_based.utils.io_utils import load_lidar, load_scene
from baselines.physics_based.utils.metrics import evaluate, save_results
from baselines.physics_based.utils.physics import blackbody, estimate_T_air
from baselines.physics_based.utils.vis import save_distance_png, save_error_png


def _nearest_idx(lambda_um: np.ndarray, target_um: float) -> int:
	return int(np.argmin(np.abs(lambda_um - float(target_um))))


def _pick_bands(lambda_um: np.ndarray, attenuation: np.ndarray, lambda_min: float, lambda_max: float) -> tuple[int, int, int, int]:
	# idx1/idx2: vapor line vs clear band (local)
	mask = (lambda_um >= lambda_min) & (lambda_um <= lambda_max)
	idxs = np.where(mask)[0]
	if len(idxs) < 10:
		raise ValueError("Spectral range insufficient for auto-selection")

	idx1 = int(idxs[np.argmax(attenuation[idxs])])
	lam1 = float(lambda_um[idx1])

	# idx2: band near idx1 with minimum attenuation (clear band)
	win = np.where(np.abs(lambda_um - lam1) <= 0.15)[0]
	win = win[(lambda_um[win] >= lambda_min) & (lambda_um[win] <= lambda_max)]
	if len(win) < 2:
		idx2 = int(idxs[np.argmin(attenuation[idxs])])
	else:
		idx2 = int(win[np.argmin(attenuation[win])])

	# idx3/idx4: ozone line around 9.6µm
	ozone_center = 9.6
	ozone_win = np.where((lambda_um >= 9.4) & (lambda_um <= 9.8))[0]
	if len(ozone_win) == 0:
		idx3 = _nearest_idx(lambda_um, ozone_center)
	else:
		idx3 = int(ozone_win[np.argmax(attenuation[ozone_win])])
	lam3 = float(lambda_um[idx3])

	# idx4: clear band near idx3
	win3 = np.where(np.abs(lambda_um - lam3) <= 0.15)[0]
	if len(win3) < 2:
		idx4 = int(idxs[np.argmin(attenuation[idxs])])
	else:
		idx4 = int(win3[np.argmin(attenuation[win3])])

	# Ensure distinct indices
	uniq = {idx1, idx2, idx3, idx4}
	if len(uniq) < 4:
		# fallback: pick 4 well-separated indices within the range
		sorted_idxs = idxs[np.argsort(attenuation[idxs])]
		idx2 = int(sorted_idxs[0])
		idx4 = int(sorted_idxs[min(10, len(sorted_idxs) - 1)])
		uniq = {idx1, idx2, idx3, idx4}
		if len(uniq) < 4:
			raise ValueError("Auto-selection failed: repeated indices")

	return idx1, idx2, idx3, idx4


def quadspectral_distance(
	lambda_um: np.ndarray,
	measurements: np.ndarray,
	attenuation: np.ndarray,
	idx1: int,
	idx2: int,
	idx3: int,
	idx4: int,
	T_air: float,
	cor_coeff: float,
) -> np.ndarray:
	"""Return d_hat (H, W) in meters, with NaN for invalid pixels."""
	bb = blackbody(lambda_um, T_air)  # (K,)

	L1 = measurements[:, :, idx1] - bb[idx1]
	L2 = measurements[:, :, idx2] - bb[idx2]
	L3 = measurements[:, :, idx3]
	L4 = measurements[:, :, idx4]

	downwelling_correction = float(cor_coeff) * (L4 - L3)

	with np.errstate(divide="ignore", invalid="ignore"):
		transmittance_hat = (L2 - downwelling_correction) / L1
		d_hat = (-10.0 * np.log10(transmittance_hat)) / (attenuation[idx2] - attenuation[idx1])

	invalid = ~np.isfinite(d_hat) | (d_hat < 0)
	d_hat = d_hat.astype(np.float64)
	d_hat[invalid] = np.nan
	return d_hat


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description=__doc__)
	here = Path(__file__).resolve().parent
	default_data = here / "data"
	p.add_argument("--hsi-hdr", type=Path, required=True)
	p.add_argument("--lidar-mat", type=Path, default=None)
	p.add_argument("--data-dir", type=Path, default=default_data)
	p.add_argument(
		"--attenuation-profile",
		choices=["auto", "standard", "ozone_cues"],
		default="auto",
		help="Attenuation source profile.",
	)
	p.add_argument("--out-dir", type=Path, default=here / "outputs")
	p.add_argument("--t-air", type=float, default=None)
	p.add_argument("--lambda-min", type=float, default=8.5)
	p.add_argument("--lambda-max", type=float, default=12.0)
	p.add_argument("--cor-coeff", type=float, default=1.0)

	p.add_argument("--idx1", type=int, default=None)
	p.add_argument("--idx2", type=int, default=None)
	p.add_argument("--idx3", type=int, default=None)
	p.add_argument("--idx4", type=int, default=None)

	p.add_argument("--save-npy", action="store_true")
	p.add_argument("--save-fig", action="store_true", help="Save a PNG visualization of d_hat")
	return p.parse_args()


def main() -> None:
	args = parse_args()
	meas, lambda_um, attenuation, _downwelling, sensor = load_scene(
		str(args.hsi_hdr),
		str(args.data_dir / "ozone_cues"),
		str(args.data_dir / "standard"),
		attenuation_profile=args.attenuation_profile,
	)

	if args.t_air is None:
		T_air, _ = estimate_T_air(meas, lambda_um, attenuation, lambda_min=args.lambda_min, lambda_max=args.lambda_max)
	else:
		T_air = float(args.t_air)
		print(f"  T_air (manual): {T_air:.2f} K")

	if None in (args.idx1, args.idx2, args.idx3, args.idx4):
		idx1, idx2, idx3, idx4 = _pick_bands(lambda_um, attenuation, args.lambda_min, args.lambda_max)
	else:
		idx1, idx2, idx3, idx4 = int(args.idx1), int(args.idx2), int(args.idx3), int(args.idx4)

	print(
		"  Bands:\n"
		f"    idx1={idx1} λ={lambda_um[idx1]:.3f}µm atten={attenuation[idx1]:.5f}\n"
		f"    idx2={idx2} λ={lambda_um[idx2]:.3f}µm atten={attenuation[idx2]:.5f}\n"
		f"    idx3={idx3} λ={lambda_um[idx3]:.3f}µm atten={attenuation[idx3]:.5f}\n"
		f"    idx4={idx4} λ={lambda_um[idx4]:.3f}µm atten={attenuation[idx4]:.5f}\n"
		f"  cor_coeff={args.cor_coeff}"
	)

	d_hat = quadspectral_distance(
		lambda_um,
		meas,
		attenuation,
		idx1,
		idx2,
		idx3,
		idx4,
		T_air,
		args.cor_coeff,
	)

	out_dir = args.out_dir
	out_dir.mkdir(parents=True, exist_ok=True)
	scene_name = args.hsi_hdr.stem
	method = f"quadspectral_{sensor}"

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
		results = evaluate(d_hat, gt, method_name=method, verbose=True)
		save_results(results, str(out_dir), scene_name, method)


if __name__ == "__main__":
	main()
