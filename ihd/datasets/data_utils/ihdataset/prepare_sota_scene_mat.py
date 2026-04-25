#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import scipy.io
import spectral as spy


def _load_envi_cube(hdr_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      cube: (H, W, K) float32
      wavelengths_um: (K,) float64
    """
    img = spy.open_image(hdr_path)
    cube = np.array(img.load(), dtype=np.float32)

    if "wavelength" not in img.metadata:
        raise KeyError(f"ENVI metadata missing 'wavelength' for {hdr_path}")
    wavelengths_um = np.array([float(x) for x in img.metadata["wavelength"]], dtype=np.float64)

    if cube.shape[2] != wavelengths_um.shape[0]:
        raise ValueError(f"Cube bands {cube.shape[2]} != wavelength count {wavelengths_um.shape[0]}")

    return cube, wavelengths_um


def _load_sota_lambda(lambda_mat_path: str) -> np.ndarray:
    """
    SOTA lambda.mat contains variable 'lambda' in micrometers.
    Returns (K,) float64.
    """
    d = scipy.io.loadmat(lambda_mat_path)
    if "lambda" not in d:
        raise KeyError(f"{lambda_mat_path} missing variable 'lambda'")
    lam = np.array(d["lambda"]).squeeze()
    if lam.ndim != 1:
        lam = lam.reshape(-1)
    return lam.astype(np.float64)


def _nearest_band_indices(src_lambda_um: np.ndarray, tgt_lambda_um: np.ndarray) -> np.ndarray:
    """
    For each target wavelength, pick the nearest source band index.
    Deterministic and fast, preserves original radiance values.
    """
    src = src_lambda_um
    tgt = tgt_lambda_um

    if not (np.all(np.isfinite(src)) and np.all(np.isfinite(tgt))):
        raise ValueError("Non-finite wavelengths found.")
    if src.ndim != 1 or tgt.ndim != 1:
        raise ValueError("Wavelength arrays must be 1D.")

    # Ensure monotonic src for searchsorted; if not, sort but keep inverse mapping
    if np.any(np.diff(src) < 0):
        order = np.argsort(src)
        src_sorted = src[order]
        idx_sorted = np.searchsorted(src_sorted, tgt, side="left")
        idx_sorted = np.clip(idx_sorted, 0, len(src_sorted) - 1)
        idx_prev = np.clip(idx_sorted - 1, 0, len(src_sorted) - 1)
        choose_prev = np.abs(src_sorted[idx_prev] - tgt) <= np.abs(src_sorted[idx_sorted] - tgt)
        idx_sorted = np.where(choose_prev, idx_prev, idx_sorted)
        return order[idx_sorted]

    idx = np.searchsorted(src, tgt, side="left")
    idx = np.clip(idx, 0, len(src) - 1)
    idx_prev = np.clip(idx - 1, 0, len(src) - 1)
    choose_prev = np.abs(src[idx_prev] - tgt) <= np.abs(src[idx] - tgt)
    return np.where(choose_prev, idx_prev, idx)


def main() -> None:
    p = argparse.ArgumentParser(description="Convert an IH ENVI cube to SOTA hyperspectral_estimation .mat format.")
    p.add_argument("--hdr-path", required=True, help="Path to ENVI .hdr file")
    p.add_argument(
        "--sota-lambda-mat",
        default="third_party/sota_ozone/data/lambda.mat",
        help="Path to SOTA lambda.mat (variable 'lambda' in µm)",
    )
    p.add_argument(
        "--out-mat",
        required=True,
        help="Output .mat path to write (will contain meas, lambda, T_air)",
    )
    p.add_argument(
        "--t-air",
        type=float,
        default=289.7,
        help="Air temperature (K) saved as T_air in the .mat (default: 289.7)",
    )
    p.add_argument(
        "--crop-last-cols",
        type=int,
        default=None,
        help="Optional: keep only the last N columns (to mimic SOTA cropping).",
    )
    args = p.parse_args()

    cube, src_lambda_um = _load_envi_cube(args.hdr_path)
    tgt_lambda_um = _load_sota_lambda(args.sota_lambda_mat)

    if args.crop_last_cols is not None:
        n = int(args.crop_last_cols)
        if n <= 0 or n > cube.shape[1]:
            raise ValueError(f"--crop-last-cols must be in [1, W], got {n} (W={cube.shape[1]})")
        cube = cube[:, -n:, :]

    band_idx = _nearest_band_indices(src_lambda_um, tgt_lambda_um)
    meas = cube[:, :, band_idx].astype(np.float32, copy=False)

    out_dir = os.path.dirname(os.path.abspath(args.out_mat))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    scipy.io.savemat(
        args.out_mat,
        {
            "meas": meas,
            "lambda": tgt_lambda_um.reshape(1, -1).astype(np.float32),
            "T_air": np.array([[float(args.t_air)]], dtype=np.float32),
        },
        do_compression=True,
    )

    print("Wrote:", args.out_mat)
    print("meas shape:", meas.shape, "dtype:", meas.dtype)
    print(
        "lambda shape:",
        (1, tgt_lambda_um.shape[0]),
        "range(um):",
        float(tgt_lambda_um.min()),
        float(tgt_lambda_um.max()),
    )
    print("T_air:", float(args.t_air))
    print("Max |lambda_src - lambda_tgt| (um):", float(np.max(np.abs(src_lambda_um[band_idx] - tgt_lambda_um))))


if __name__ == "__main__":
    main()

