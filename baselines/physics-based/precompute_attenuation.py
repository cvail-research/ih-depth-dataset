"""precompute_attenuation.py

Generate attenuation profiles (dB/m) per sensor (LWHSI1 / LWHSI2).

Rationale:
- There are two sensors with different spectral grids.
- The standard transmittance is unique; it is resampled to each sensor grid.

This script uses the `wavelength` stored in the downwelling .npz files to
ensure an exact match to the wavelength grid expected by each sensor.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from utils.spectral import precompute_attenuation


DOWNWELLING_FILES = {
    "LWHSI1": "downwelling_IHTest202104_LWHSI1.npz",
    "LWHSI2": "downwelling_IHTest202204_LWHSI2.npz",
}


def _load_sensor_wavelengths(downwelling_npz: Path) -> np.ndarray:
    z = np.load(downwelling_npz)
    if "wavelength" not in z:
        raise KeyError(f"{downwelling_npz} does not contain key 'wavelength'")
    wl = np.asarray(z["wavelength"], dtype=np.float64)
    if wl.ndim != 1:
        wl = wl.reshape(-1)
    return wl


def precompute_all(data_dir: Path, std_trans_path: Path, out_dir: Path, *, overwrite: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for sensor, fname in DOWNWELLING_FILES.items():
        npz_path = data_dir / fname
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing downwelling for {sensor}: {npz_path}")

        out_path = out_dir / f"attenuation_{sensor}.npy"
        if out_path.exists() and not overwrite:
            print(f"[skip] {out_path}")
            continue

        sensor_wl = _load_sensor_wavelengths(npz_path)
        attenuation, wl_matched = precompute_attenuation(sensor_wl, str(std_trans_path))

        np.save(out_path, attenuation.astype(np.float32))
        print(
            f"[ok] {sensor}: saved {out_path.name} | bands={len(attenuation)} | "
            f"λ {wl_matched.min():.3f}–{wl_matched.max():.3f} µm"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    default_data = here / "data"
    p.add_argument(
        "--data-dir",
        type=Path,
        default=default_data,
        help="Directory containing downwelling_*.npz and transmittance_atten_1mAir.npy",
    )
    p.add_argument(
        "--std-trans",
        type=Path,
        default=None,
        help="Path to the standard transmittance .npy (default: data/transmittance_atten_1mAir.npy)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/precomputed)",
    )
    p.add_argument("--overwrite", action="store_true", help="Recompute even if it already exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    std_trans_path = args.std_trans or (data_dir / "transmittance_atten_1mAir.npy")
    out_dir = args.out_dir or (data_dir / "precomputed")

    if not std_trans_path.exists():
        raise FileNotFoundError(f"Missing standard transmittance: {std_trans_path}")

    precompute_all(data_dir, std_trans_path, out_dir, overwrite=bool(args.overwrite))


if __name__ == "__main__":
    main()