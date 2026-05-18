"""
utils/io_utils.py

Data loading: HSI image, attenuation, downwelling, and lidar ground truth.
"""

from pathlib import Path

import numpy as np
import scipy.io as sio
import spectral as spy

from baselines.physics_based.utils.spectral import (
    adjust_spectral_data,
    detect_sensor,
    precompute_attenuation,
)
 
 
# ── per-sensor file mapping ─────────────────────────────────────────────────-
 
DOWNWELLING_FILES = {
    "LWHSI1": "downwelling_IHTest202104_LWHSI1.npz",
    "LWHSI2": "downwelling_IHTest202204_LWHSI2.npz",
}


def _load_attenuation(
    sensor: str,
    lambda_um: np.ndarray,
    standard_root: Path,
    ozone_cues_root: Path,
    profile: str,
) -> np.ndarray:
    if profile == "auto":
        profile = "ozone_cues" if sensor == "LWHSI1" else "standard"

    if profile == "standard":
        std_trans_path = standard_root / "transmittance_atten_1mAir.npy"
        if not std_trans_path.exists():
            raise FileNotFoundError(f"Missing standard transmittance file: {std_trans_path}")
        attenuation, _ = precompute_attenuation(
            sensor_wavelengths=np.asarray(lambda_um, dtype=np.float64).reshape(-1),
            std_transmittance_path=str(std_trans_path),
        )
        return np.asarray(attenuation, dtype=np.float64).reshape(-1)

    if profile == "ozone_cues":
        if sensor != "LWHSI1":
            raise ValueError(
                "ozone_cues attenuation is currently available only for LWHSI1. "
                f"Got sensor={sensor}."
            )
        npz_path = ozone_cues_root / "attenuation.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing ozone-cues attenuation file: {npz_path}")
        z = np.load(npz_path)
        if "attenuation" in z:
            return np.asarray(z["attenuation"], dtype=np.float64).reshape(-1)
        if len(z.files) != 1:
            raise KeyError(
                f"{npz_path} does not contain 'attenuation' and has multiple arrays: {z.files}"
            )
        return np.asarray(z[z.files[0]], dtype=np.float64).reshape(-1)

    raise ValueError(f"Unknown attenuation profile: {profile}")


# ── scene loading ─────────────────────────────────────────────────────────---

def load_scene(hsi_path, ozone_cues_root, standard_root, attenuation_profile="auto"):
    """
    Load all data required to run a baseline.
 
    Parameters
    ----------
    hsi_path : str
        Path to the HSI .hdr file.
    ozone_cues_root : str
        Directory containing paper attenuation files, including attenuation.npz.
    standard_root : str
        Directory containing standard atmospheric assets (downwelling, transmittance).
 
    Returns
    -------
    meas        : np.ndarray (H, W, K)  HSI image in microflicks
    lambda_um   : np.ndarray (K,)       sensor wavelengths in µm
    attenuation : np.ndarray (K,)       attenuation in dB/m
    downwelling : np.ndarray (10, K)    downwelling radiance per angle
    sensor      : str                   'LWHSI1' or 'LWHSI2'
    """
    # Read HDR
    img       = spy.open_image(hsi_path)
    lambda_um = np.array(img.metadata['wavelength']).astype(np.float64)
    meas      = img.load().astype(np.float64)   # (H, W, K)
 
    # Detect sensor
    sensor = detect_sensor(lambda_um)
    print(f"  Sensor: {sensor} | {len(lambda_um)} bands | "
          f"λ {lambda_um.min():.3f}–{lambda_um.max():.3f} µm")
 
    # Load attenuation profile
    attenuation = _load_attenuation(
        sensor=sensor,
        lambda_um=lambda_um,
        standard_root=Path(standard_root),
        ozone_cues_root=Path(ozone_cues_root),
        profile=str(attenuation_profile),
    )
 
    # Load downwelling
    dw_path = Path(standard_root) / DOWNWELLING_FILES[sensor]
    if not dw_path.exists():
        raise FileNotFoundError(f"Missing downwelling file: {dw_path}")
    dw_data = np.load(dw_path)
    downwelling = dw_data["downwelling"].astype(np.float64)  # (10, K)
    dw_lambda_um = (
        np.asarray(dw_data.get("wavelength", None), dtype=np.float64).reshape(-1)
        if "wavelength" in dw_data
        else None
    )

    # --- Validation / resampling to the HDR grid (robustness) ---
    # Usually K matches per sensor, but some pipelines store subsets or slightly different grids.
    if dw_lambda_um is not None and (len(dw_lambda_um) != len(lambda_um) or np.max(np.abs(dw_lambda_um - lambda_um)) > 1e-6):
        # Resample downwelling (10, K_src) → (10, K_tgt)
        res_nm = float(np.mean(np.diff(lambda_um)) * 1e3) if len(lambda_um) > 1 else 10.0
        dw_array = np.column_stack([dw_lambda_um, downwelling.T])  # (K_src, 1+10)
        dw_adj, _ = adjust_spectral_data(dw_array, lambda_um, res_nm)  # (K_tgt, 10)
        downwelling = dw_adj.T
        print("  Downwelling resampled to the HDR grid")

    if len(attenuation) != len(lambda_um):
        # Resample attenuation (K_src,) → (K_tgt,)
        # Note: attenuation_{sensor}.npy is typically on the sensor grid; if the HDR was cropped,
        # this keeps everything consistent.
        if dw_lambda_um is None:
            raise ValueError(
                "Cannot resample attenuation: band mismatch and downwelling .npz has no wavelength grid."
            )
        res_nm = float(np.mean(np.diff(lambda_um)) * 1e3) if len(lambda_um) > 1 else 10.0
        att_array = np.column_stack([dw_lambda_um, attenuation.reshape(-1, 1)])
        att_adj, _ = adjust_spectral_data(att_array, lambda_um, res_nm)  # (K_tgt, 1)
        attenuation = att_adj[:, 0]
        print("  Attenuation resampled to the HDR grid")
 
    return meas, lambda_um, attenuation, downwelling, sensor
 
 
# ── lidar loading ─────────────────────────────────────────────────────────---
 
def load_lidar(lidar_path, key='depthMapReg'):
    """
    Load distance ground truth from a MATLAB .mat file.
 
    The lidar.mat file typically contains:
        depthMap    (260, 1600)  raw depth map
        depthMapReg (260, 1600)  depth map registered to the HSI sensor  ← use this
        tempMap     (260, 1600)  temperature map
 
    Parameters
    ----------
    lidar_path : str
        Path to lidar.mat.
    key : str
        Variable to load. Default 'depthMapReg' (registered to HSI).
 
    Returns
    -------
    depth : np.ndarray (H, W)
        Depth map in meters. NaN where no valid measurement exists.
    """
    if not Path(lidar_path).exists():
        raise FileNotFoundError(f"Missing lidar file: {lidar_path}")
 
    # Try scipy first (.mat v5/v7). If that fails, try h5py (.mat v7.3).
    try:
        mat = sio.loadmat(lidar_path)
        depth = mat[key].astype(np.float64)
    except NotImplementedError:
        try:
            import h5py
            with h5py.File(lidar_path, 'r') as f:
                depth = np.array(f[key]).astype(np.float64).T  # h5py stores MATLAB arrays transposed
        except ImportError:
            raise ImportError(
                "This .mat file is v7.3. Install h5py: pip install h5py"
            )
 
    # Mark zeros/negatives as NaN (no valid measurement)
    depth[depth <= 0] = np.nan

    print(
        f"  Lidar loaded: shape={depth.shape}, "
        f"valid={np.sum(~np.isnan(depth))}/{depth.size}, "
        f"range={np.nanmin(depth):.1f}–{np.nanmax(depth):.1f} m"
    )
 
    return depth
 
