"""
utils/spectral.py
 
Spectral utilities: sensor detection and attenuation resampling.
Based on adjust_spectral_data from cvail-research/deep-thermal-ranging.
"""
 
import numpy as np
from scipy.signal import fftconvolve
from scipy.signal.windows import gaussian
 
 
# ── sensor detection ─────────────────────────────────────────────────────────
 
def detect_sensor(wavelengths):
    """
    Identify the sensor from its wavelength grid (µm).
    Add another branch here if a third sensor is introduced.
 
    Parameters
    ----------
    wavelengths : np.ndarray (K,)
        Sensor wavelengths in µm.
 
    Returns
    -------
    str : 'LWHSI1' o 'LWHSI2'
    """
    wl_min = float(wavelengths.min())
    n_bands = int(len(wavelengths))

    # Robust heuristic:
    # - LWHSI1 typically starts ~8.1µm
    # - LWHSI2 typically starts ~6.84µm
    # Some pipelines crop bands (K != 256/250). In those cases we still
    # classify by spectral range and only warn.
    if wl_min > 7.5:
        sensor = 'LWHSI1'
        expected = 256
    else:
        sensor = 'LWHSI2'
        expected = 250

    if n_bands != expected:
        print(f"  [warn] {sensor}: bands={n_bands} (expected {expected}); assuming {sensor} from λ_min={wl_min:.3f}µm")

    return sensor
 
 
# ── spectral resampling ──────────────────────────────────────────────────────
 
def adjust_spectral_data(data_array, lambda_vals, resolution_nm):
    """
    Smooth and interpolate data_array[:, 1:] onto the lambda_vals grid.
    Implementation from cvail-research/deep-thermal-ranging (spectral_data.py).
 
    Parameters
    ----------
    data_array : np.ndarray (N, 1+M)
        First column: wavelengths in µm. Remaining columns: values to resample.
    lambda_vals : np.ndarray (C,)
        Target grid in µm.
    resolution_nm : float
        Target spectral resolution in nm.
 
    Returns
    -------
    adjusted : np.ndarray (C, M)
    wl_matched : np.ndarray (C,)
    """
    data  = data_array[:, 1:]
    w_std = resolution_nm * 1e-3 / np.mean(np.diff(data_array[:, 0]))
    alpha = len(data_array) / w_std
    std   = (len(data_array) - 1) / (2 * alpha)
 
    w  = gaussian(len(data_array), std=std)
    w /= np.sum(w)
 
    pad_width     = 2 * int(np.ceil(w_std))
    adjusted_data = np.pad(data, ((pad_width, pad_width), (0, 0)),
                           mode='constant', constant_values=1)
    w = np.pad(w, pad_width, mode='constant')
 
    for i in range(data.shape[1]):
        adjusted_data[:, i] = fftconvolve(adjusted_data[:, i], w, mode='same')
 
    wavelength_padded = np.pad(
        data_array[:, 0], pad_width, mode='linear_ramp',
        end_values=(
            data_array[0, 0]  - pad_width * np.mean(np.diff(data_array[:, 0])),
            data_array[-1, 0] + pad_width * np.mean(np.diff(data_array[:, 0]))
        )
    )
 
    index = np.clip(
        np.searchsorted(wavelength_padded, lambda_vals),
        0, len(wavelength_padded) - 1
    )
 
    return adjusted_data[index + pad_width], wavelength_padded[index]
 
 
# ── attenuation precompute ───────────────────────────────────────────────────
 
def precompute_attenuation(sensor_wavelengths, std_transmittance_path):
    """
    Resample the standard transmittance onto the sensor grid
    and convert it to attenuation in dB/m.
 
    Parameters
    ----------
    sensor_wavelengths : np.ndarray (K,)
        Sensor wavelengths in µm.
    std_transmittance_path : str
        Path to the .npy with columns [wavelength_µm, transmittance, ...].
 
    Returns
    -------
    attenuation : np.ndarray (K,)
        Attenuation in dB/m for each sensor channel.
    wl_matched : np.ndarray (K,)
        Wavelengths effectively used after resampling.
    """
    std       = np.load(std_transmittance_path)
    std_array = np.column_stack([std[:, 0], std[:, 1]])  # [λ, transmittance]
 
    res_nm = np.mean(np.diff(sensor_wavelengths)) * 1e3  # nm
 
    trans_resampled, wl_matched = adjust_spectral_data(
        std_array, sensor_wavelengths, res_nm
    )
 
    # transmittance → dB/m: -10·log10(T)
    attenuation = -10 * np.log10(np.clip(trans_resampled[:, 0], 1e-10, 1.0))
 
    return attenuation, wl_matched