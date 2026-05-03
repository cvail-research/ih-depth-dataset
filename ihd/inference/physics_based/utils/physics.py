"""
utils/physics.py

Core physics utilities for LWIR ranging.
Direct translation of Dorken et al. blackbody.m and forward_model.m.
"""

import numpy as np


# ── physical constants ───────────────────────────────────────────────────────

H  = 6.63e-34   # Planck (J·s)
C  = 3e8        # speed of light (m/s)
K_B = 1.38e-23  # Boltzmann (J/K)


# ── blackbody ─────────────────────────────────────────────────────────────────

def blackbody(lambda_um, T):
    """
    Blackbody spectral radiance (Planck) in microflicks.
    Direct translation of Dorken et al. blackbody.m.

    Parameters
    ----------
    lambda_um : np.ndarray (K,)
        Wavelengths in µm.
    T : float
        Temperature in Kelvin.

    Returns
    -------
    np.ndarray (K,)
        Radiance in microflicks [µW / (cm² · sr · µm)].
    """
    lambda_m = lambda_um * 1e-6
    exponent = np.clip(H * C / (lambda_m * K_B * T), None, 700)
    return 1e-4 * (2 * H * C**2 / lambda_m**5) / (np.exp(exponent) - 1)


# ── brightness temperature ─────────────────────────────────────────────────────

def brightness_temperature(lambda_um, radiance_microflicks):
    """
    Invert Planck's law to obtain brightness temperature.
    Used to estimate T_air from the maximum-attenuation band.

    Parameters
    ----------
    lambda_um : float
        Wavelength in µm (scalar; selected band).
    radiance_microflicks : np.ndarray (H, W) o escalar
        Measured radiance in microflicks.

    Returns
    -------
    np.ndarray (H, W) o escalar
        Brightness temperature in Kelvin.
    """
    lambda_m = lambda_um * 1e-6
    # Invertir: T = h*c / (λ * k_B * ln(2hc²/(λ⁵ * L * 1e4) + 1))
    L = np.clip(radiance_microflicks, 1e-10, None)
    inner = (2 * H * C**2) / (lambda_m**5 * L * 1e4) + 1
    inner = np.clip(inner, 1 + 1e-10, None)   # ln argument must be > 0
    T = (H * C) / (lambda_m * K_B * np.log(inner))
    return T


def estimate_T_air(measurements, lambda_um, attenuation, lambda_min=8.5, lambda_max=12.0):
    """
    Estimate T_air as the median brightness temperature at the
    maximum-attenuation band within [lambda_min, lambda_max] µm.

    The median over all pixels is more robust than the mean
    in the presence of hot/cold outliers in the scene.

    Parameters
    ----------
    measurements : np.ndarray (H, W, K)
        HSI image in microflicks.
    lambda_um : np.ndarray (K,)
        Sensor wavelengths in µm.
    attenuation : np.ndarray (K,)
        Precomputed attenuation in dB/m.
    lambda_min, lambda_max : float
        Spectral range where to search for the maximum-attenuation band.
        Default 8.5–12.0 µm to avoid noisy edges.

    Returns
    -------
    T_air : float
        Estimated air temperature in Kelvin.
    idx_band : int
        Band index used for the estimate.
    """
    # Find maximum-attenuation band within the safe range
    mask    = (lambda_um >= lambda_min) & (lambda_um <= lambda_max)
    indices = np.where(mask)[0]

    if len(indices) == 0:
        raise ValueError(
            f"No bands in range {lambda_min}–{lambda_max} µm. "
            f"Sensor range: {lambda_um.min():.3f}–{lambda_um.max():.3f} µm."
        )

    idx_band = indices[np.argmax(attenuation[mask])]
    lam_band = lambda_um[idx_band]

    # Brightness temperature at that band, per pixel
    T_map = brightness_temperature(lam_band, measurements[:, :, idx_band])

    # Median → robust to outliers
    T_air = float(np.nanmedian(T_map))

    print(
        f"  Estimated T_air: {T_air:.2f} K  "
        f"(band idx={idx_band}, λ={lam_band:.3f} µm, "
        f"atten={attenuation[idx_band]:.5f} dB/m)"
    )

    return T_air, idx_band


# ── forward model ─────────────────────────────────────────────────────────────

def forward_model(lambda_um, T, emissivity, V, downwelling, attenuation, d, T_air):
    """
    Radiance forward model at the sensor.
    Direct translation of Dorken et al. forward_model.m.

    Parameters
    ----------
    lambda_um   : np.ndarray (1, 1, K, 1)   wavelengths in µm
    T           : np.ndarray (H, W, 1, 1)   object temperature (K)
    emissivity  : np.ndarray (H, W, K, 1)   spectral emissivity
    V           : np.ndarray (H, W, 1, L)   angular downwelling weights
    downwelling : np.ndarray (1, 1, K, L)   downwelling radiance (microflicks)
    attenuation : np.ndarray (1, 1, K, 1)   attenuation (dB/m)
    d           : np.ndarray (H, W, 1, 1)   distance (m)
    T_air       : float                      air temperature (K)

    Returns
    -------
    np.ndarray (H, W, K, 1)
        Simulated radiance at the sensor (microflicks).
    """
    bb_obj = blackbody(lambda_um, T)           # (H, W, K, 1) via broadcasting

    obj_emission  = bb_obj * emissivity
    obj_reflection = (1 - emissivity) * np.sum(V * downwelling, axis=3, keepdims=True)

    tau = 10 ** (-attenuation * d / 10)        # transmittance

    bb_air = blackbody(lambda_um, T_air)       # (1, 1, K, 1)

    radiance_sensor = tau * (obj_emission + obj_reflection) + (1 - tau) * bb_air

    return radiance_sensor