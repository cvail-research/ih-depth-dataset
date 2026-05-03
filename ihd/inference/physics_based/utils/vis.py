"""utils/vis.py

Small visualization helpers for baselines.

All functions are safe to run in headless (Slurm) environments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def save_distance_png(
    d_m: np.ndarray,
    out_path: Path,
    *,
    title: Optional[str] = None,
    vmin: float = 0.0,
    vmax: float = 150.0,
    cmap: str = "viridis_r",
) -> None:
    """Save a distance map as a PNG.

    Parameters
    ----------
    d_m : np.ndarray
        Distance map in meters. NaNs are treated as invalid pixels.
    out_path : Path
        Where to write the PNG.
    title : Optional[str]
        Figure title.
    vmin, vmax : float
        Color scaling limits.
    cmap : str
        Matplotlib colormap name.
    """

    # Import matplotlib lazily so core baselines can run without it if needed.
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    d_show = np.array(d_m, dtype=np.float64, copy=True)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    im = ax.imshow(d_show, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    if title:
        ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.046, pad=0.06)
    cbar.set_label("Distance (m)")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_error_png(
    pred_m: np.ndarray,
    gt_m: np.ndarray,
    out_path: Path,
    *,
    title: Optional[str] = None,
    vmin: float = -20.0,
    vmax: float = 20.0,
    cmap: str = "coolwarm",
) -> None:
    """Save a signed error map (pred - gt) as a PNG.

    NaNs in either input are treated as invalid pixels.
    """

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pred = np.asarray(pred_m, dtype=np.float64)
    gt = np.asarray(gt_m, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch for error map: pred={pred.shape} vs gt={gt.shape}")

    err = pred - gt
    invalid = ~np.isfinite(err)
    err = err.copy()
    err[invalid] = np.nan

    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    im = ax.imshow(err, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    if title:
        ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.046, pad=0.06)
    cbar.set_label("Error (m) = pred - gt")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
