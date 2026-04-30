from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import spectral as spy


def load_hyperspectral_cube(hdr_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an ENVI hyperspectral cube as H x W x B float32 plus wavelengths in meters."""
    img = spy.open_image(str(hdr_path))
    cube = np.asarray(img.load(), dtype=np.float32)
    wavelengths = np.asarray(
        [float(w) * 1e-6 for w in img.metadata.get("wavelength", [])],
        dtype=np.float32,
    )
    return cube, wavelengths


def hsi_to_pseudobroadband_rgb(hsi: np.ndarray) -> np.ndarray:
    """Sum all bands, min-max normalize, and replicate to 3-channel uint8 RGB."""
    cube = np.nan_to_num(np.asarray(hsi, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    gray = np.sum(cube, axis=2)
    lo = float(np.min(gray))
    hi = float(np.max(gray))
    if hi > lo:
        gray = (gray - lo) / (hi - lo)
    else:
        gray = np.zeros_like(gray, dtype=np.float32)
    gray_u8 = np.clip(gray * 255.0, 0.0, 255.0).astype(np.uint8)
    return np.repeat(gray_u8[:, :, None], 3, axis=2)


def load_pseudobroadband_rgb(hdr_path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    hsi, wavelengths_m = load_hyperspectral_cube(hdr_path)
    rgb = hsi_to_pseudobroadband_rgb(hsi)
    meta = {
        "hdr_path": str(hdr_path),
        "input_encoding": "pseudo_broadband_sum_all_bands_minmax_rgb",
        "hsi_shape": list(hsi.shape),
        "num_wavelengths": int(len(wavelengths_m)),
    }
    if len(wavelengths_m):
        meta["wavelength_min_m"] = float(np.min(wavelengths_m))
        meta["wavelength_max_m"] = float(np.max(wavelengths_m))
    return rgb, meta


def save_depth_prediction(
    depth_m: np.ndarray,
    out_dir: str | Path,
    model_name: str,
    hdr_path: str | Path,
    metadata: dict[str, Any] | None = None,
    save_visualization: bool = True,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    depth = np.asarray(depth_m, dtype=np.float32)
    npz_path = out / "depth_prediction.npz"
    payload = {
        "depth_m": depth,
        "model_name": np.asarray(model_name),
        "hdr_path": np.asarray(str(hdr_path)),
        "units": np.asarray("meters"),
    }
    if metadata:
        for key, value in metadata.items():
            if np.isscalar(value) or isinstance(value, str):
                payload[key] = np.asarray(value)
    np.savez_compressed(npz_path, **payload)
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "hdr_path": str(hdr_path),
                "depth_npz": str(npz_path),
                "depth_shape": list(depth.shape),
                "depth_min_m": float(np.nanmin(depth)) if np.isfinite(depth).any() else None,
                "depth_max_m": float(np.nanmax(depth)) if np.isfinite(depth).any() else None,
                **(metadata or {}),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if save_visualization:
        save_depth_visualization(depth, out / "depth_prediction.png")
    return npz_path


def save_depth_visualization(depth_m: np.ndarray, out_path: str | Path) -> None:
    depth = np.asarray(depth_m, dtype=np.float32)
    finite = np.isfinite(depth)
    if not np.any(finite):
        Image.fromarray(np.zeros(depth.shape, dtype=np.uint8)).save(out_path)
        return
    plt.figure(figsize=(12, 3))
    plt.imshow(depth, cmap="viridis")
    plt.axis("off")
    cbar = plt.colorbar(fraction=0.025, pad=0.01)
    cbar.set_label("Depth (m)")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=160, bbox_inches="tight", pad_inches=0)
    plt.close()


def read_prediction_input_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Prediction input manifest is empty: {path}")
    if "hdr_path" not in rows[0]:
        raise ValueError(f"Prediction input manifest must contain hdr_path: {path}")
    return rows


def scene_out_dir(out_root: str | Path, model_slug: str, row: dict[str, str]) -> Path:
    collection = row.get("collection") or "unknown_collection"
    path = row.get("path") or "unknown_path"
    step = row.get("step") or Path(row["hdr_path"]).stem
    return Path(out_root) / model_slug / collection / path / step


def write_prediction_manifest(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

