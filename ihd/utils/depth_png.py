from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


DEPTH_SCALE = 128.0


def encode_depth_u16(depth_m: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth map, got shape {depth.shape}.")
    encoded = np.zeros(depth.shape, dtype=np.uint16)
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        return encoded
    scaled = np.rint(depth[valid] * DEPTH_SCALE)
    if np.any(scaled < 1.0):
        raise ValueError("Valid depth values must encode to at least 1.")
    if np.any(scaled > np.iinfo(np.uint16).max):
        raise ValueError(
            f"Depth exceeds uint16 range at scale {DEPTH_SCALE:g}; "
            f"max representable depth is {np.iinfo(np.uint16).max / DEPTH_SCALE:.6f} m."
        )
    encoded[valid] = scaled.astype(np.uint16)
    return encoded


def decode_depth_u16(encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth_u16 = np.asarray(encoded)
    if depth_u16.ndim != 2:
        raise ValueError(f"Expected a single-channel 2D uint16 PNG, got shape {depth_u16.shape}.")
    if depth_u16.dtype != np.uint16:
        raise ValueError(f"Expected uint16 PNG data, got {depth_u16.dtype}.")
    valid_mask = depth_u16 > 0
    depth_m = depth_u16.astype(np.float32) / np.float32(DEPTH_SCALE)
    depth_m[~valid_mask] = 0.0
    return depth_m, valid_mask


def load_depth_png(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    file_path = Path(path)
    try:
        with Image.open(file_path) as image:
            data = np.asarray(image)
            mode = image.mode
    except Exception as exc:
        raise ValueError(f"Could not read PNG {file_path}: {exc}") from exc
    if data.dtype != np.uint16:
        raise ValueError(f"Invalid PNG encoding for {file_path}: expected uint16, got mode={mode!r}, dtype={data.dtype}.")
    if data.ndim != 2:
        raise ValueError(f"Invalid PNG encoding for {file_path}: expected a single channel image, got shape {data.shape}.")
    return decode_depth_u16(data)


def save_depth_png(depth_m: np.ndarray, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(encode_depth_u16(depth_m), mode="I;16").save(out_path)
    return out_path
