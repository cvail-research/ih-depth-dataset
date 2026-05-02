from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
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


def _step_number(step: str) -> int:
    match = re.search(r"step(\d+)", str(step), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse step number from {step}")
    return int(match.group(1))


def _path_number(path: str) -> int:
    match = re.search(r"path(\d+)", str(path), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse path number from {path}")
    return int(match.group(1))


def _collection_tag(collection: str) -> str:
    match = re.match(r"^(IHTest_\d{6})_DistStA", collection)
    if not match:
        return collection.split("_DistStA")[0]
    return match.group(1)


def _find_hdr(row: pd.Series, disk_root: Path) -> str | None:
    if "hdr_path" in row and pd.notna(row["hdr_path"]) and Path(str(row["hdr_path"])).exists():
        return str(row["hdr_path"])
    if "disk_reference" in row and pd.notna(row["disk_reference"]):
        scene_dir = Path(str(row["disk_reference"])).parent
        hdrs = sorted(scene_dir.glob("*LWHSI1*.hdr"))
        if hdrs:
            collect0 = [p for p in hdrs if "collect0" in p.name]
            return str((collect0 or hdrs)[0])

    collection = str(row["collection"])
    pnum = _path_number(str(row["path"]))
    snum = _step_number(str(row["step"]))
    tag = _collection_tag(collection)
    path_dir = disk_root / collection / f"Path{pnum}_DistStA"
    candidates = [
        path_dir / f"Path{pnum}_Step{snum}" / f"{tag}_Path{pnum}_Step{snum}_LWHSI1_DistStA.hdr",
        path_dir / f"Path{pnum}_Step{snum}_DistStA" / f"{tag}_Path{pnum}_Step{snum}_LWHSI1_collect0_DistStA.hdr",
        path_dir / f"Path{pnum}_Step{snum}_DistStA" / f"{tag}_Path{pnum}_Step{snum}_LWHSI1_DistStA.hdr",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    step_dirs = sorted(path_dir.glob(f"Path{pnum}_Step{snum}*"))
    for step_dir in step_dirs:
        hdrs = sorted(step_dir.glob("*LWHSI1*.hdr"))
        if hdrs:
            collect0 = [p for p in hdrs if "collect0" in p.name]
            return str((collect0 or hdrs)[0])
    return None


def _label_path(row: pd.Series, depth_label_root: Path) -> str | None:
    if "label_path" in row and pd.notna(row["label_path"]) and Path(str(row["label_path"])).exists():
        return str(row["label_path"])
    p = depth_label_root / str(row["collection"]) / str(row["path"]) / str(row["step"]) / "projected_lidar_depth_label.npz"
    return str(p) if p.exists() else None


def build_prediction_input_rows_from_scene_manifest(
    scene_manifest: str | Path,
    depth_label_root: str | Path = "analysis/depth_labels/platform_sphere_r4p0",
    disk_root: str | Path = "/disk",
    limit: int | None = None,
) -> list[dict[str, str]]:
    df = pd.read_csv(scene_manifest)
    rows: list[dict[str, str]] = []
    for row in df.sort_values(["collection", "path", "step"]).itertuples(index=False):
        series = pd.Series(row._asdict())
        hdr = _find_hdr(series, Path(disk_root))
        label = _label_path(series, Path(depth_label_root))
        if not hdr or not label:
            continue
        scene = series.get("scene") or series.get("scene_id") or f"{series['collection']} / {series['path']} / {series['step']}"
        rows.append(
            {
                "scene": scene,
                "collection": series["collection"],
                "path": series["path"],
                "step": series["step"],
                "hdr_path": hdr,
                "label_path": label,
            }
        )
        if limit and len(rows) >= limit:
            break
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
