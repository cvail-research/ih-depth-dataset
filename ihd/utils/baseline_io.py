from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import spectral as spy

from .depth_png import save_depth_png


def canonical_prediction_filename(hdr_path: str | Path) -> str:
    return f"{Path(hdr_path).stem}_depth.png"


def load_hyperspectral_cube(hdr_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    img = spy.open_image(str(hdr_path))
    cube = np.asarray(img.load(), dtype=np.float32)
    wavelengths = np.asarray([float(w) * 1e-6 for w in img.metadata.get("wavelength", [])], dtype=np.float32)
    return cube, wavelengths


def hsi_to_pseudobroadband_rgb(hsi: np.ndarray) -> np.ndarray:
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
    meta: dict[str, Any] = {
        "hdr_path": str(hdr_path),
        "input_encoding": "pseudo_broadband_sum_all_bands_minmax_rgb",
        "hsi_shape": list(hsi.shape),
        "num_wavelengths": int(len(wavelengths_m)),
    }
    if len(wavelengths_m):
        meta["wavelength_min_m"] = float(np.min(wavelengths_m))
        meta["wavelength_max_m"] = float(np.max(wavelengths_m))
    return rgb, meta


def save_depth_visualization(depth_m: np.ndarray, out_path: str | Path) -> None:
    depth = np.asarray(depth_m, dtype=np.float32)
    finite = np.isfinite(depth) & (depth > 0.0)
    if not np.any(finite):
        Image.fromarray(np.zeros(depth.shape, dtype=np.uint8)).save(out_path)
        return
    depth_vis = depth.copy()
    depth_vis[~finite] = np.nan
    plt.figure(figsize=(12, 3))
    plt.imshow(depth_vis, cmap="viridis")
    plt.axis("off")
    cbar = plt.colorbar(fraction=0.025, pad=0.01)
    cbar.set_label("Depth (m)")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=160, bbox_inches="tight", pad_inches=0)
    plt.close()


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
    png_path = save_depth_png(depth, out / canonical_prediction_filename(hdr_path))
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "hdr_path": str(hdr_path),
                "prediction_png": str(png_path),
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
        save_depth_visualization(depth, out / "depth_prediction_preview.png")
    return png_path


def save_input_prediction_groundtruth_figures(
    *,
    input_gray_u8: np.ndarray,
    prediction_m: np.ndarray,
    out_dir: str | Path,
    ground_truth_m: np.ndarray | None = None,
    ground_truth_mask: np.ndarray | None = None,
) -> None:
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    input_img = np.asarray(input_gray_u8, dtype=np.uint8)
    pred = np.asarray(prediction_m, dtype=np.float32)
    gt = np.asarray(ground_truth_m, dtype=np.float32) if ground_truth_m is not None else None
    gt_mask = np.asarray(ground_truth_mask, dtype=bool) if ground_truth_mask is not None else None

    if gt is not None and gt_mask is not None and np.any(gt_mask):
        shared_vmin = float(np.nanmin(gt[gt_mask]))
        shared_vmax = float(np.nanmax(gt[gt_mask]))
    else:
        finite_pred = np.isfinite(pred) & (pred > 0.0)
        shared_vmin = float(np.nanmin(pred[finite_pred])) if np.any(finite_pred) else 0.0
        shared_vmax = float(np.nanmax(pred[finite_pred])) if np.any(finite_pred) else 1.0

    def _append_empty_colorbar_axis(ax) -> None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.8%", pad=0.04)
        cax.set_xticks([])
        cax.set_yticks([])
        for spine in cax.spines.values():
            spine.set_visible(False)
        cax.patch.set_alpha(0.0)

    def _save_single_input() -> None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 3))
        ax.imshow(input_img, cmap="gray", aspect="auto")
        ax.axis("off")
        _append_empty_colorbar_axis(ax)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.02)
        fig.savefig(out / "input.png", dpi=180, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    def _save_single_depth(depth: np.ndarray, filename: str, *, vmin: float, vmax: float) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 3))
        im = ax.imshow(depth, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.axis("off")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.8%", pad=0.04)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Depth (m)", labelpad=12)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.02)
        fig.savefig(out / filename, dpi=180, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    _save_single_input()
    _save_single_depth(pred, "prediction.png", vmin=shared_vmin, vmax=shared_vmax)

    if gt is None or gt_mask is None:
        return
    gt_vis = np.where(gt_mask, gt, np.nan)
    _save_single_depth(gt_vis, "ground_truth.png", vmin=shared_vmin, vmax=shared_vmax)


def read_prediction_input_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
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
    collection = str(row["collection"])
    pnum = _path_number(str(row["path"]))
    snum = _step_number(str(row["step"]))
    tag = _collection_tag(collection)
    path_dir = disk_root / collection / f"Path{pnum}_DistStA"
    candidates = [
        path_dir / f"Path{pnum}_Step{snum}" / f"{tag}_Path{pnum}_Step{snum}_LWHSI1_DistStA.hdr",
        path_dir / f"Path{pnum}_Step{snum}" / f"{tag}_Path{pnum}_Step{snum}_LWHSI2_DistStA.hdr",
        path_dir / f"Path{pnum}_Step{snum}_DistStA" / f"{tag}_Path{pnum}_Step{snum}_LWHSI1_collect0_DistStA.hdr",
        path_dir / f"Path{pnum}_Step{snum}_DistStA" / f"{tag}_Path{pnum}_Step{snum}_LWHSI2_collect0_DistStA.hdr",
        path_dir / f"Path{pnum}_Step{snum}_DistStA" / f"{tag}_Path{pnum}_Step{snum}_LWHSI1_DistStA.hdr",
        path_dir / f"Path{pnum}_Step{snum}_DistStA" / f"{tag}_Path{pnum}_Step{snum}_LWHSI2_DistStA.hdr",
        path_dir / f"Path{pnum}_Step{snum}_DistStA" / f"{tag}_Path{pnum}_Step{snum}_LWHSI1__DistStA.hdr",
        path_dir / f"Path{pnum}_Step{snum}_DistStA" / f"{tag}_Path{pnum}_Step{snum}_LWHSI2__DistStA.hdr",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    step_dirs = sorted(path_dir.glob(f"Path{pnum}_Step{snum}*"))
    for step_dir in step_dirs:
        hdrs = sorted(step_dir.glob("*LWHSI*.hdr"))
        if hdrs:
            collect0 = [path for path in hdrs if "collect0" in path.name]
            return str((collect0 or hdrs)[0])
    return None


def _label_path(row: pd.Series, depth_label_root: Path) -> str | None:
    if "label_path" in row and pd.notna(row["label_path"]) and Path(str(row["label_path"])).exists():
        return str(row["label_path"])
    path = depth_label_root / str(row["collection"]) / str(row["path"]) / str(row["step"]) / "projected_lidar_depth_label.npz"
    return str(path) if path.exists() else None


def infer_sensor_metadata(hdr_path: str | Path) -> tuple[str | None, int | None]:
    path = Path(hdr_path)
    sensor_id: str | None = None
    name = path.name.upper()
    if "LWHSI1" in name:
        sensor_id = "LWHSI1"
    elif "LWHSI2" in name:
        sensor_id = "LWHSI2"

    sensor_num_bands: int | None = None
    try:
        img = spy.open_image(str(path))
        sensor_num_bands = int(len(img.metadata.get("wavelength", [])))
        if sensor_id is None:
            if sensor_num_bands == 256:
                sensor_id = "LWHSI1"
            elif sensor_num_bands == 250:
                sensor_id = "LWHSI2"
    except Exception:
        pass
    return sensor_id, sensor_num_bands


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
        sensor_id, sensor_num_bands = infer_sensor_metadata(hdr)
        rows.append(
            {
                "scene": scene,
                "collection": series["collection"],
                "path": series["path"],
                "step": series["step"],
                "hdr_path": hdr,
                "label_path": label,
                "sensor_id": sensor_id,
                "sensor_num_bands": sensor_num_bands,
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
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
