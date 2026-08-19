from __future__ import annotations

import argparse
import math
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from ihd.utils.baseline_io import (
    load_hyperspectral_cube,
    read_prediction_input_manifest,
    save_depth_prediction,
    scene_out_dir,
    valid_voxel_mask,
    write_prediction_manifest,
)


MODEL_SLUG = "depthanythingv2_hsi_patch"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run Depth Anything V2 with a hyperspectral patch embedding. "
            "The RGB DINO patch projection is replaced with a B-channel projection."
        )
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hdr", help="Single ENVI .hdr path.")
    src.add_argument("--manifest", help="CSV with hdr_path and optional label_path columns.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-name", default="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf")
    ap.add_argument(
        "--model-revision",
        help="Optional Hugging Face Hub revision/commit SHA passed to from_pretrained for reproducible weights.",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--input-height", type=int, default=518)
    ap.add_argument("--normalization", default="per-band-standardize", choices=["per-band-standardize", "per-band-minmax"])
    ap.add_argument(
        "--no-voxel-mask",
        action="store_true",
        help="Skip corrupted-voxel/bad-scan-line masking and normalize the raw cube as-is.",
    )
    ap.add_argument("--no-vis", action="store_true")
    return ap.parse_args()


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return int(math.ceil(float(value) / float(multiple)) * multiple)


def dino_compatible_size(height: int, width: int, target_height: int, patch_size: int) -> tuple[int, int]:
    scale = float(target_height) / float(height)
    resized_h = _round_up_to_multiple(target_height, patch_size)
    resized_w = _round_up_to_multiple(max(1, int(round(width * scale))), patch_size)
    return resized_h, resized_w


def normalize_hsi_cube(cube: np.ndarray, mode: str, mask_invalid: bool = True) -> np.ndarray:
    
    if mode not in ("per-band-minmax", "per-band-standardize"):
        raise ValueError(f"Unknown HSI normalization: {mode}")

    with np.errstate(invalid="ignore"):
        hsi = np.asarray(cube, dtype=np.float32)
    # Both branches below hand back a fresh array, so the rest of this function
    # can work in place without touching the caller's cube.
    if mask_invalid:
        hsi = np.where(valid_voxel_mask(hsi), hsi, np.nan)
    else:
        hsi = np.nan_to_num(hsi, nan=0.0, posinf=0.0, neginf=0.0)

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        if mode == "per-band-minmax":
            offset = _band_stat(np.nanmin, hsi, fill=0.0)
            scale = _band_stat(np.nanmax, hsi, fill=0.0) - offset
        else:
            offset = _band_stat(np.nanmean, hsi, fill=0.0)
            scale = _band_stat(np.nanstd, hsi, fill=1.0)
        # A band with no usable spread -- fully rejected, constant, or with
        # statistics that overflowed -- is passed through instead of being
        # scaled to infinity.
        scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0).astype(np.float32)
        hsi -= offset
        hsi /= scale
    if not mask_invalid:
        return hsi
    # Rejected voxels land on 0, the normalized position of the band's own
    # minimum (minmax) or mean (standardize).
    return np.nan_to_num(hsi, nan=0.0, posinf=0.0, neginf=0.0, copy=False)


def _band_stat(reduce_fn, hsi: np.ndarray, fill: float) -> np.ndarray:
    """Per-band statistic over the valid voxels, with unusable bands filled."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # fully rejected bands are filled below
        stat = reduce_fn(hsi, axis=(0, 1), keepdims=True)
    return np.where(np.isfinite(stat), stat, fill).astype(np.float32)


def load_hsi_tensor(hdr_path: str | Path, *, normalization: str, mask_invalid: bool = True):
    import torch

    cube, wavelengths_m = load_hyperspectral_cube(hdr_path)
    norm = normalize_hsi_cube(cube, normalization, mask_invalid=mask_invalid)
    tensor = torch.from_numpy(norm).permute(2, 0, 1).float()
    meta: dict[str, Any] = {
        "hdr_path": str(hdr_path),
        "input_encoding": f"full_hsi_{normalization}" + ("_masked" if mask_invalid else ""),
        "voxel_masking": bool(mask_invalid),
        "hsi_shape": list(cube.shape),
        "num_hsi_channels": int(cube.shape[2]),
        "num_wavelengths": int(len(wavelengths_m)),
    }
    if len(wavelengths_m):
        meta["wavelength_min_m"] = float(np.min(wavelengths_m))
        meta["wavelength_max_m"] = float(np.max(wavelengths_m))
    return tensor, meta


def adapt_depthanythingv2_patch_embedding(model, num_channels: int) -> None:
    import torch
    import torch.nn as nn

    patch_embeddings = model.backbone.embeddings.patch_embeddings
    old_projection = patch_embeddings.projection
    if old_projection.in_channels == num_channels:
        patch_embeddings.num_channels = num_channels
        model.backbone.config.num_channels = num_channels
        return

    new_projection = nn.Conv2d(
        in_channels=num_channels,
        out_channels=old_projection.out_channels,
        kernel_size=old_projection.kernel_size,
        stride=old_projection.stride,
        padding=old_projection.padding,
        dilation=old_projection.dilation,
        groups=old_projection.groups,
        bias=old_projection.bias is not None,
        padding_mode=old_projection.padding_mode,
    )
    with torch.no_grad():
        mean_kernel = old_projection.weight.mean(dim=1, keepdim=True)
        new_projection.weight.copy_(mean_kernel.repeat(1, num_channels, 1, 1) * (3.0 / float(num_channels)))
        if old_projection.bias is not None:
            new_projection.bias.copy_(old_projection.bias)
    new_projection = new_projection.to(device=old_projection.weight.device, dtype=old_projection.weight.dtype)
    patch_embeddings.projection = new_projection
    patch_embeddings.num_channels = num_channels
    model.backbone.config.num_channels = num_channels
    if hasattr(model.config, "backbone_config"):
        model.config.backbone_config.num_channels = num_channels


def load_model(model_name: str, device: str, num_channels: int, model_revision: str | None = None):
    import torch
    from transformers import AutoModelForDepthEstimation

    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device '{device}', but CUDA is not available.")
    actual_device = torch.device(device)
    model = AutoModelForDepthEstimation.from_pretrained(model_name, revision=model_revision)
    adapt_depthanythingv2_patch_embedding(model, num_channels)
    model = model.to(actual_device).eval()
    return model, actual_device


def predict_hsi_tensor(
    model,
    device,
    hsi_tensor,
    meta: dict[str, Any],
    hdr_path: str,
    out_dir: Path,
    model_name: str,
    model_revision: str | None,
    *,
    input_height: int,
    save_vis: bool,
) -> Path:
    import torch
    import torch.nn.functional as F

    _, orig_h, orig_w = hsi_tensor.shape
    patch_size = int(model.backbone.embeddings.patch_embeddings.projection.kernel_size[0])
    resized_h, resized_w = dino_compatible_size(orig_h, orig_w, input_height, patch_size)
    pixel_values = F.interpolate(
        hsi_tensor.unsqueeze(0),
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
    ).to(device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        pred = outputs.predicted_depth
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        pred = F.interpolate(pred, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    depth = pred.squeeze().detach().cpu().numpy().astype(np.float32)
    meta.update(
        {
            "inference_seconds": time.time() - t0,
            "model_slug": MODEL_SLUG,
            "model_patch_embedding": "rgb_conv_mean_repeated_scaled_3_over_channels",
            "input_height": int(input_height),
            "model_input_shape": [int(resized_h), int(resized_w)],
        }
    )
    if model_revision:
        meta["model_revision"] = model_revision
    return save_depth_prediction(depth, out_dir, model_name, hdr_path, meta, save_visualization=save_vis)


def run_manifest(args: argparse.Namespace) -> None:
    rows_out = []
    model = None
    device = None
    active_channels = None
    for row in read_prediction_input_manifest(args.manifest):
        hsi_tensor, meta = load_hsi_tensor(row["hdr_path"], normalization=args.normalization, mask_invalid=not args.no_voxel_mask)
        num_channels = int(hsi_tensor.shape[0])
        if model is None or active_channels != num_channels:
            model, device = load_model(args.model_name, args.device, num_channels, args.model_revision)
            active_channels = num_channels
        out_dir = scene_out_dir(args.out_dir, MODEL_SLUG, row)
        pred = predict_hsi_tensor(
            model,
            device,
            hsi_tensor,
            meta,
            row["hdr_path"],
            out_dir,
            args.model_name,
            args.model_revision,
            input_height=args.input_height,
            save_vis=not args.no_vis,
        )
        rows_out.append(
            {
                **row,
                "model": MODEL_SLUG,
                "model_name": args.model_name,
                "model_revision": args.model_revision or "",
                "prediction_path": str(pred),
            }
        )
    write_prediction_manifest(Path(args.out_dir) / MODEL_SLUG / "prediction_manifest.csv", rows_out)


def main() -> None:
    args = parse_args()
    if args.hdr:
        hsi_tensor, meta = load_hsi_tensor(args.hdr, normalization=args.normalization, mask_invalid=not args.no_voxel_mask)
        model, device = load_model(args.model_name, args.device, int(hsi_tensor.shape[0]), args.model_revision)
        pred = predict_hsi_tensor(
            model,
            device,
            hsi_tensor,
            meta,
            args.hdr,
            Path(args.out_dir),
            args.model_name,
            args.model_revision,
            input_height=args.input_height,
            save_vis=not args.no_vis,
        )
        print(pred)
        return
    run_manifest(args)


if __name__ == "__main__":
    main()
