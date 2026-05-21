from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from ihd.utils.baseline_io import (
    read_prediction_input_manifest,
    save_depth_prediction,
    scene_out_dir,
    write_prediction_manifest,
)
from baselines.depthanythingv2_hsi import load_hsi_tensor


MODEL_SLUG = "unik3d_hsi_patch"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run UniK3D with a hyperspectral patch embedding. "
            "The RGB DINO patch projection is replaced with a B-channel projection."
        )
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hdr", help="Single ENVI .hdr path.")
    src.add_argument("--manifest", help="CSV with hdr_path and optional label_path columns.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-name", default="lpiccinelli/unik3d-vitl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resolution-level", type=int, default=9)
    ap.add_argument("--normalization", default="per-band-standardize", choices=["per-band-standardize", "per-band-minmax"])
    ap.add_argument("--no-vis", action="store_true")
    return ap.parse_args()


def disable_unik3d_xformers(device_type: str) -> None:
    import unik3d.models.metadinov2.attention as unik3d_attention
    import unik3d.models.metadinov2.block as unik3d_block
    import unik3d.models.unik3d as unik3d_module

    unik3d_attention.XFORMERS_AVAILABLE = False
    unik3d_block.XFORMERS_AVAILABLE = False
    if device_type == "cpu":
        unik3d_module.DEVICE = "cpu"
        unik3d_module.ENABLED = False


def adapt_unik3d_patch_embedding(model, num_channels: int) -> None:
    import torch
    import torch.nn as nn

    patch_embed = model.pixel_encoder.patch_embed
    old_projection = patch_embed.proj
    if old_projection.in_channels == num_channels:
        patch_embed.in_chans = num_channels
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
    patch_embed.proj = new_projection
    patch_embed.in_chans = num_channels


def load_model(model_name: str, device: str, resolution_level: int, num_channels: int):
    import torch
    from unik3d.models import UniK3D

    actual_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    disable_unik3d_xformers(actual_device.type)
    model = UniK3D.from_pretrained(model_name)
    adapt_unik3d_patch_embedding(model, num_channels)
    model.resolution_level = resolution_level
    model.interpolation_mode = "bilinear"
    model = model.to(actual_device).eval()
    return model, actual_device


def predict_hsi_tensor(model, device, hsi_tensor, meta: dict[str, Any], hdr_path: str, out_dir: Path, model_name: str, save_vis: bool) -> Path:
    import torch
    import torch.nn.functional as F
    import unik3d.models.unik3d as unik3d_module

    ratio_bounds = model.shape_constraints["ratio_bounds"]
    pixels_bounds = [
        model.shape_constraints["pixels_min"],
        model.shape_constraints["pixels_max"],
    ]
    if hasattr(model, "resolution_level"):
        pixels_range = pixels_bounds[1] - pixels_bounds[0]
        interval = pixels_range / 10
        pixels_bounds = (
            model.resolution_level * interval + pixels_bounds[0],
            (model.resolution_level + 1) * interval + pixels_bounds[0],
        )

    _, h, w = hsi_tensor.shape
    paddings, (padded_h, padded_w) = unik3d_module.get_paddings((h, w), ratio_bounds)
    pad_left, pad_right, pad_top, pad_bottom = paddings
    _, (new_h, new_w) = unik3d_module.get_resize_factor((padded_h, padded_w), pixels_bounds)
    pixel_values = F.pad(hsi_tensor.unsqueeze(0), (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    pixel_values = F.interpolate(pixel_values, size=(new_h, new_w), mode="bilinear", align_corners=False).to(device)

    t0 = time.time()
    with torch.no_grad():
        _, model_outputs = model.encode_decode({"image": pixel_values}, image_metas={})
        depth = model_outputs["points"][:, -1:]
        depth = unik3d_module._postprocess(
            depth,
            (padded_h, padded_w),
            paddings=paddings,
            interpolation_mode=model.interpolation_mode,
        )
    depth_np = depth.squeeze().detach().cpu().numpy().astype(np.float32)
    meta.update(
        {
            "inference_seconds": time.time() - t0,
            "model_slug": MODEL_SLUG,
            "model_patch_embedding": "rgb_conv_mean_repeated_scaled_3_over_channels",
            "model_input_shape": [int(new_h), int(new_w)],
        }
    )
    return save_depth_prediction(depth_np, out_dir, model_name, hdr_path, meta, save_visualization=save_vis)


def predict_one(model, device, hdr_path: str, out_dir: Path, model_name: str, normalization: str, save_vis: bool) -> Path:
    hsi_tensor, meta = load_hsi_tensor(hdr_path, normalization=normalization)
    return predict_hsi_tensor(model, device, hsi_tensor, meta, hdr_path, out_dir, model_name, save_vis)


def run_manifest(args: argparse.Namespace) -> None:
    rows_out = []
    model = None
    device = None
    active_channels = None
    for row in read_prediction_input_manifest(args.manifest):
        hsi_tensor, meta = load_hsi_tensor(row["hdr_path"], normalization=args.normalization)
        num_channels = int(hsi_tensor.shape[0])
        if model is None or active_channels != num_channels:
            model, device = load_model(args.model_name, args.device, args.resolution_level, num_channels)
            active_channels = num_channels
        out_dir = scene_out_dir(args.out_dir, MODEL_SLUG, row)
        pred = predict_hsi_tensor(model, device, hsi_tensor, meta, row["hdr_path"], out_dir, args.model_name, not args.no_vis)
        rows_out.append({**row, "model": MODEL_SLUG, "model_name": args.model_name, "prediction_path": str(pred)})
    write_prediction_manifest(Path(args.out_dir) / MODEL_SLUG / "prediction_manifest.csv", rows_out)


def main() -> None:
    args = parse_args()
    if args.hdr:
        hsi_tensor, meta = load_hsi_tensor(args.hdr, normalization=args.normalization)
        model, device = load_model(args.model_name, args.device, args.resolution_level, int(hsi_tensor.shape[0]))
        pred = predict_hsi_tensor(model, device, hsi_tensor, meta, args.hdr, Path(args.out_dir), args.model_name, not args.no_vis)
        print(pred)
        return
    run_manifest(args)


if __name__ == "__main__":
    main()
