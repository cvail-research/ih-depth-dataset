from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ihd.evaluation.model_io import (
    load_pseudobroadband_rgb,
    read_prediction_input_manifest,
    save_depth_prediction,
    scene_out_dir,
    write_prediction_manifest,
)


MODEL_SLUG = "unik3d"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run UniK3D on IH pseudo-broadband LWHSI inputs.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hdr", help="Single ENVI .hdr path.")
    src.add_argument("--manifest", help="CSV with hdr_path,label_path columns.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-name", default="lpiccinelli/unik3d-vitl")
    ap.add_argument("--device", default="cpu", help="Use cpu for numerical stability unless verified otherwise.")
    ap.add_argument("--resolution-level", type=int, default=9)
    ap.add_argument("--no-vis", action="store_true")
    return ap.parse_args()


def load_model(model_name: str, device: str, resolution_level: int):
    import torch
    import unik3d.models.unik3d as unik3d_module
    from unik3d.models import UniK3D

    if device == "cpu":
        unik3d_module.DEVICE = "cpu"
        unik3d_module.ENABLED = False
    model = UniK3D.from_pretrained(model_name)
    model.resolution_level = resolution_level
    model.interpolation_mode = "bilinear"
    model = model.to(torch.device(device)).eval()
    return model


def predict_one(model, hdr_path: str, out_dir: Path, model_name: str, device: str, save_vis: bool) -> Path:
    import torch

    rgb, meta = load_pseudobroadband_rgb(hdr_path)
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1).float().to(torch.device(device))
    t0 = time.time()
    with torch.no_grad():
        outputs = model.infer(rgb=rgb_torch, camera=None, normalize=True, rays=None)
    depth = outputs["depth"].detach().cpu().numpy().squeeze().astype(np.float32)
    meta.update({"inference_seconds": time.time() - t0, "model_slug": MODEL_SLUG})
    return save_depth_prediction(depth, out_dir, model_name, hdr_path, meta, save_visualization=save_vis)


def main() -> None:
    args = parse_args()
    model = load_model(args.model_name, args.device, args.resolution_level)
    rows_out = []
    if args.hdr:
        pred = predict_one(model, args.hdr, Path(args.out_dir), args.model_name, args.device, not args.no_vis)
        print(pred)
        return

    for row in read_prediction_input_manifest(args.manifest):
        out_dir = scene_out_dir(args.out_dir, MODEL_SLUG, row)
        pred = predict_one(model, row["hdr_path"], out_dir, args.model_name, args.device, not args.no_vis)
        rows_out.append({**row, "model": MODEL_SLUG, "model_name": args.model_name, "prediction_path": str(pred)})
    write_prediction_manifest(Path(args.out_dir) / MODEL_SLUG / "prediction_manifest.csv", rows_out)


if __name__ == "__main__":
    main()

