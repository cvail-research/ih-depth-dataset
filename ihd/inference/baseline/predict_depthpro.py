from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ihd.evaluation.model_io import (
    load_pseudobroadband_rgb,
    read_prediction_input_manifest,
    save_depth_prediction,
    scene_out_dir,
    write_prediction_manifest,
)


MODEL_SLUG = "depthpro"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Depth Pro on IH pseudo-broadband LWHSI inputs.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hdr", help="Single ENVI .hdr path.")
    src.add_argument("--manifest", help="CSV with hdr_path,label_path columns.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint-path", default="checkpoints/depth_pro.pt")
    ap.add_argument("--no-vis", action="store_true")
    return ap.parse_args()


def ensure_checkpoint(checkpoint_path: str) -> Path:
    path = Path(checkpoint_path)
    if path.exists():
        return path
    from huggingface_hub import hf_hub_download

    path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id="apple/DepthPro",
        filename="depth_pro.pt",
        local_dir=str(path.parent),
        local_dir_use_symlinks=False,
    )
    downloaded_path = Path(downloaded)
    if downloaded_path != path and not path.exists():
        path.write_bytes(downloaded_path.read_bytes())
    return path


def load_model(device: str, checkpoint_path: str):
    import torch
    import depth_pro
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

    actual_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint = ensure_checkpoint(checkpoint_path)
    config = DEFAULT_MONODEPTH_CONFIG_DICT
    config.checkpoint_uri = str(checkpoint)
    model, transform = depth_pro.create_model_and_transforms(config=config, device=actual_device)
    model = model.eval()
    return model, transform, actual_device


def predict_one(model, transform, device, hdr_path: str, out_dir: Path, save_vis: bool) -> Path:
    import torch

    rgb, meta = load_pseudobroadband_rgb(hdr_path)
    image = Image.fromarray(rgb)
    image_t = transform(image).to(device)
    t0 = time.time()
    with torch.no_grad():
        prediction = model.infer(image_t, f_px=None)
    depth = prediction["depth"].detach().cpu().numpy().squeeze().astype(np.float32)
    if "focallength_px" in prediction:
        meta["predicted_focallength_px"] = float(prediction["focallength_px"])
    meta.update({"inference_seconds": time.time() - t0, "model_slug": MODEL_SLUG})
    return save_depth_prediction(depth, out_dir, "apple/ml-depth-pro", hdr_path, meta, save_visualization=save_vis)


def main() -> None:
    args = parse_args()
    model, transform, device = load_model(args.device, args.checkpoint_path)
    rows_out = []
    if args.hdr:
        pred = predict_one(model, transform, device, args.hdr, Path(args.out_dir), not args.no_vis)
        print(pred)
        return

    for row in read_prediction_input_manifest(args.manifest):
        out_dir = scene_out_dir(args.out_dir, MODEL_SLUG, row)
        pred = predict_one(model, transform, device, row["hdr_path"], out_dir, not args.no_vis)
        rows_out.append({**row, "model": MODEL_SLUG, "model_name": "apple/ml-depth-pro", "prediction_path": str(pred)})
    write_prediction_manifest(Path(args.out_dir) / MODEL_SLUG / "prediction_manifest.csv", rows_out)


if __name__ == "__main__":
    main()
