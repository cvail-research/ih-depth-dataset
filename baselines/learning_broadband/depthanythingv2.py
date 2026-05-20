from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ihd.utils.baseline_io import (
    build_prediction_input_rows_from_scene_manifest,
    load_ground_truth_depth,
    load_pseudobroadband_rgb,
    read_prediction_input_manifest,
    save_depth_prediction,
    save_input_prediction_groundtruth_figures,
    scene_out_dir,
    write_prediction_manifest,
)


MODEL_SLUG = "depthanythingv2"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Depth Anything V2 on IH broadband LWHSI inputs.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hdr", help="Single ENVI .hdr path.")
    src.add_argument("--manifest", help="CSV with hdr_path and optional label_path columns.")
    src.add_argument("--scene-manifest", help="Scene manifest with collection/path/step columns.")
    ap.add_argument("--label-path", help="Ground-truth IH-Depth uint16 PNG path when using --hdr.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-name", default="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf")
    ap.add_argument(
        "--model-revision",
        help="Optional Hugging Face Hub revision/commit SHA passed to from_pretrained for reproducible weights.",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--depth-label-root", default="analysis/depth_labels/platform_sphere_r4p0")
    ap.add_argument("--disk-root", default="/disk")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-vis", action="store_true")
    return ap.parse_args()


def load_model(model_name: str, device: str, model_revision: str | None = None):
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    actual_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_name, revision=model_revision)
    model = AutoModelForDepthEstimation.from_pretrained(model_name, revision=model_revision).to(actual_device).eval()
    return processor, model, actual_device


def predict_one(
    processor,
    model,
    device,
    hdr_path: str,
    out_dir: Path,
    model_name: str,
    model_revision: str | None,
    save_vis: bool,
    label_path: str | None = None,
) -> Path:
    import torch

    rgb, meta = load_pseudobroadband_rgb(hdr_path)
    image = Image.fromarray(rgb)
    inputs = processor(images=image, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    post = processor.post_process_depth_estimation(outputs, target_sizes=[(image.height, image.width)])
    depth = post[0]["predicted_depth"].detach().cpu().numpy().astype(np.float32)
    meta.update({"inference_seconds": time.time() - t0, "model_slug": MODEL_SLUG})
    if model_revision:
        meta["model_revision"] = model_revision
    pred_path = save_depth_prediction(depth, out_dir, model_name, hdr_path, meta, save_visualization=save_vis)
    gt_depth = None
    gt_mask = None
    if label_path and Path(label_path).exists():
        gt_depth, gt_mask = load_ground_truth_depth(label_path)
    input_gray_u8 = np.mean(rgb.astype(np.float32), axis=2).clip(0, 255).astype(np.uint8)
    save_input_prediction_groundtruth_figures(
        input_gray_u8=input_gray_u8,
        prediction_m=depth,
        out_dir=out_dir,
        ground_truth_m=gt_depth,
        ground_truth_mask=gt_mask,
    )
    return pred_path


def main() -> None:
    args = parse_args()
    processor, model, device = load_model(args.model_name, args.device, args.model_revision)
    rows_out = []
    if args.hdr:
        pred = predict_one(
            processor,
            model,
            device,
            args.hdr,
            Path(args.out_dir),
            args.model_name,
            args.model_revision,
            not args.no_vis,
            label_path=args.label_path,
        )
        print(pred)
        return

    if args.scene_manifest:
        rows_in = build_prediction_input_rows_from_scene_manifest(
            args.scene_manifest,
            depth_label_root=args.depth_label_root,
            disk_root=args.disk_root,
            limit=args.limit,
        )
    else:
        rows_in = read_prediction_input_manifest(args.manifest)

    for row in rows_in:
        out_dir = scene_out_dir(args.out_dir, MODEL_SLUG, row)
        pred = predict_one(
            processor,
            model,
            device,
            row["hdr_path"],
            out_dir,
            args.model_name,
            args.model_revision,
            not args.no_vis,
            label_path=row.get("label_path"),
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


if __name__ == "__main__":
    main()
