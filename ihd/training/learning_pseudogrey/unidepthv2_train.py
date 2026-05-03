from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ihd.evaluation.model_io import load_pseudobroadband_rgb
from ihd.training.utils import (
    batch_metrics,
    init_wandb,
    load_depth_label,
    mean_metric,
    pad_depth_and_mask_items,
    read_manifest,
    save_prediction_preview,
    scene_label,
    seed_everything,
    silog_loss,
)


MODEL_SLUG = "unidepthv2"


@dataclass(frozen=True)
class TrainConfig:
    train_manifest: str
    val_manifest: str
    out_dir: str
    model_name: str
    device: str
    resolution_level: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    seed: int
    log_every: int
    eval_every_steps: int
    checkpoint_every_steps: int
    max_train_steps: int | None
    max_val_batches: int | None
    silog_lambda: float
    min_depth_m: float
    max_depth_m: float
    wandb_project: str | None
    wandb_entity: str | None
    wandb_run_name: str | None
    wandb_mode: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Fine-tune UniDepthV2 on IH pseudo-broadband LWHSI inputs and projected LiDAR depth labels."
    )
    ap.add_argument("--train-manifest", required=True, help="CSV with hdr_path,label_path columns.")
    ap.add_argument("--val-manifest", required=True, help="CSV with hdr_path,label_path columns.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-name", default="lpiccinelli/unidepth-v2-vitl14")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resolution-level", type=int, default=9)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every-steps", type=int, default=250)
    ap.add_argument("--checkpoint-every-steps", type=int, default=1000)
    ap.add_argument("--max-train-steps", type=int)
    ap.add_argument("--max-val-batches", type=int)
    ap.add_argument("--silog-lambda", type=float, default=0.85)
    ap.add_argument("--min-depth-m", type=float, default=1e-3)
    ap.add_argument("--max-depth-m", type=float, default=300.0)
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    return ap.parse_args()


class IHDepthDataset:
    def __init__(self, manifest: str | Path):
        self.rows = read_manifest(manifest)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import torch

        row = self.rows[idx]
        rgb, _ = load_pseudobroadband_rgb(row["hdr_path"])
        depth, mask = load_depth_label(row["label_path"])

        return {
            "rgb": torch.from_numpy(rgb).permute(2, 0, 1).float(),
            "depth_m": torch.from_numpy(depth),
            "valid_mask": torch.from_numpy(mask),
            "scene": scene_label(row),
            "hdr_path": row["hdr_path"],
            "label_path": row["label_path"],
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    depths, masks, max_h, max_w = pad_depth_and_mask_items(batch)
    images = []
    for item in batch:
        _, h, w = item["rgb"].shape
        pad = (0, max_w - w, 0, max_h - h)
        images.append(F.pad(item["rgb"], pad, value=0.0))

    return {
        "rgb": torch.stack(images, dim=0),
        "depth_m": depths,
        "valid_mask": masks,
        "scene": [b["scene"] for b in batch],
        "hdr_path": [b["hdr_path"] for b in batch],
        "label_path": [b["label_path"] for b in batch],
    }


def move_batch_to_device(batch: dict[str, Any], device) -> dict[str, Any]:
    return {
        **batch,
        "rgb": batch["rgb"].to(device, non_blocking=True),
        "depth_m": batch["depth_m"].to(device, non_blocking=True),
        "valid_mask": batch["valid_mask"].to(device, non_blocking=True),
    }


def predict_depth(model, rgb, *, normalize: bool = True):
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF
    import unidepth.models.unidepthv2.unidepthv2 as unidepth_module

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

    _, _, h, w = rgb.shape
    paddings, (padded_h, padded_w) = unidepth_module.get_paddings((h, w), ratio_bounds)
    pad_left, pad_right, pad_top, pad_bottom = paddings
    _, (new_h, new_w) = unidepth_module.get_resize_factor((padded_h, padded_w), pixels_bounds)

    image = rgb
    if normalize:
        image = TF.normalize(
            image.float() / 255.0,
            mean=unidepth_module.IMAGENET_DATASET_MEAN,
            std=unidepth_module.IMAGENET_DATASET_STD,
        )
    image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    image = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)
    _, model_outputs = model.encode_decode({"image": image, "camera": None}, image_metas=[])
    depth = unidepth_module._postprocess(
        model_outputs["points"][:, -1:],
        (padded_h, padded_w),
        paddings=paddings,
        interpolation_mode=model.interpolation_mode,
    )
    return depth.squeeze(1)


def save_checkpoint(out_dir: Path, model, optimizer, step: int, epoch: int, config: TrainConfig) -> Path:
    import torch

    ckpt_dir = out_dir / "checkpoints" / f"step_{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir / "model")
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
        },
        ckpt_dir / "training_state.pt",
    )
    return ckpt_dir


def evaluate(model, loader, device, config: TrainConfig, max_batches: int | None = None) -> dict[str, float]:
    import torch

    model.eval()
    losses: list[float] = []
    metrics: list[dict[str, float]] = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            pred = predict_depth(model, batch["rgb"])
            loss = silog_loss(
                pred,
                batch["depth_m"],
                batch["valid_mask"],
                min_depth_m=config.min_depth_m,
                max_depth_m=config.max_depth_m,
                lam=config.silog_lambda,
            )
            losses.append(float(loss.detach().cpu()))
            metrics.append(
                batch_metrics(
                    pred,
                    batch["depth_m"],
                    batch["valid_mask"],
                    min_depth_m=config.min_depth_m,
                    max_depth_m=config.max_depth_m,
                )
            )
    model.train()
    return {
        "val_silog_loss": float(np.mean(losses)) if losses else math.nan,
        "val_abs_rel": mean_metric(metrics, "abs_rel"),
        "val_rmse_m": mean_metric(metrics, "rmse_m"),
        "val_valid_pixels": mean_metric(metrics, "valid_pixels"),
    }


def main() -> None:
    args = parse_args()
    config = TrainConfig(**vars(args))
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n")
    seed_everything(config.seed)

    import torch
    import unidepth.models.backbones.metadinov2.attention as unidepth_attention
    import unidepth.models.backbones.metadinov2.block as unidepth_block
    from torch.utils.data import DataLoader
    from unidepth.models import UniDepthV2

    device = torch.device(config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu")
    # The workstation GPU can be newer than the installed xFormers kernels.
    # Force the DINOv2 backbone to use PyTorch attention for reproducibility.
    unidepth_attention.XFORMERS_AVAILABLE = False
    unidepth_block.XFORMERS_AVAILABLE = False
    model = UniDepthV2.from_pretrained(config.model_name).to(device)
    model.resolution_level = config.resolution_level
    model.interpolation_mode = "bilinear"
    model.train()

    train_ds = IHDepthDataset(config.train_manifest)
    val_ds = IHDepthDataset(config.val_manifest)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    run = init_wandb(config)

    print(f"Training {config.model_name} on {len(train_ds)} scenes; validating on {len(val_ds)} scenes.")
    print(f"Device: {device}; batch_size={config.batch_size}; epochs={config.epochs}")

    step = 0
    t0 = time.time()
    last_pred = None
    for epoch in range(config.epochs):
        for batch in train_loader:
            step += 1
            batch = move_batch_to_device(batch, device)
            pred = predict_depth(model, batch["rgb"])
            last_pred = (pred, batch["depth_m"], batch["valid_mask"])
            loss = silog_loss(
                pred,
                batch["depth_m"],
                batch["valid_mask"],
                min_depth_m=config.min_depth_m,
                max_depth_m=config.max_depth_m,
                lam=config.silog_lambda,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step % config.log_every == 0:
                metrics = batch_metrics(
                    pred,
                    batch["depth_m"],
                    batch["valid_mask"],
                    min_depth_m=config.min_depth_m,
                    max_depth_m=config.max_depth_m,
                )
                log = {
                    "train_silog_loss": float(loss.detach().cpu()),
                    "train_abs_rel": metrics["abs_rel"],
                    "train_rmse_m": metrics["rmse_m"],
                    "epoch": epoch,
                    "step": step,
                    "elapsed_minutes": (time.time() - t0) / 60.0,
                }
                print(json.dumps(log, sort_keys=True))
                if run:
                    run.log(log, step=step)

            if step % config.eval_every_steps == 0:
                val_log = evaluate(model, val_loader, device, config, config.max_val_batches)
                val_log.update({"epoch": epoch, "step": step})
                print(json.dumps(val_log, sort_keys=True))
                if run:
                    run.log(val_log, step=step)
                if last_pred:
                    save_prediction_preview(out_dir, step, *last_pred)

            if step % config.checkpoint_every_steps == 0:
                ckpt_dir = save_checkpoint(out_dir, model, optimizer, step, epoch, config)
                print(f"Saved checkpoint: {ckpt_dir}")

            if config.max_train_steps is not None and step >= config.max_train_steps:
                break
        if config.max_train_steps is not None and step >= config.max_train_steps:
            break

    final_metrics = evaluate(model, val_loader, device, config, config.max_val_batches)
    final_metrics.update({"step": step, "elapsed_minutes": (time.time() - t0) / 60.0})
    (out_dir / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2, sort_keys=True) + "\n")
    if last_pred:
        save_prediction_preview(out_dir, step, *last_pred)
    ckpt_dir = save_checkpoint(out_dir, model, optimizer, step, config.epochs - 1, config)
    print(f"Saved final checkpoint: {ckpt_dir}")
    print(json.dumps(final_metrics, sort_keys=True))
    if run:
        run.log(final_metrics, step=step)
        run.finish()


if __name__ == "__main__":
    main()
