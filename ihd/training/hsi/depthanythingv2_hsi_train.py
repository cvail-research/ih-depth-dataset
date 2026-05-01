from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ihd.hsi.depthanythingv2_hsi import (
    adapt_depthanythingv2_patch_embedding,
    dino_compatible_size,
    load_hsi_tensor,
)
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


MODEL_SLUG = "depthanythingv2_hsi_patch"


@dataclass(frozen=True)
class TrainConfig:
    train_manifest: str
    val_manifest: str
    out_dir: str
    model_name: str
    device: str
    input_height: int
    normalization: str
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
        description=(
            "Fine-tune Depth Anything V2 with a full-LWHSI patch embedding and "
            "projected LiDAR depth labels."
        )
    )
    ap.add_argument("--train-manifest", required=True, help="CSV with hdr_path,label_path columns.")
    ap.add_argument("--val-manifest", required=True, help="CSV with hdr_path,label_path columns.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-name", default="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--input-height", type=int, default=518)
    ap.add_argument("--normalization", default="per-band-standardize", choices=["per-band-standardize", "per-band-minmax"])
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


class IHDepthHSIDataset:
    def __init__(self, manifest: str | Path, *, normalization: str):
        self.rows = read_manifest(manifest)
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import torch

        row = self.rows[idx]
        hsi, hsi_meta = load_hsi_tensor(row["hdr_path"], normalization=self.normalization)
        depth, mask = load_depth_label(row["label_path"])
        return {
            "hsi": hsi,
            "depth_m": torch.from_numpy(depth),
            "valid_mask": torch.from_numpy(mask),
            "scene": scene_label(row),
            "hdr_path": row["hdr_path"],
            "label_path": row["label_path"],
            "num_hsi_channels": hsi_meta["num_hsi_channels"],
        }


def infer_num_channels(manifest: str | Path, *, normalization: str) -> int:
    row = read_manifest(manifest)[0]
    hsi, _ = load_hsi_tensor(row["hdr_path"], normalization=normalization)
    return int(hsi.shape[0])


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    channel_counts = {int(b["hsi"].shape[0]) for b in batch}
    if len(channel_counts) != 1:
        raise ValueError(f"Cannot batch scenes with different HSI channel counts: {sorted(channel_counts)}")

    depths, masks, max_h, max_w = pad_depth_and_mask_items(batch)
    images = []
    for item in batch:
        _, h, w = item["hsi"].shape
        pad = (0, max_w - w, 0, max_h - h)
        images.append(F.pad(item["hsi"], pad, value=0.0))

    return {
        "pixel_values": torch.stack(images, dim=0),
        "depth_m": depths,
        "valid_mask": masks,
        "scene": [b["scene"] for b in batch],
        "hdr_path": [b["hdr_path"] for b in batch],
        "label_path": [b["label_path"] for b in batch],
        "num_hsi_channels": sorted(channel_counts)[0],
    }


def move_batch_to_device(batch: dict[str, Any], device) -> dict[str, Any]:
    return {
        **batch,
        "pixel_values": batch["pixel_values"].to(device, non_blocking=True),
        "depth_m": batch["depth_m"].to(device, non_blocking=True),
        "valid_mask": batch["valid_mask"].to(device, non_blocking=True),
    }


def predict_depth(model, pixel_values, target_hw: tuple[int, int], *, input_height: int):
    import torch.nn.functional as F

    _, _, padded_h, padded_w = pixel_values.shape
    projection = model.backbone.embeddings.patch_embeddings.projection
    if int(pixel_values.shape[1]) != int(projection.in_channels):
        raise ValueError(
            f"Model patch embedding expects {projection.in_channels} channels but batch has {pixel_values.shape[1]}."
        )
    patch_size = int(projection.kernel_size[0])
    resized_h, resized_w = dino_compatible_size(padded_h, padded_w, input_height, patch_size)
    model_input = F.interpolate(pixel_values, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
    outputs = model(pixel_values=model_input)
    pred = outputs.predicted_depth
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)
    pred = F.interpolate(pred, size=target_hw, mode="bilinear", align_corners=False)
    return pred.squeeze(1)


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
            target_hw = tuple(batch["depth_m"].shape[-2:])
            pred = predict_depth(model, batch["pixel_values"], target_hw, input_height=config.input_height)
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
    from torch.utils.data import DataLoader
    from transformers import AutoModelForDepthEstimation

    device = torch.device(config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu")
    num_channels = infer_num_channels(config.train_manifest, normalization=config.normalization)
    model = AutoModelForDepthEstimation.from_pretrained(config.model_name)
    adapt_depthanythingv2_patch_embedding(model, num_channels)
    model = model.to(device)
    model.train()

    train_ds = IHDepthHSIDataset(config.train_manifest, normalization=config.normalization)
    val_ds = IHDepthHSIDataset(config.val_manifest, normalization=config.normalization)
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

    print(
        f"Training {config.model_name} with {num_channels} HSI channels on {len(train_ds)} scenes; "
        f"validating on {len(val_ds)} scenes."
    )
    print(f"Device: {device}; batch_size={config.batch_size}; epochs={config.epochs}")

    step = 0
    t0 = time.time()
    last_pred = None
    for epoch in range(config.epochs):
        for batch in train_loader:
            if int(batch["num_hsi_channels"]) != num_channels:
                raise ValueError(
                    f"Model was initialized for {num_channels} channels but batch has {batch['num_hsi_channels']}."
                )
            step += 1
            batch = move_batch_to_device(batch, device)
            target_hw = tuple(batch["depth_m"].shape[-2:])
            pred = predict_depth(model, batch["pixel_values"], target_hw, input_height=config.input_height)
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
