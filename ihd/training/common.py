from __future__ import annotations

import csv
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ihd.evaluation.model_io import save_depth_visualization


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    required = {"hdr_path", "label_path"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Manifest {path} is missing columns: {sorted(missing)}")
    return rows


def scene_label(row: dict[str, str]) -> str:
    return row.get("scene") or f"{row.get('collection', '')} / {row.get('path', '')} / {row.get('step', '')}"


def load_depth_label(label_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    label_npz = np.load(label_path)
    depth = np.asarray(label_npz["depth_m"], dtype=np.float32)
    if "valid_mask" in label_npz:
        mask = np.asarray(label_npz["valid_mask"], dtype=bool)
    else:
        mask = np.isfinite(depth) & (depth > 0.0)
    mask = mask & np.isfinite(depth) & (depth > 0.0)
    return depth, mask


def pad_depth_and_mask_items(batch: list[dict[str, Any]]) -> tuple[Any, Any, int, int]:
    import torch
    import torch.nn.functional as F

    max_h = max(b["depth_m"].shape[0] for b in batch)
    max_w = max(b["depth_m"].shape[1] for b in batch)
    depths = []
    masks = []
    for item in batch:
        h, w = item["depth_m"].shape
        pad = (0, max_w - w, 0, max_h - h)
        depths.append(F.pad(item["depth_m"], pad, value=0.0))
        masks.append(F.pad(item["valid_mask"], pad, value=False))
    return torch.stack(depths, dim=0), torch.stack(masks, dim=0), max_h, max_w


def silog_loss(
    pred_m,
    target_m,
    valid_mask,
    *,
    min_depth_m: float,
    max_depth_m: float,
    lam: float,
):
    import torch

    pred = torch.clamp(pred_m, min=min_depth_m, max=max_depth_m)
    target = torch.clamp(target_m, min=min_depth_m, max=max_depth_m)
    mask = valid_mask & torch.isfinite(pred) & torch.isfinite(target) & (target > min_depth_m)
    if not torch.any(mask):
        return pred.sum() * 0.0
    diff = torch.log(pred[mask]) - torch.log(target[mask])
    return torch.sqrt(torch.clamp(torch.mean(diff * diff) - lam * torch.mean(diff) ** 2, min=0.0))


def batch_metrics(pred_m, target_m, valid_mask, *, min_depth_m: float, max_depth_m: float) -> dict[str, float]:
    import torch

    pred = torch.clamp(pred_m.detach(), min=min_depth_m, max=max_depth_m)
    target = torch.clamp(target_m.detach(), min=min_depth_m, max=max_depth_m)
    mask = valid_mask & torch.isfinite(pred) & torch.isfinite(target) & (target > min_depth_m)
    if not torch.any(mask):
        return {"abs_rel": math.nan, "rmse_m": math.nan, "valid_pixels": 0.0}
    p = pred[mask]
    t = target[mask]
    return {
        "abs_rel": float(torch.mean(torch.abs(p - t) / t).cpu()),
        "rmse_m": float(torch.sqrt(torch.mean((p - t) ** 2)).cpu()),
        "valid_pixels": float(mask.sum().cpu()),
    }


def mean_metric(rows: list[dict[str, float]], key: str) -> float:
    vals = [row[key] for row in rows if math.isfinite(row[key])]
    return float(np.mean(vals)) if vals else math.nan


def init_wandb(config: Any):
    if not config.wandb_project or config.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        print("wandb is not installed; continuing without wandb.")
        return None
    return wandb.init(
        entity=config.wandb_entity,
        project=config.wandb_project,
        name=config.wandb_run_name,
        mode=config.wandb_mode,
        config=asdict(config),
    )


def save_prediction_preview(out_dir: Path, step: int, pred_m, target_m, valid_mask) -> None:
    preview_dir = out_dir / "previews" / f"step_{step:07d}"
    preview_dir.mkdir(parents=True, exist_ok=True)
    pred = pred_m[0].detach().cpu().numpy()
    target = target_m[0].detach().cpu().numpy()
    mask = valid_mask[0].detach().cpu().numpy().astype(bool)
    save_depth_visualization(pred, preview_dir / "prediction.png")
    save_depth_visualization(np.where(mask, target, np.nan), preview_dir / "target.png")
