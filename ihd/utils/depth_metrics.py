from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


EPS = 1e-6


@dataclass(frozen=True)
class DepthEvalConfig:
    min_depth_m: float = 0.0
    max_depth_m: float | None = None
    apply_median_scale: bool = False
    compute_ssi: bool = True


def valid_depth_mask(
    prediction_m: np.ndarray,
    target_m: np.ndarray,
    target_mask: np.ndarray | None = None,
    min_depth_m: float = 0.0,
    max_depth_m: float | None = None,
) -> np.ndarray:
    pred = np.asarray(prediction_m)
    target = np.asarray(target_m)
    mask = np.isfinite(pred) & np.isfinite(target) & (pred > 0.0) & (target > min_depth_m)
    if target_mask is not None:
        mask &= np.asarray(target_mask).astype(bool)
    if max_depth_m is not None:
        mask &= target <= max_depth_m
    return mask


def median_scale_prediction(prediction_m: np.ndarray, target_m: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    pred_valid = prediction_m[mask]
    target_valid = target_m[mask]
    if pred_valid.size == 0:
        return prediction_m.copy(), float("nan")
    pred_median = float(np.median(pred_valid))
    target_median = float(np.median(target_valid))
    if not np.isfinite(pred_median) or abs(pred_median) < EPS:
        return prediction_m.copy(), float("nan")
    scale = target_median / pred_median
    return prediction_m * scale, float(scale)


def affine_align_prediction(prediction_m: np.ndarray, target_m: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    pred = prediction_m[mask].astype(np.float64)
    target = target_m[mask].astype(np.float64)
    if pred.size < 2:
        return prediction_m.copy(), float("nan"), float("nan")
    design = np.stack([pred, np.ones_like(pred)], axis=1)
    scale, shift = np.linalg.lstsq(design, target, rcond=None)[0]
    aligned = prediction_m.astype(np.float64) * float(scale) + float(shift)
    return aligned.astype(np.float32), float(scale), float(shift)


def depth_metrics_from_arrays(
    prediction_m: np.ndarray,
    target_m: np.ndarray,
    target_mask: np.ndarray | None = None,
    config: DepthEvalConfig | None = None,
) -> dict[str, float]:
    cfg = config or DepthEvalConfig()
    pred = np.asarray(prediction_m, dtype=np.float32)
    target = np.asarray(target_m, dtype=np.float32)
    if pred.shape != target.shape:
        raise ValueError(f"Prediction shape {pred.shape} does not match target shape {target.shape}.")

    mask = valid_depth_mask(pred, target, target_mask, cfg.min_depth_m, cfg.max_depth_m)
    n_valid = int(mask.sum())
    n_total = int(mask.size)
    out: dict[str, float] = {
        "valid_pixels": float(n_valid),
        "total_pixels": float(n_total),
        "valid_pixel_ratio": float(n_valid / n_total) if n_total else float("nan"),
    }
    if n_valid == 0:
        return out

    eval_pred = pred
    if cfg.apply_median_scale:
        eval_pred, median_scale = median_scale_prediction(pred, target, mask)
        out["median_scale"] = median_scale

    out.update(_standard_depth_metrics(eval_pred[mask], target[mask], prefix=""))

    if cfg.compute_ssi:
        ssi_pred, ssi_scale, ssi_shift = affine_align_prediction(pred, target, mask)
        out["ssi_scale"] = ssi_scale
        out["ssi_shift_m"] = ssi_shift
        ssi_mask = valid_depth_mask(ssi_pred, target, target_mask, cfg.min_depth_m, cfg.max_depth_m)
        if int(ssi_mask.sum()) > 0:
            out.update(_standard_depth_metrics(ssi_pred[ssi_mask], target[ssi_mask], prefix="ssi_"))
            out["ssi_valid_pixels"] = float(int(ssi_mask.sum()))
    return out


def _standard_depth_metrics(prediction: np.ndarray, target: np.ndarray, prefix: str) -> dict[str, float]:
    pred = np.maximum(prediction.astype(np.float64), EPS)
    gt = np.maximum(target.astype(np.float64), EPS)
    diff = pred - gt
    abs_diff = np.abs(diff)
    sq_diff = diff * diff
    log_diff = np.log(pred) - np.log(gt)
    ratio = np.maximum(pred / gt, gt / pred)
    inv_diff = (1.0 / pred) - (1.0 / gt)
    silog_core = float(np.mean(log_diff * log_diff) - np.mean(log_diff) ** 2)
    return {
        f"{prefix}abs_rel": float(np.mean(abs_diff / gt)),
        f"{prefix}sq_rel": float(np.mean(sq_diff / gt)),
        f"{prefix}rmse_m": float(np.sqrt(np.mean(sq_diff))),
        f"{prefix}rmse_log": float(np.sqrt(np.mean(log_diff * log_diff))),
        f"{prefix}silog": float(100.0 * np.sqrt(max(silog_core, 0.0))),
        f"{prefix}mae_m": float(np.mean(abs_diff)),
        f"{prefix}imae_1_per_km": float(1000.0 * np.mean(np.abs(inv_diff))),
        f"{prefix}irmse_1_per_km": float(1000.0 * np.sqrt(np.mean(inv_diff * inv_diff))),
        f"{prefix}delta1": float(np.mean(ratio < 1.25)),
        f"{prefix}delta2": float(np.mean(ratio < 1.25**2)),
        f"{prefix}delta3": float(np.mean(ratio < 1.25**3)),
    }


def summarize_metric_rows(rows: list[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row.keys()})
    summary: dict[str, float] = {"num_scenes": float(len(rows))}
    for key in keys:
        numeric_values: list[float] = []
        for row in rows:
            if key not in row:
                continue
            try:
                value = float(row[key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                numeric_values.append(value)
        values = np.asarray(numeric_values, dtype=np.float64)
        if values.size == 0:
            continue
        summary[f"mean_{key}"] = float(np.mean(values))
        summary[f"median_{key}"] = float(np.median(values))
    return summary
