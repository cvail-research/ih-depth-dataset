from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .depth_metrics import DepthEvalConfig, depth_metrics_from_arrays, summarize_metric_rows
from .depth_png import DEPTH_SCALE, load_depth_png
from .baseline_io import load_pseudobroadband_rgb


OUTPUT_DIR_NAMES = {"errors_out", "errors_img", "depth_gt", "depth_pred", "input_preview"}


def sanitize_relpath(relative_path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", str(relative_path))


def _png_map(root: Path) -> dict[Path, Path]:
    files: dict[Path, Path] = {}
    for path in sorted(root.rglob("*.png")):
        rel = path.relative_to(root)
        if rel.parts and rel.parts[0] in OUTPUT_DIR_NAMES:
            continue
        if path.name.endswith("_depth.png"):
            files[rel] = path
    return files


def pair_evaluation_files(gt_root: str | Path, prediction_root: str | Path) -> list[tuple[Path, Path, Path]]:
    gt_dir = Path(gt_root)
    pred_dir = Path(prediction_root)
    if not gt_dir.is_dir():
        raise ValueError(f"GT_DIR does not exist or is not a directory: {gt_dir}")
    if not pred_dir.is_dir():
        raise ValueError(f"PREDICTION_DIR does not exist or is not a directory: {pred_dir}")

    gt_files = _png_map(gt_dir)
    pred_files = _png_map(pred_dir)
    gt_keys = set(gt_files)
    pred_keys = set(pred_files)
    missing_pred = sorted(gt_keys - pred_keys)
    missing_gt = sorted(pred_keys - gt_keys)
    problems: list[str] = []
    if missing_pred:
        problems.extend([f"Missing prediction counterpart for {path}" for path in missing_pred])
    if missing_gt:
        problems.extend([f"Missing GT counterpart for {path}" for path in missing_gt])

    common_dirs = sorted({path.parent for path in gt_keys} | {path.parent for path in pred_keys})
    for rel_dir in common_dirs:
        gt_names = sorted(path.name for path in gt_keys if path.parent == rel_dir)
        pred_names = sorted(path.name for path in pred_keys if path.parent == rel_dir)
        if gt_names and pred_names and gt_names != pred_names:
            problems.append(
                f"Filename mismatch under {rel_dir}: gt={gt_names}, prediction={pred_names}. "
                "GT and predictions must use identical basenames."
            )
    if problems:
        raise ValueError("\n".join(problems))
    return [(rel, gt_files[rel], pred_files[rel]) for rel in sorted(gt_keys)]


def _render_depth(depth_m: np.ndarray, valid_mask: np.ndarray, out_path: Path, title: str) -> None:
    plot_data = np.where(valid_mask, depth_m, np.nan)
    finite = np.isfinite(plot_data)
    if np.any(finite):
        vmin = float(np.nanmin(plot_data[finite]))
        vmax = float(np.nanmax(plot_data[finite]))
        if vmax <= vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(plot_data, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Depth (m)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _render_error(gt_depth_m: np.ndarray, pred_depth_m: np.ndarray, valid_mask: np.ndarray, out_path: Path) -> None:
    error = np.where(valid_mask, np.abs(pred_depth_m - gt_depth_m), np.nan)
    vmax = float(np.nanmax(error)) if np.isfinite(error).any() else 1.0
    if vmax <= 0.0:
        vmax = 1.0
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(error, cmap="magma", vmin=0.0, vmax=vmax, aspect="auto")
    ax.set_title("Absolute Error (m)")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("|pred - gt| (m)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _write_input_preview(gt_png_path: Path, out_path: Path) -> bool:
    scene_dir = gt_png_path.parent
    png_candidates = sorted(
        path for path in scene_dir.glob("*.png") if path.name != gt_png_path.name and not path.name.endswith("_depth.png")
    )
    if png_candidates:
        out_path.write_bytes(png_candidates[0].read_bytes())
        return True
    hdr_candidates = sorted(scene_dir.glob("*LWHSI*.hdr"))
    if not hdr_candidates:
        return False
    try:
        rgb, _ = load_pseudobroadband_rgb(hdr_candidates[0])
    except Exception:
        return False
    plt.imsave(out_path, rgb)
    return True


def evaluate_scene_pair(
    relative_path: Path,
    gt_png_path: Path,
    pred_png_path: Path,
    prediction_root: str | Path,
    config: DepthEvalConfig | None = None,
) -> dict[str, float | str]:
    gt_depth_m, gt_valid = load_depth_png(gt_png_path)
    pred_depth_m, _pred_valid = load_depth_png(pred_png_path)
    if gt_depth_m.shape != pred_depth_m.shape:
        raise ValueError(
            f"Shape mismatch for {relative_path}: gt={gt_depth_m.shape}, prediction={pred_depth_m.shape}."
        )
    metrics = depth_metrics_from_arrays(pred_depth_m, gt_depth_m, gt_valid, config or DepthEvalConfig())
    metrics["scene"] = str(relative_path)
    metrics["gt_path"] = str(gt_png_path)
    metrics["prediction_path"] = str(pred_png_path)

    safe_name = sanitize_relpath(relative_path.with_suffix(""))
    out_root = Path(prediction_root)
    for dirname in OUTPUT_DIR_NAMES:
        (out_root / dirname).mkdir(parents=True, exist_ok=True)

    (out_root / "errors_out" / f"{safe_name}.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    _render_error(gt_depth_m, pred_depth_m, gt_valid, out_root / "errors_img" / f"{safe_name}.png")
    _render_depth(gt_depth_m, gt_valid, out_root / "depth_gt" / f"{safe_name}.png", "GT Depth (m)")
    _render_depth(pred_depth_m, pred_depth_m > 0.0, out_root / "depth_pred" / f"{safe_name}.png", "Prediction Depth (m)")
    _write_input_preview(gt_png_path, out_root / "input_preview" / f"{safe_name}.png")
    return metrics


def format_summary_text(rows: list[dict[str, float | str]], summary: dict[str, float]) -> str:
    lines = [
        "IH-Depth evaluation summary",
        f"num_scenes: {int(summary.get('num_scenes', 0.0))}",
        f"depth_png_scale: {DEPTH_SCALE:g}",
        "",
        "Aggregate metrics",
    ]
    preferred = [
        "mean_abs_rel",
        "mean_sq_rel",
        "mean_rmse_m",
        "mean_rmse_log",
        "mean_silog",
        "mean_mae_m",
        "mean_imae_1_per_km",
        "mean_irmse_1_per_km",
        "mean_delta1",
        "mean_delta2",
        "mean_delta3",
        "mean_valid_pixel_ratio",
    ]
    for key in preferred:
        if key in summary:
            lines.append(f"{key}: {summary[key]:.6f}")
    lines.extend(["", "Per-scene"])
    for row in rows:
        scene = str(row["scene"])
        rmse = float(row.get("rmse_m", float("nan")))
        abs_rel = float(row.get("abs_rel", float("nan")))
        valid_ratio = float(row.get("valid_pixel_ratio", float("nan")))
        lines.append(f"{scene}: rmse_m={rmse:.6f}, abs_rel={abs_rel:.6f}, valid_pixel_ratio={valid_ratio:.6f}")
    return "\n".join(lines) + "\n"


def write_stats_report(prediction_root: str | Path, output_name: str, rows: list[dict[str, float | str]]) -> tuple[Path, dict[str, float], str]:
    summary = summarize_metric_rows(rows)
    text = format_summary_text(rows, summary)
    report_path = Path(prediction_root) / output_name
    report_path.write_text(text)
    return report_path, summary, text
