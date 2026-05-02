from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ihd.evaluation.depth_metrics import DepthEvalConfig, depth_metrics_from_arrays


MODEL_ORDER = [
    ("depthpro", "DepthPro"),
    ("unidepthv2", "UniDepthV2"),
    ("depthanythingv2", "DepthAnythingV2"),
    ("unik3d", "UniK3D"),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot GT + four model predictions for one scene.")
    ap.add_argument("--scene", required=True, help="Scene label, e.g. 'IHTest_202009_DistStA / path3 / path3_step6'.")
    ap.add_argument(
        "--prediction-root",
        action="append",
        required=True,
        help="Model prediction root containing prediction_manifest.csv and metrics_per_scene.csv. Repeat 4 times.",
    )
    ap.add_argument(
        "--output-dir",
        default="analysis/evaluation/scene_model_comparison",
        help="Output directory for plot and summary.",
    )
    ap.add_argument(
        "--label-key",
        default="depth_m",
        help="Ground-truth depth key in the .npz label file.",
    )
    ap.add_argument(
        "--mask-key",
        default="valid_mask",
        help="Valid-mask key in the .npz label file.",
    )
    ap.add_argument(
        "--prediction-key",
        default="depth_m",
        help="Prediction key in the .npz prediction file.",
    )
    ap.add_argument(
        "--min-depth-m",
        type=float,
        default=0.0,
        help="Minimum valid target depth for metrics.",
    )
    ap.add_argument(
        "--max-depth-m",
        type=float,
        default=None,
        help="Maximum valid target depth for metrics.",
    )
    return ap.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def find_scene_row(prediction_root: Path, scene: str) -> dict[str, str]:
    manifest_path = prediction_root / "prediction_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    rows = read_csv_rows(manifest_path)
    for row in rows:
        if row.get("scene") == scene:
            return row
    raise KeyError(f"Scene '{scene}' not found in {manifest_path}")


def load_npz_array(path: Path, key: str) -> np.ndarray:
    with np.load(path) as data:
        if key not in data:
            raise KeyError(f"{path} missing key '{key}'. Available: {list(data.files)}")
        return np.asarray(data[key], dtype=np.float32)


def load_mask(path: Path, key: str) -> np.ndarray | None:
    with np.load(path) as data:
        if key not in data:
            return None
        return np.asarray(data[key]).astype(bool)


def color_limits(arrays: list[np.ndarray]) -> tuple[float, float]:
    finite = [a[np.isfinite(a)] for a in arrays if np.isfinite(a).any()]
    if not finite:
        return 0.0, 1.0
    values = np.concatenate(finite)
    lo = float(np.percentile(values, 2))
    hi = float(np.percentile(values, 98))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def finite_limits(arr: np.ndarray) -> tuple[float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def load_scene_comparison(
    scene: str,
    prediction_roots: list[Path],
    label_key: str,
    mask_key: str,
    prediction_key: str,
    min_depth_m: float,
    max_depth_m: float | None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, np.ndarray], dict[str, dict[str, Any]], Path]:
    rows = {}
    for root in prediction_roots:
        row = find_scene_row(root, scene)
        rows[row["model"]] = row

    missing_models = [slug for slug, _ in MODEL_ORDER if slug not in rows]
    if missing_models:
        raise SystemExit(f"Missing model outputs for: {missing_models}")

    gt_path = Path(next(iter(rows.values()))["label_path"])
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)

    gt = load_npz_array(gt_path, label_key)
    gt_mask = load_mask(gt_path, mask_key)
    preds: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, Any]] = {}

    config = DepthEvalConfig(min_depth_m=min_depth_m, max_depth_m=max_depth_m)
    for model_slug, model_title in MODEL_ORDER:
        row = rows[model_slug]
        pred_path = Path(row["prediction_path"])
        pred = load_npz_array(pred_path, prediction_key)
        preds[model_title] = pred
        metrics[model_title] = depth_metrics_from_arrays(pred, gt, gt_mask, config)

    return gt, gt_mask, preds, metrics, gt_path


def render_scene_comparison(
    scene: str,
    gt: np.ndarray,
    preds: dict[str, np.ndarray],
    metrics: dict[str, dict[str, Any]],
    output_dir: Path,
) -> tuple[Path, Path]:
    vmin, vmax = finite_limits(gt)
    fig, axes = plt.subplots(5, 1, figsize=(14, 18))
    fig.subplots_adjust(hspace=0.12)

    def render(row_idx: int, arr: np.ndarray, title: str, subtitle: str | None = None):
        ax = axes[row_idx]
        im = ax.imshow(arr, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
        # Match the axes box to the rendered image aspect so the colorbar height
        # tracks the actual prediction panel instead of the whole subplot cell.
        ax.set_box_aspect(arr.shape[0] / arr.shape[1])
        ax.set_title(title + (f" | {subtitle}" if subtitle else ""), fontsize=13, pad=3)
        ax.axis("off")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.08)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Depth (m)")
        return im

    render(
        0,
        gt,
        "Ground Truth Depth",
        f"{scene}",
    )

    for idx, (_, model_title) in enumerate(MODEL_ORDER, start=1):
        m = metrics[model_title]
        render(
            idx,
            preds[model_title],
            model_title,
            f"abs_rel={m.get('abs_rel', float('nan')):.3f}  rmse_m={m.get('rmse_m', float('nan')):.3f}",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_scene = scene.replace(" / ", "__").replace("/", "_")
    plot_path = output_dir / f"{safe_scene}_model_comparison.png"
    summary_path = output_dir / f"{safe_scene}_model_comparison_metrics.json"

    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "scene": scene,
        "models": metrics,
        "score_mean_abs_rel": float(np.mean([metrics[title].get("abs_rel", np.nan) for _, title in MODEL_ORDER])),
        "score_std_abs_rel": float(np.std([metrics[title].get("abs_rel", np.nan) for _, title in MODEL_ORDER])),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return plot_path, summary_path


def main() -> None:
    args = parse_args()
    prediction_roots = [Path(path) for path in args.prediction_root]
    if len(prediction_roots) != 4:
        raise SystemExit("Provide exactly four --prediction-root values, one per model.")

    gt, _, preds, metrics, _ = load_scene_comparison(
        args.scene,
        prediction_roots,
        args.label_key,
        args.mask_key,
        args.prediction_key,
        args.min_depth_m,
        args.max_depth_m,
    )
    plot_path, summary_path = render_scene_comparison(args.scene, gt, preds, metrics, Path(args.output_dir))
    print(f"Saved plot to {plot_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
