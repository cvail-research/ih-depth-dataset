from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ihd.evaluation.depth_metrics import DepthEvalConfig, depth_metrics_from_arrays, summarize_metric_rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Evaluate IH-Depth metric-depth predictions against sparse projected LiDAR labels."
    )
    input_group = ap.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--prediction", help="Single prediction .npy/.npz path.")
    input_group.add_argument("--manifest", help="CSV with prediction_path and label_path columns.")
    ap.add_argument("--label", help="Single ground-truth label .npz/.npy path.")
    ap.add_argument("--prediction-key", default="depth_m", help="Prediction key for .npz files.")
    ap.add_argument("--label-key", default="depth_m", help="Ground-truth depth key for .npz files.")
    ap.add_argument("--mask-key", default="valid_mask", help="Ground-truth valid mask key for .npz files.")
    ap.add_argument("--min-depth-m", type=float, default=0.0)
    ap.add_argument("--max-depth-m", type=float)
    ap.add_argument("--median-scale", action="store_true", help="Evaluate after median scale alignment.")
    ap.add_argument("--no-ssi", action="store_true", help="Disable affine scale/shift-invariant metrics.")
    ap.add_argument("--out-json", default="analysis/evaluation/depth_metrics_summary.json")
    ap.add_argument("--out-csv", default="analysis/evaluation/depth_metrics_per_scene.csv")
    return ap.parse_args()


def load_array(path: Path, key: str) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        with np.load(path) as data:
            if key not in data:
                raise KeyError(f"{path} does not contain key '{key}'. Available keys: {list(data.files)}")
            return data[key]
    raise ValueError(f"Unsupported array format for {path}. Use .npy or .npz.")


def load_optional_mask(path: Path, key: str) -> np.ndarray | None:
    if path.suffix != ".npz":
        return None
    with np.load(path) as data:
        if key not in data:
            return None
        return data[key]


def evaluate_one(
    prediction_path: Path,
    label_path: Path,
    prediction_key: str,
    label_key: str,
    mask_key: str,
    config: DepthEvalConfig,
) -> dict[str, Any]:
    pred = load_array(prediction_path, prediction_key)
    label = load_array(label_path, label_key)
    mask = load_optional_mask(label_path, mask_key)
    metrics = depth_metrics_from_arrays(pred, label, mask, config)
    return {
        "prediction_path": str(prediction_path),
        "label_path": str(label_path),
        **metrics,
    }


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"prediction_path", "label_path"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"Manifest {path} is missing columns: {sorted(missing)}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = DepthEvalConfig(
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        apply_median_scale=args.median_scale,
        compute_ssi=not args.no_ssi,
    )

    if args.prediction:
        if not args.label:
            raise SystemExit("--label is required when using --prediction.")
        rows = [
            evaluate_one(
                Path(args.prediction),
                Path(args.label),
                args.prediction_key,
                args.label_key,
                args.mask_key,
                config,
            )
        ]
    else:
        manifest_rows = read_manifest(Path(args.manifest))
        rows = []
        for row in manifest_rows:
            metrics = evaluate_one(
                Path(row["prediction_path"]),
                Path(row["label_path"]),
                row.get("prediction_key") or args.prediction_key,
                row.get("label_key") or args.label_key,
                row.get("mask_key") or args.mask_key,
                config,
            )
            for key, value in row.items():
                if key not in metrics:
                    metrics[key] = value
            rows.append(metrics)

    summary = summarize_metric_rows(rows)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_csv(Path(args.out_csv), rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

