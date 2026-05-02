from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ihd.evaluation.plot_scene_model_comparison import load_scene_comparison, render_scene_comparison


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Render model comparison plots for a batch of scenes.")
    ap.add_argument("--scene-manifest", required=True, help="CSV containing scenes to render.")
    ap.add_argument(
        "--prediction-root",
        action="append",
        required=True,
        help="Model prediction root containing prediction_manifest.csv. Repeat 4 times.",
    )
    ap.add_argument("--output-dir", default="analysis/evaluation/scene_model_comparison_batch")
    ap.add_argument("--label-key", default="depth_m")
    ap.add_argument("--mask-key", default="valid_mask")
    ap.add_argument("--prediction-key", default="depth_m")
    ap.add_argument("--include-only", action="store_true", help="Render only rows with release_decision == include.")
    ap.add_argument("--limit", type=int, help="Optional max number of scenes to render.")
    ap.add_argument("--min-depth-m", type=float, default=0.0)
    ap.add_argument("--max-depth-m", type=float, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    prediction_roots = [Path(path) for path in args.prediction_root]
    if len(prediction_roots) != 4:
        raise SystemExit("Provide exactly four --prediction-root values, one per model.")

    manifest = pd.read_csv(args.scene_manifest)
    if args.include_only and "release_decision" in manifest.columns:
        manifest = manifest[manifest["release_decision"].astype(str).str.lower() == "include"].copy()
    if "scene" not in manifest.columns:
        if {"collection", "path", "step"}.issubset(manifest.columns):
            manifest["scene"] = manifest["collection"].astype(str) + " / " + manifest["path"].astype(str) + " / " + manifest["step"].astype(str)
        else:
            raise SystemExit("Scene manifest must contain a 'scene' column or collection/path/step columns.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, scene in enumerate(manifest["scene"].astype(str)):
        if args.limit is not None and idx >= args.limit:
            break
        try:
            gt, _, preds, metrics, _ = load_scene_comparison(
                scene,
                prediction_roots,
                args.label_key,
                args.mask_key,
                args.prediction_key,
                args.min_depth_m,
                args.max_depth_m,
            )
            plot_path, summary_path = render_scene_comparison(scene, gt, preds, metrics, out_dir)
            rows.append(
                {
                    "scene": scene,
                    "status": "ok",
                    "plot_path": str(plot_path),
                    "summary_path": str(summary_path),
                    "score_mean_abs_rel": json.loads(summary_path.read_text()).get("score_mean_abs_rel"),
                    "score_std_abs_rel": json.loads(summary_path.read_text()).get("score_std_abs_rel"),
                }
            )
        except Exception as exc:  # pragma: no cover - batch resilience
            rows.append(
                {
                    "scene": scene,
                    "status": "error",
                    "error": str(exc),
                    "plot_path": "",
                    "summary_path": "",
                    "score_mean_abs_rel": None,
                    "score_std_abs_rel": None,
                }
            )

    index_df = pd.DataFrame(rows)
    index_df.to_csv(out_dir / "batch_index.csv", index=False)
    summary = {
        "scene_count": int(len(index_df)),
        "ok_count": int((index_df["status"] == "ok").sum()),
        "error_count": int((index_df["status"] == "error").sum()),
        "output_dir": str(out_dir),
    }
    (out_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
