from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Merge per-model IH depth metrics into a long-format scene score table.")
    ap.add_argument("--predictions-root", required=True)
    ap.add_argument("--out-csv", default="analysis/evaluation/baseline_scene_scores.csv")
    ap.add_argument("--metric", default="abs_rel")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.predictions_root)
    rows = []
    for metrics_path in sorted(root.glob("*/metrics_per_scene.csv")):
        model = metrics_path.parent.name
        df = pd.read_csv(metrics_path)
        if args.metric not in df.columns:
            continue
        for row in df.itertuples(index=False):
            data = row._asdict()
            rows.append(
                {
                    "collection": data.get("collection"),
                    "path": data.get("path"),
                    "step": data.get("step"),
                    "scene": data.get("scene"),
                    "model": model,
                    args.metric: data.get(args.metric),
                    "prediction_path": data.get("prediction_path"),
                    "label_path": data.get("label_path"),
                }
            )
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {len(rows)} model-scene metric rows to {out}")


if __name__ == "__main__":
    main()

