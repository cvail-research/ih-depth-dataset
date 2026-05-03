from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ihd.evaluation.model_io import build_prediction_input_rows_from_scene_manifest


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build an HDR/label manifest for baseline prediction jobs.")
    ap.add_argument("--scene-manifest", required=True, help="Frozen or accepted-scene CSV.")
    ap.add_argument("--depth-label-root", default="analysis/depth_labels/platform_sphere_r4p0")
    ap.add_argument("--disk-root", default="/disk")
    ap.add_argument("--out-csv", default="analysis/evaluation/baseline_prediction_inputs.csv")
    ap.add_argument("--limit", type=int)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_prediction_input_rows_from_scene_manifest(
        args.scene_manifest,
        depth_label_root=args.depth_label_root,
        disk_root=args.disk_root,
        limit=args.limit,
    )
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {len(rows)} prediction inputs to {out}")


if __name__ == "__main__":
    main()
