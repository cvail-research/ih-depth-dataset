from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build the common-scene manifest across prediction manifests.")
    ap.add_argument(
        "--prediction-manifest",
        action="append",
        required=True,
        help="Prediction manifest CSV. Repeat once per model.",
    )
    ap.add_argument(
        "--base-manifest",
        help="Optional manifest to filter against, e.g. a frozen include manifest.",
    )
    ap.add_argument(
        "--out-csv",
        default="analysis/evaluation/common_scene_manifest.csv",
        help="Output CSV containing the intersected scene list.",
    )
    return ap.parse_args()


def load_scenes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "scene" not in df.columns:
        raise KeyError(f"{path} missing 'scene' column.")
    return df[["scene"]].drop_duplicates().copy()


def main() -> None:
    args = parse_args()
    manifests = [load_scenes(Path(path)) for path in args.prediction_manifest]
    if not manifests:
        raise SystemExit("No prediction manifests provided.")

    common = manifests[0]
    for other in manifests[1:]:
        common = common.merge(other, on="scene", how="inner")

    if args.base_manifest:
        base = pd.read_csv(args.base_manifest)
        if "scene" not in base.columns:
            raise KeyError(f"{args.base_manifest} missing 'scene' column.")
        common = common.merge(base[["scene"]].drop_duplicates(), on="scene", how="inner")

    common = common.sort_values("scene").reset_index(drop=True)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    common.to_csv(out, index=False)
    print(f"Saved {len(common)} common scenes to {out}")


if __name__ == "__main__":
    main()
