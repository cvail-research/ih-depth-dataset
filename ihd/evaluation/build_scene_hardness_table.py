from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Aggregate per-scene depth metrics from multiple models into a hardness table."
    )
    ap.add_argument(
        "--metrics-csv",
        action="append",
        required=True,
        help="Per-scene metrics CSV produced by evaluate_depth_prediction.py. Repeat for each model.",
    )
    ap.add_argument(
        "--score-col",
        default="abs_rel",
        help="Metric column to rank scenes by. Default: abs_rel.",
    )
    ap.add_argument(
        "--scene-col",
        default="scene",
        help="Scene identifier column in the metrics CSVs.",
    )
    ap.add_argument(
        "--model-col",
        default="model",
        help="Model name column in the metrics CSVs.",
    )
    ap.add_argument(
        "--out-csv",
        default="analysis/evaluation/scene_hardness_table.csv",
        help="Output scene hardness table.",
    )
    ap.add_argument(
        "--out-json",
        default="analysis/evaluation/scene_hardness_table_summary.json",
        help="Output summary JSON.",
    )
    return ap.parse_args()


def read_metrics(path: Path, score_col: str, scene_col: str, model_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if scene_col not in df.columns:
        raise KeyError(f"{path} missing scene column '{scene_col}'.")
    if model_col not in df.columns:
        raise KeyError(f"{path} missing model column '{model_col}'.")
    if score_col not in df.columns:
        raise KeyError(f"{path} missing score column '{score_col}'.")
    out = df[[scene_col, model_col, score_col]].copy()
    out = out.rename(columns={scene_col: "scene", model_col: "model", score_col: "score"})
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out = out[out["score"].notna()].copy()
    return out


def main() -> None:
    args = parse_args()
    metrics = [read_metrics(Path(path), args.score_col, args.scene_col, args.model_col) for path in args.metrics_csv]
    if not metrics:
        raise SystemExit("No metrics CSVs provided.")

    combined = pd.concat(metrics, ignore_index=True)
    grouped = combined.groupby("scene")["score"]
    summary = grouped.agg(
        hardness_mean="mean",
        hardness_std="std",
        hardness_num_models="count",
    ).reset_index()
    summary["hardness_std"] = summary["hardness_std"].fillna(0.0)
    summary["hardness_rank"] = summary["hardness_mean"].rank(method="dense", ascending=False).astype(int)
    summary = summary.sort_values(["hardness_mean", "scene"], ascending=[False, True]).reset_index(drop=True)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "scene_count": int(len(summary)),
                "model_count": int(combined["model"].nunique()),
                "score_col": args.score_col,
                "mean_of_hardness_mean": float(summary["hardness_mean"].mean()) if len(summary) else None,
                "median_of_hardness_mean": float(summary["hardness_mean"].median()) if len(summary) else None,
                "high_std_scene_count_ge_0p10": int((summary["hardness_std"] >= 0.10).sum()),
                "high_std_scene_count_ge_0p20": int((summary["hardness_std"] >= 0.20).sum()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Saved hardness table to {out_csv}")
    print(f"Saved summary to {out_json}")


if __name__ == "__main__":
    main()
