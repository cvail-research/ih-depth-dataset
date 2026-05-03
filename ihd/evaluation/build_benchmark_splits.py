from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_GROUP_COLS = ("collection", "path")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build deterministic IH-Depth train/val/test splits from a frozen or accepted-scene manifest. "
            "If model hardness scores are provided, test groups are selected from the worst-performing groups."
        )
    )
    ap.add_argument("--manifest", required=True, help="Frozen manifest or accepted-scene CSV.")
    ap.add_argument("--out-dir", default="analysis/splits/ih_depth_v0")
    ap.add_argument(
        "--group-cols",
        default="collection,path",
        help="Comma-separated columns used to prevent leakage across splits. Default: collection,path.",
    )
    ap.add_argument(
        "--accepted-only",
        action="store_true",
        help="If release_decision exists, keep only rows with release_decision == include.",
    )
    ap.add_argument(
        "--hardness-csv",
        help=(
            "Optional long-format baseline score CSV with scene identifiers, model column, and score column. "
            "Higher scores are treated as harder unless --lower-is-harder is set."
        ),
    )
    ap.add_argument("--hardness-score-col", default="abs_rel")
    ap.add_argument("--hardness-model-col", default="model")
    ap.add_argument("--lower-is-harder", action="store_true")
    ap.add_argument("--test-fraction", type=float, default=0.10)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--min-test-groups", type=int, default=1)
    ap.add_argument("--min-val-groups", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument(
        "--ambiguity-std-threshold",
        type=float,
        default=0.10,
        help="Scenes with baseline score std >= threshold are flagged for optional human review.",
    )
    return ap.parse_args()


def stable_scene_id(row: pd.Series) -> str:
    if "scene_id" in row and pd.notna(row["scene_id"]):
        return str(row["scene_id"])
    if "scene" in row and pd.notna(row["scene"]):
        return str(row["scene"])
    collection = str(row.get("collection", ""))
    path = str(row.get("path", ""))
    step = str(row.get("step", ""))
    return f"{collection} / {path} / {step}"


def stable_group_key(row: pd.Series, group_cols: list[str]) -> str:
    values = []
    for col in group_cols:
        if col not in row:
            raise KeyError(f"Missing split group column '{col}' in manifest.")
        values.append(str(row[col]))
    return " / ".join(values)


def deterministic_shuffle(values: list[str], seed: int) -> list[str]:
    shuffled = list(values)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled


def load_manifest(path: Path, group_cols: list[str], accepted_only: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    if accepted_only and "release_decision" in df.columns:
        df = df[df["release_decision"].astype(str).str.lower() == "include"].copy()
    if df.empty:
        raise SystemExit(f"No scenes available after filtering manifest {path}")
    df["scene_id_for_split"] = df.apply(stable_scene_id, axis=1)
    df["split_group"] = df.apply(lambda row: stable_group_key(row, group_cols), axis=1)
    return df


def load_hardness_scores(path: Path, score_col: str, model_col: str, lower_is_harder: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    if score_col not in df.columns:
        raise KeyError(f"Hardness CSV {path} is missing score column '{score_col}'.")
    if model_col not in df.columns:
        raise KeyError(f"Hardness CSV {path} is missing model column '{model_col}'.")
    df["scene_id_for_split"] = df.apply(stable_scene_id, axis=1)
    df["hardness_score_raw"] = pd.to_numeric(df[score_col], errors="coerce")
    df = df[np.isfinite(df["hardness_score_raw"])].copy()
    df["hardness_score"] = -df["hardness_score_raw"] if lower_is_harder else df["hardness_score_raw"]
    return df


def load_hardness_summary(path: Path, scene_col: str = "scene") -> pd.DataFrame:
    df = pd.read_csv(path)
    if scene_col not in df.columns:
        raise KeyError(f"Hardness CSV {path} is missing scene column '{scene_col}'.")
    required = {"hardness_mean", "hardness_std"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Hardness CSV {path} is missing required columns: {sorted(missing)}")
    out = df.copy()
    out["scene_id_for_split"] = out[scene_col].astype(str)
    out["hardness_mean"] = pd.to_numeric(out["hardness_mean"], errors="coerce")
    out["hardness_std"] = pd.to_numeric(out["hardness_std"], errors="coerce").fillna(0.0)
    if "hardness_num_models" not in out.columns:
        out["hardness_num_models"] = 0
    out["hardness_ambiguous_for_review"] = False
    return out


def attach_hardness(
    manifest: pd.DataFrame,
    hardness_scores: pd.DataFrame | None,
    ambiguity_std_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if hardness_scores is None:
        scene = manifest[["scene_id_for_split", "split_group"]].drop_duplicates().copy()
        scene["hardness_mean"] = np.nan
        scene["hardness_std"] = np.nan
        scene["hardness_num_models"] = 0
        scene["hardness_ambiguous_for_review"] = False
    elif "hardness_score" in hardness_scores.columns:
        grouped = hardness_scores.groupby("scene_id_for_split")["hardness_score"]
        scene = grouped.agg(
            hardness_mean="mean",
            hardness_std="std",
            hardness_num_models="count",
        ).reset_index()
        scene["hardness_std"] = scene["hardness_std"].fillna(0.0)
        scene["hardness_ambiguous_for_review"] = scene["hardness_std"] >= ambiguity_std_threshold
        scene = manifest[["scene_id_for_split", "split_group"]].drop_duplicates().merge(
            scene, on="scene_id_for_split", how="left"
        )
        scene["hardness_num_models"] = scene["hardness_num_models"].fillna(0).astype(int)
    else:
        scene = manifest[["scene_id_for_split", "split_group"]].drop_duplicates().merge(
            hardness_scores[["scene_id_for_split", "hardness_mean", "hardness_std", "hardness_num_models"]],
            on="scene_id_for_split",
            how="left",
        )
        scene["hardness_ambiguous_for_review"] = scene["hardness_std"] >= ambiguity_std_threshold

    manifest = manifest.merge(scene, on=["scene_id_for_split", "split_group"], how="left")
    return manifest, scene.sort_values(["hardness_mean", "scene_id_for_split"], ascending=[False, True])


def group_collection(group_key: str) -> str:
    return group_key.split(" / ")[0]


def select_test_groups(group_table: pd.DataFrame, test_fraction: float, min_test_groups: int) -> set[str]:
    test_groups: set[str] = set()
    for collection, sub in group_table.groupby("collection_for_split"):
        if len(sub) == 0:
            continue
        n_test = max(min_test_groups, int(math.ceil(len(sub) * test_fraction)))
        n_test = min(n_test, max(len(sub) - 1, 1))
        ranked = sub.sort_values(["group_hardness_mean", "split_group"], ascending=[False, True])
        test_groups.update(ranked.head(n_test)["split_group"].tolist())
    return test_groups


def select_val_groups(
    remaining_groups: list[str],
    val_fraction: float,
    min_val_groups: int,
    seed: int,
    group_to_collection: dict[str, str],
) -> set[str]:
    by_collection: dict[str, list[str]] = defaultdict(list)
    for group in remaining_groups:
        by_collection[group_to_collection[group]].append(group)

    val_groups: set[str] = set()
    for collection, groups in by_collection.items():
        if len(groups) <= 1:
            continue
        n_val = max(min_val_groups, int(math.ceil(len(groups) * val_fraction)))
        n_val = min(n_val, len(groups) - 1)
        collection_seed = seed + int(hashlib.sha1(collection.encode("utf-8")).hexdigest()[:8], 16)
        val_groups.update(deterministic_shuffle(groups, collection_seed)[:n_val])
    return val_groups


def assign_splits(manifest: pd.DataFrame, seed: int, test_fraction: float, val_fraction: float, min_test_groups: int, min_val_groups: int) -> pd.DataFrame:
    group_table = (
        manifest.groupby("split_group")
        .agg(
            num_scenes=("scene_id_for_split", "count"),
            group_hardness_mean=("hardness_mean", "mean"),
            group_hardness_max=("hardness_mean", "max"),
        )
        .reset_index()
    )
    group_collection_map = (
        manifest.groupby("split_group")["collection"].first().astype(str).to_dict()
    )
    group_table["collection_for_split"] = group_table["split_group"].map(group_collection_map)
    no_hardness = ~np.isfinite(group_table["group_hardness_mean"])
    if no_hardness.all():
        # Without baseline scores, use a deterministic hash ordering. The split
        # is reproducible, but the test set is not yet hard-case selected.
        group_table["group_hardness_mean"] = group_table["split_group"].map(
            lambda value: int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:12], 16)
        )
        group_table["group_hardness_max"] = group_table["group_hardness_mean"]
    else:
        group_table["group_hardness_mean"] = group_table["group_hardness_mean"].fillna(-np.inf)
        group_table["group_hardness_max"] = group_table["group_hardness_max"].fillna(-np.inf)

    test_groups = select_test_groups(group_table, test_fraction, min_test_groups)
    remaining_groups = sorted(set(group_table["split_group"]) - test_groups)
    val_groups = select_val_groups(remaining_groups, val_fraction, min_val_groups, seed, group_collection_map)

    split_by_group = {group: "train" for group in group_table["split_group"]}
    split_by_group.update({group: "test" for group in test_groups})
    split_by_group.update({group: "val" for group in val_groups})

    out = manifest.copy()
    out["split"] = out["split_group"].map(split_by_group)
    return out


def write_split_outputs(out_dir: Path, split_df: pd.DataFrame, scene_hardness: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(out_dir / "scene_splits.csv", index=False)
    scene_hardness.to_csv(out_dir / "hard_scene_candidates.csv", index=False)
    scene_hardness[scene_hardness["hardness_ambiguous_for_review"] == True].to_csv(
        out_dir / "ambiguous_hardness_review_candidates.csv", index=False
    )

    summary: dict[str, Any] = {
        "num_scenes": int(len(split_df)),
        "num_groups": int(split_df["split_group"].nunique()),
        "split_scene_counts": {k: int(v) for k, v in split_df["split"].value_counts().sort_index().items()},
        "split_group_counts": {
            k: int(v)
            for k, v in split_df.drop_duplicates("split_group")["split"].value_counts().sort_index().items()
        },
        "collections": sorted(split_df["collection"].astype(str).unique().tolist()) if "collection" in split_df.columns else [],
        "hardness_available_scenes": int(np.isfinite(split_df["hardness_mean"]).sum()),
        "ambiguous_hardness_scenes": int((split_df["hardness_ambiguous_for_review"] == True).sum()),
    }
    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    group_cols = [col.strip() for col in args.group_cols.split(",") if col.strip()]
    if not group_cols:
        group_cols = list(DEFAULT_GROUP_COLS)

    manifest = load_manifest(Path(args.manifest), group_cols, args.accepted_only)
    hardness = None
    if args.hardness_csv:
        hardness_path = Path(args.hardness_csv)
        hardness_df = pd.read_csv(hardness_path)
        if {"hardness_mean", "hardness_std"}.issubset(hardness_df.columns):
            hardness = load_hardness_summary(hardness_path)
        else:
            hardness = load_hardness_scores(
                hardness_path,
                args.hardness_score_col,
                args.hardness_model_col,
                args.lower_is_harder,
            )
    manifest, scene_hardness = attach_hardness(manifest, hardness, args.ambiguity_std_threshold)
    split_df = assign_splits(
        manifest,
        seed=args.seed,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        min_test_groups=args.min_test_groups,
        min_val_groups=args.min_val_groups,
    )
    write_split_outputs(Path(args.out_dir), split_df, scene_hardness)
    print(f"Saved split files to {args.out_dir}")
    print(split_df["split"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
