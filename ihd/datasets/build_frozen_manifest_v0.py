from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


GOOD_VERDICT = "good"
CAUTION_VERDICT = "usable with caution"
BAD_VERDICT = "bad"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build the frozen IH-Depth v0 manifest.")
    ap.add_argument(
        "--quality-manifest",
        default="manifests/05_scene_quality_manifest_current.csv",
        help="Current scene quality manifest CSV.",
    )
    ap.add_argument(
        "--cleanup-manifest",
        default="manifests/06_occlusion_cleanup_manifest_current.csv",
        help="Current occlusion cleanup manifest CSV.",
    )
    ap.add_argument(
        "--output-csv",
        default="manifests/07_frozen_manifest_v0.csv",
        help="Frozen manifest CSV to write.",
    )
    ap.add_argument(
        "--output-summary-json",
        default="manifests/07_frozen_manifest_v0_summary.json",
        help="Summary JSON to write.",
    )
    return ap.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def scene_join_key(collection: str, path: str, step: str) -> str:
    return f"{collection}|{path}|{step}"


def cleanup_join_key(row: pd.Series) -> str:
    path = str(row["path_key"]).strip()
    step_num = int(row["step"])
    return scene_join_key(str(row["collection"]).strip(), path, f"{path}_step{step_num}")


def merge_quality_and_cleanup(quality: pd.DataFrame, cleanup: pd.DataFrame) -> pd.DataFrame:
    quality = quality.copy()
    cleanup = cleanup.copy()

    quality["join_key"] = quality.apply(
        lambda row: scene_join_key(str(row["collection"]).strip(), str(row["path"]).strip(), str(row["step"]).strip()),
        axis=1,
    )
    cleanup["join_key"] = cleanup.apply(cleanup_join_key, axis=1)

    merged = quality.merge(
        cleanup[
            [
                "join_key",
                "cleanup_status",
                "cleanup_region_count",
                "cleanup_region_ids_json",
                "cleanup_regions_json",
                "selection_mode_summary_json",
                "center_x_m",
                "center_y_m",
                "center_z_m",
                "half_extent_m",
                "removed_points",
                "kept_points",
                "fit_rmse_total_px",
                "source_projection_las",
                "cleaned_las",
                "raw_overlay",
                "cleanup_overlay",
                "updated_at",
            ]
        ],
        on="join_key",
        how="left",
        suffixes=("", "_cleanup"),
    )

    merged["cleanup_status"] = merged["cleanup_status"].fillna("not_reviewed")
    merged["cleanup_region_count"] = merged["cleanup_region_count"].fillna(0).astype(int)

    merged["release_decision"] = merged.apply(determine_release_decision, axis=1)
    merged["release_reason"] = merged.apply(determine_release_reason, axis=1)
    merged["frozen_manifest_version"] = "v0"
    merged["has_cleanup_review"] = merged["cleanup_status"] != "not_reviewed"
    merged["frozen_quant_gate_pass"] = (
        merged["verdict"].astype(str).str.lower().eq(GOOD_VERDICT)
        & merged["candidate_rmse5_distance5_current"].astype(bool)
    )

    column_order = [
        "frozen_manifest_version",
        "collection",
        "path",
        "step",
        "scene",
        "source_pool",
        "release_decision",
        "release_reason",
        "frozen_quant_gate_pass",
        "verdict",
        "annotation_status",
        "fit_ready",
        "has_session_json",
        "has_fit_json",
        "fit_rmse_total_px",
        "fit_rmse_u_px",
        "fit_rmse_v_px",
        "rmse_pass_le_5px",
        "rmse_pass_le_10px",
        "num_picked_pairs",
        "elapsed_seconds_plausible_for_summary",
        "replacement_count",
        "clear_count",
        "distance_agreement_status",
        "distance_agreement_available_current",
        "distance_sampled_points",
        "distance_missing_depth_points",
        "distance_scene_error_count",
        "distance_near_points_0_10m",
        "distance_middle_points_10_100m",
        "distance_far_points_100_inf",
        "distance_max_error_percent_of_picked_range",
        "distance_median_error_percent_of_picked_range",
        "distance_mean_error_percent_of_picked_range",
        "distance_max_error_m",
        "distance_median_error_m",
        "distance_all_points_pass_le_1pct",
        "distance_all_points_pass_le_5pct",
        "candidate_rmse5_distance5_current",
        "has_cleanup_review",
        "cleanup_status",
        "cleanup_region_count",
        "cleanup_region_ids_json",
        "cleanup_regions_json",
        "selection_mode_summary_json",
        "center_x_m",
        "center_y_m",
        "center_z_m",
        "half_extent_m",
        "removed_points",
        "kept_points",
        "source_projection_las",
        "cleaned_las",
        "raw_overlay",
        "cleanup_overlay",
        "updated_at",
        "fit_json",
        "session_json",
        "exclusion_reason",
        "timing_exclusion_reason",
        "qc_status",
        "qc_majority_verdict",
        "qc_num_reviews",
        "qc_vote_good",
        "qc_vote_usable_with_caution",
        "qc_vote_bad",
        "qc_mean_seconds",
        "qc_notes",
        "corresp_path",
        "cyl_path",
        "hdr_path",
        "bsq_path",
        "las_path",
        "join_key",
    ]

    for column in column_order:
        if column not in merged.columns:
            merged[column] = pd.NA

    return merged[column_order].sort_values(["collection", "path", "step", "scene"]).reset_index(drop=True)


def determine_release_decision(row: pd.Series) -> str:
    verdict = str(row.get("verdict", "")).strip().lower()
    if verdict == BAD_VERDICT:
        return "exclude"
    if verdict == CAUTION_VERDICT:
        return "defer"
    if verdict != GOOD_VERDICT:
        return "defer"
    if not bool(row.get("fit_ready", False)):
        return "exclude"
    if not bool(row.get("distance_agreement_available_current", False)):
        return "exclude"
    if not bool(row.get("candidate_rmse5_distance5_current", False)):
        return "defer"
    if str(row.get("cleanup_status", "")).strip().lower() == "rejected":
        return "exclude"
    return "include"


def determine_release_reason(row: pd.Series) -> str | None:
    verdict = str(row.get("verdict", "")).strip().lower()
    if verdict == BAD_VERDICT:
        return "qc verdict bad"
    if verdict == CAUTION_VERDICT:
        return "qc verdict usable with caution"
    if verdict != GOOD_VERDICT:
        return "missing final qc verdict"
    if not bool(row.get("fit_ready", False)):
        return "missing fit-ready scene artifacts"
    if not bool(row.get("distance_agreement_available_current", False)):
        return "missing distance agreement"
    if not bool(row.get("candidate_rmse5_distance5_current", False)):
        return "does not meet frozen rmse5+distance5 gate"
    if str(row.get("cleanup_status", "")).strip().lower() == "rejected":
        return "occlusion cleanup rejected"
    return None


def build_summary(df: pd.DataFrame, quality_path: Path, cleanup_path: Path) -> dict:
    release_counts = df["release_decision"].value_counts(dropna=False).to_dict()
    cleanup_counts = df["cleanup_status"].value_counts(dropna=False).to_dict()
    return {
        "frozen_manifest_version": "v0",
        "quality_manifest": str(quality_path),
        "cleanup_manifest": str(cleanup_path),
        "scene_count": int(len(df)),
        "release_decision_counts": release_counts,
        "cleanup_status_counts": cleanup_counts,
        "include_count": int((df["release_decision"] == "include").sum()),
        "defer_count": int((df["release_decision"] == "defer").sum()),
        "exclude_count": int((df["release_decision"] == "exclude").sum()),
        "cleanup_reviewed_count": int(df["has_cleanup_review"].sum()),
        "quant_gate_pass_count": int(df["frozen_quant_gate_pass"].sum()),
        "good_count": int((df["verdict"].astype(str).str.lower() == GOOD_VERDICT).sum()),
        "candidate_count": int(df["candidate_rmse5_distance5_current"].astype(bool).sum()),
        "good_and_candidate_count": int(((df["verdict"].astype(str).str.lower() == GOOD_VERDICT) & df["candidate_rmse5_distance5_current"].astype(bool)).sum()),
    }


def main() -> None:
    args = parse_args()
    quality_path = Path(args.quality_manifest)
    cleanup_path = Path(args.cleanup_manifest)
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_summary_json)

    quality = load_csv(quality_path)
    cleanup = load_csv(cleanup_path)
    frozen = merge_quality_and_cleanup(quality, cleanup)
    summary = build_summary(frozen, quality_path, cleanup_path)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frozen.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"Saved frozen manifest to {output_csv}")
    print(f"Saved summary to {output_json}")
    print(f"Scene count: {summary['scene_count']}")
    print(f"Decision counts: {summary['release_decision_counts']}")


if __name__ == "__main__":
    main()
