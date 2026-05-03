from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
    ap.add_argument(
        "--depth-label-root",
        default="analysis/depth_labels/platform_sphere_r4p0",
        help="Depth-label root used to determine prediction readiness.",
    )
    ap.add_argument(
        "--scene-spot-mapping",
        default="manifests/scene_spot_mapping_v0.csv",
        help="Optional manual path-to-scene-spot mapping CSV.",
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


def normalized_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def projected_depth_label_path(depth_label_root: Path, row: pd.Series) -> Path:
    return depth_label_root / str(row["collection"]) / str(row["path"]) / str(row["step"]) / "projected_lidar_depth_label.npz"


def scene_spot_id_default(row: pd.Series) -> str:
    return f"{str(row['collection']).strip()} / {str(row['path']).strip()}"


def merge_scene_spot_mapping(df: pd.DataFrame, mapping_path: Path) -> pd.DataFrame:
    out = df.copy()
    out["scene_spot_id"] = out.apply(scene_spot_id_default, axis=1)
    if not mapping_path.exists():
        return out

    mapping = pd.read_csv(mapping_path)
    required = {"collection", "path", "scene_spot_id"}
    missing = required - set(mapping.columns)
    if missing:
        raise KeyError(f"Scene-spot mapping {mapping_path} is missing columns: {sorted(missing)}")
    mapping = mapping.copy()
    mapping["collection"] = mapping["collection"].astype(str).str.strip()
    mapping["path"] = mapping["path"].astype(str).str.strip()
    mapping["scene_spot_id"] = mapping["scene_spot_id"].astype(str).str.strip()
    mapping = mapping[["collection", "path", "scene_spot_id"]].drop_duplicates()
    out = out.merge(mapping, on=["collection", "path"], how="left", suffixes=("", "_mapped"))
    out["scene_spot_id"] = out["scene_spot_id_mapped"].fillna(out["scene_spot_id"])
    out = out.drop(columns=["scene_spot_id_mapped"])
    return out


def registration_provenance_current(row: pd.Series) -> str:
    annotation_status = str(row.get("annotation_status", "")).strip().lower()
    legacy_pool = str(row.get("source_pool", "")).strip().lower()
    has_session_json = bool(row.get("has_session_json", False))
    has_fit_json = bool(row.get("has_fit_json", False))
    fit_ready = bool(row.get("fit_ready", False))

    if annotation_status == "skipped_lwhsi_artifacts":
        return "skipped_before_annotation_lwhsi_artifacts"
    if fit_ready:
        return "ours_fit_ready"
    if has_fit_json:
        return "ours_fit_not_ready"
    if has_session_json:
        return "ours_session_no_fit"
    if legacy_pool == "with_prior_cyl":
        return "legacy_prior_pool_no_ours_fit"
    return "no_ours_fit_artifacts"


def prediction_exclusion_reason_current(row: pd.Series) -> str | None:
    if bool(row.get("prediction_ready_current", False)):
        return None
    annotation_status = str(row.get("annotation_status", "")).strip().lower()
    verdict = str(row.get("verdict", "")).strip().lower()
    if annotation_status == "skipped_lwhsi_artifacts":
        return "skipped_lwhsi_artifacts"
    if verdict == BAD_VERDICT:
        return "no_fit_ready_bad_scene"
    if not bool(row.get("fit_ready", False)):
        return "no_fit_ready_missing_projected_depth_label"
    return "missing_projected_depth_label"


def merge_quality_and_cleanup(
    quality: pd.DataFrame,
    cleanup: pd.DataFrame,
    depth_label_root: Path,
    scene_spot_mapping: Path,
) -> pd.DataFrame:
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
    merged["scene_manifest_scope"] = "all_306_scenes"
    merged["release_registration_policy"] = "ours_only"
    merged["legacy_source_pool"] = merged["source_pool"]
    merged["legacy_prior_cyl_member"] = merged["legacy_source_pool"].astype(str).str.lower().eq("with_prior_cyl")
    merged = merge_scene_spot_mapping(merged, scene_spot_mapping)
    merged["registration_provenance_current"] = merged.apply(registration_provenance_current, axis=1)
    merged["projected_depth_label_path_current"] = merged.apply(
        lambda row: str(projected_depth_label_path(depth_label_root, row)),
        axis=1,
    )
    merged["prediction_ready_current"] = merged["projected_depth_label_path_current"].map(lambda p: Path(p).exists())
    merged["prediction_exclusion_reason_current"] = merged.apply(prediction_exclusion_reason_current, axis=1)
    merged["has_cleanup_review"] = merged["cleanup_status"] != "not_reviewed"
    merged["frozen_quant_gate_pass"] = (
        merged["verdict"].astype(str).str.lower().eq(GOOD_VERDICT)
        & merged["candidate_rmse5_distance5_current"].astype(bool)
    )

    column_order = [
        "frozen_manifest_version",
        "scene_manifest_scope",
        "collection",
        "path",
        "step",
        "scene",
        "release_registration_policy",
        "registration_provenance_current",
        "legacy_prior_cyl_member",
        "legacy_source_pool",
        "scene_spot_id",
        "release_decision",
        "release_reason",
        "frozen_quant_gate_pass",
        "prediction_ready_current",
        "prediction_exclusion_reason_current",
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
        "manual_release_decision",
        "manual_release_reason",
        "fit_json",
        "session_json",
        "projected_depth_label_path_current",
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
    manual_decision = normalized_text(row.get("manual_release_decision", ""))
    if manual_decision:
        if manual_decision.lower() not in {"include", "defer", "exclude"}:
            raise ValueError(f"Invalid manual_release_decision value: {manual_decision}")
        return manual_decision.lower()

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
    if normalized_text(row.get("exclusion_reason", "")):
        return "defer"
    if str(row.get("cleanup_status", "")).strip().lower() == "rejected":
        return "exclude"
    return "include"


def determine_release_reason(row: pd.Series) -> str | None:
    manual_reason = normalized_text(row.get("manual_release_reason", ""))
    manual_decision = normalized_text(row.get("manual_release_decision", ""))
    if manual_decision:
        if manual_reason:
            return manual_reason
        return f"manual release decision: {manual_decision.lower()}"

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
    exclusion_reason = normalized_text(row.get("exclusion_reason", ""))
    if exclusion_reason:
        return f"manual qc exclusion: {exclusion_reason}"
    if str(row.get("cleanup_status", "")).strip().lower() == "rejected":
        return "occlusion cleanup rejected"
    return None


def build_summary(df: pd.DataFrame, quality_path: Path, cleanup_path: Path, depth_label_root: Path) -> dict:
    release_counts = df["release_decision"].value_counts(dropna=False).to_dict()
    cleanup_counts = df["cleanup_status"].value_counts(dropna=False).to_dict()
    return {
        "frozen_manifest_version": "v0",
        "quality_manifest": str(quality_path),
        "cleanup_manifest": str(cleanup_path),
        "depth_label_root": str(depth_label_root),
        "scene_count": int(len(df)),
        "release_decision_counts": release_counts,
        "cleanup_status_counts": cleanup_counts,
        "legacy_source_pool_counts": df["legacy_source_pool"].value_counts(dropna=False).to_dict(),
        "scene_spot_id_counts": df["scene_spot_id"].value_counts(dropna=False).to_dict(),
        "registration_provenance_current_counts": df["registration_provenance_current"].value_counts(dropna=False).to_dict(),
        "include_count": int((df["release_decision"] == "include").sum()),
        "defer_count": int((df["release_decision"] == "defer").sum()),
        "exclude_count": int((df["release_decision"] == "exclude").sum()),
        "cleanup_reviewed_count": int(df["has_cleanup_review"].sum()),
        "quant_gate_pass_count": int(df["frozen_quant_gate_pass"].sum()),
        "good_count": int((df["verdict"].astype(str).str.lower() == GOOD_VERDICT).sum()),
        "candidate_count": int(df["candidate_rmse5_distance5_current"].astype(bool).sum()),
        "good_and_candidate_count": int(((df["verdict"].astype(str).str.lower() == GOOD_VERDICT) & df["candidate_rmse5_distance5_current"].astype(bool)).sum()),
        "prediction_ready_current_count": int(df["prediction_ready_current"].sum()),
        "prediction_not_ready_current_count": int((~df["prediction_ready_current"]).sum()),
        "prediction_exclusion_reason_current_counts": df["prediction_exclusion_reason_current"].fillna("prediction_ready").value_counts(dropna=False).to_dict(),
    }


def main() -> None:
    args = parse_args()
    quality_path = Path(args.quality_manifest)
    cleanup_path = Path(args.cleanup_manifest)
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_summary_json)
    depth_label_root = Path(args.depth_label_root)
    scene_spot_mapping = Path(args.scene_spot_mapping)

    quality = load_csv(quality_path)
    cleanup = load_csv(cleanup_path)
    frozen = merge_quality_and_cleanup(quality, cleanup, depth_label_root, scene_spot_mapping)
    summary = build_summary(frozen, quality_path, cleanup_path, depth_label_root)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frozen.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"Saved frozen manifest to {output_csv}")
    print(f"Saved summary to {output_json}")
    print(f"Scene count: {summary['scene_count']}")
    print(f"Decision counts: {summary['release_decision_counts']}")


if __name__ == "__main__":
    main()
