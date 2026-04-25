import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


GOOD_VERDICT = "good"
CAUTION_VERDICT = "usable with caution"
BAD_VERDICT = "bad"
PENDING_VERDICT = "pending"
REQUIRED_RESULTS_FILES = (
    "summary.json",
    "manual_las_points.csv",
    "manual_projection_comparison.png",
    "manual_projection_residuals.csv",
    "cyl_verification_overlay.png",
    "fitted_rigid_overlay.png",
    "fitted_rigid_projection.npz",
)


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Build a frozen scene manifest for IH-Depth from per-scene evaluation "
            "outputs and optional multi-rater QC reviews."
        )
    )
    ap.add_argument(
        "--results-root",
        required=True,
        help="Root directory containing per-scene result folders with summary.json files.",
    )
    ap.add_argument(
        "--data-root",
        help=(
            "Optional dataset root used to resolve source assets such as .cyl, .hdr, .bsq, "
            ".las, and correspondence files."
        ),
    )
    ap.add_argument(
        "--qc-reviews-csv",
        help=(
            "Optional CSV with one row per reviewer verdict. Columns: scene_label, annotator_id, "
            "verdict, notes, seconds."
        ),
    )
    ap.add_argument(
        "--overrides-csv",
        help=(
            "Optional CSV with one row per scene for manual release overrides. Supported columns: "
            "scene_label, release_decision, exclusion_reason, annotation_mode, cyl_source, qc_status, qc_notes."
        ),
    )
    ap.add_argument(
        "--output-csv",
        required=True,
        help="Output CSV path for the frozen manifest.",
    )
    ap.add_argument(
        "--output-markdown",
        help="Optional output path for a paper-ready markdown quality table.",
    )
    return ap.parse_args()


def scene_id_from_summary(summary: dict, fallback_path: Path) -> str:
    label = str(summary.get("scene_label", "")).strip()
    if label:
        return label
    return fallback_path.parent.name


def infer_step_from_dir_name(step_dir_name: str) -> str | None:
    match = re.search(r"step(\d+)", step_dir_name, flags=re.IGNORECASE)
    if not match:
        return None
    return f"Step{int(match.group(1))}"


def infer_path_from_key(path_key: str) -> str | None:
    match = re.fullmatch(r"path(\d+)", path_key, flags=re.IGNORECASE)
    if not match:
        return None
    return f"Path{int(match.group(1))}_DistStA"


def read_summary(summary_path: Path) -> dict:
    with summary_path.open("r") as f:
        return json.load(f)


def find_results_rows(results_root: Path) -> list[dict]:
    rows = []
    for summary_path in sorted(results_root.rglob("summary.json")):
        summary = read_summary(summary_path)
        rel = summary_path.relative_to(results_root)
        parts = rel.parts
        collection = parts[0] if len(parts) >= 1 else None
        path_key = parts[1] if len(parts) >= 2 else None
        step_dir = parts[2] if len(parts) >= 3 else None
        scene_label = scene_id_from_summary(summary, summary_path)
        rows.append(
            {
                "scene_label": scene_label,
                "collection": collection,
                "path_key": path_key,
                "path": infer_path_from_key(path_key) if path_key else None,
                "step_dir": step_dir,
                "step": infer_step_from_dir_name(step_dir) if step_dir else None,
                "results_dir": str(summary_path.parent),
                "summary_path": str(summary_path),
                "summary": summary,
            }
        )
    return rows


def collection_tag(collection: str | None) -> str | None:
    if not collection:
        return None
    match = re.match(r"^(IHTest_\d{6})_DistStA$", collection)
    if match:
        return match.group(1)
    return collection


def path_prefix(path_name: str | None) -> str | None:
    if not path_name:
        return None
    return path_name.replace("_DistStA", "")


def step_number(step_name: str | None) -> str | None:
    if not step_name:
        return None
    match = re.search(r"Step(\d+)", step_name)
    if not match:
        return None
    return str(int(match.group(1)))


def resolve_scene_dir(data_root: Path, collection: str | None, path_name: str | None, step_name: str | None) -> Path | None:
    if not collection or not path_name or not step_name:
        return None
    prefix = path_prefix(path_name)
    step_num = step_number(step_name)
    if not prefix or step_num is None:
        return None
    candidates = (
        data_root / collection / path_name / f"{prefix}_Step{step_num}_DistStA",
        data_root / collection / path_name / f"{prefix}_Step{step_num}",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def resolve_source_assets(data_root: Path | None, row: dict) -> dict:
    scene_dir = resolve_scene_dir(data_root, row["collection"], row["path"], row["step"]) if data_root else None
    if scene_dir is None:
        return {
            "scene_dir": None,
            "corresp_path": None,
            "cyl_path": None,
            "hdr_path": None,
            "bsq_path": None,
            "las_path": None,
        }

    tag = collection_tag(row["collection"])
    prefix = path_prefix(row["path"])
    step_num = step_number(row["step"])
    stem = f"{tag}_{prefix}_Step{step_num}"
    corresp_path = find_first_existing(
        [
            scene_dir / f"{stem}_LWHSI1_collect0_DistStA_corresp.txt",
            scene_dir / f"{stem}_LWHSI1_DistStA_corresp.txt",
        ]
    )
    cyl_path = find_first_existing(
        [
            scene_dir / f"{stem}_LWHSI1_collect0_DistStA.cyl",
            scene_dir / f"{stem}_LWHSI1_DistStA.cyl",
        ]
    )
    hdr_path = find_first_existing(
        [
            scene_dir / f"{stem}_LWHSI1_collect0_DistStA.hdr",
            scene_dir / f"{stem}_LWHSI1_DistStA.hdr",
        ]
    )
    bsq_path = hdr_path.with_suffix(".bsq") if hdr_path else None
    las_path = scene_dir / f"{stem}_HiResLIDAR_DistStA.las"
    return {
        "scene_dir": str(scene_dir),
        "corresp_path": str(corresp_path) if corresp_path else None,
        "cyl_path": str(cyl_path) if cyl_path else None,
        "hdr_path": str(hdr_path) if hdr_path else None,
        "bsq_path": str(bsq_path) if bsq_path and bsq_path.exists() else None,
        "las_path": str(las_path) if las_path.exists() else None,
    }


def required_result_presence(results_dir: Path) -> dict:
    presence = {}
    for name in REQUIRED_RESULTS_FILES:
        presence[f"has_{name.replace('.', '_')}"] = (results_dir / name).exists()
    return presence


def load_qc_reviews(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene_label = (row.get("scene_label") or "").strip()
            if not scene_label:
                continue
            grouped[scene_label].append(
                {
                    "annotator_id": (row.get("annotator_id") or "").strip(),
                    "verdict": (row.get("verdict") or "").strip().lower(),
                    "notes": (row.get("notes") or "").strip(),
                    "seconds": (row.get("seconds") or "").strip(),
                }
            )

    aggregated = {}
    for scene_label, reviews in grouped.items():
        verdicts = [r["verdict"] for r in reviews if r["verdict"]]
        counts = Counter(verdicts)
        majority_verdict = counts.most_common(1)[0][0] if counts else None
        notes = " | ".join(r["notes"] for r in reviews if r["notes"])
        seconds = []
        for review in reviews:
            try:
                seconds.append(float(review["seconds"]))
            except (TypeError, ValueError):
                continue
        aggregated[scene_label] = {
            "qc_num_reviews": len(reviews),
            "qc_majority_verdict": majority_verdict,
            "qc_vote_good": counts.get(GOOD_VERDICT, 0),
            "qc_vote_usable_with_caution": counts.get(CAUTION_VERDICT, 0),
            "qc_vote_bad": counts.get(BAD_VERDICT, 0),
            "qc_mean_seconds": sum(seconds) / len(seconds) if seconds else None,
            "qc_notes": notes or None,
            "qc_status": "complete" if len(reviews) >= 4 else "partial",
        }
    return aggregated


def load_overrides(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return {
            (row.get("scene_label") or "").strip(): row
            for row in reader
            if (row.get("scene_label") or "").strip()
        }


def choose_release_decision(
    summary_verdict: str,
    qc_majority_verdict: str | None,
    artifacts_complete: bool,
    override: dict | None,
) -> tuple[str, str | None]:
    if override and (override.get("release_decision") or "").strip():
        decision = override["release_decision"].strip().lower()
        reason = (override.get("exclusion_reason") or "").strip() or None
        return decision, reason

    effective_verdict = qc_majority_verdict or summary_verdict
    if not artifacts_complete:
        return "exclude", "missing required artifacts"
    if effective_verdict == GOOD_VERDICT:
        return "include", None
    if effective_verdict == CAUTION_VERDICT:
        return "defer", "qc verdict usable with caution"
    if effective_verdict == BAD_VERDICT:
        return "exclude", "qc verdict bad"
    return "defer", "missing final qc verdict"


def build_row(base: dict, qc: dict | None, override: dict | None, data_root: Path | None) -> dict:
    summary = base["summary"]
    results_dir = Path(base["results_dir"])
    source_assets = resolve_source_assets(data_root, base)
    result_presence = required_result_presence(results_dir)
    required_results_complete = all(result_presence.values())
    required_source_complete = all(
        source_assets.get(key)
        for key in ("corresp_path", "cyl_path", "hdr_path", "bsq_path", "las_path")
    )
    artifacts_complete = required_results_complete and required_source_complete

    qc = qc or {}
    override = override or {}
    summary_verdict = str(summary.get("verdict", PENDING_VERDICT)).strip().lower() or PENDING_VERDICT
    decision, exclusion_reason = choose_release_decision(
        summary_verdict=summary_verdict,
        qc_majority_verdict=qc.get("qc_majority_verdict"),
        artifacts_complete=artifacts_complete,
        override=override,
    )

    row = {
        "scene_label": base["scene_label"],
        "collection": base["collection"],
        "path": base["path"],
        "step": base["step"],
        "results_dir": base["results_dir"],
        "scene_dir": source_assets["scene_dir"],
        "release_decision": decision,
        "exclusion_reason": exclusion_reason,
        "qc_status": override.get("qc_status") or qc.get("qc_status") or "missing",
        "summary_verdict": summary_verdict,
        "qc_majority_verdict": qc.get("qc_majority_verdict"),
        "annotation_mode": override.get("annotation_mode") or "manual_correspondence_plus_rigid_fit",
        "cyl_source": override.get("cyl_source") or "unknown",
        "num_picked_pairs": summary.get("num_txt_points"),
        "num_manual_las_points": summary.get("num_manual_las_points"),
        "num_projected_las_points": summary.get("num_projected_las_points"),
        "fit_rmse_total_px": summary.get("fit_rmse_total"),
        "fit_rmse_u_px": summary.get("fit_rmse_u"),
        "fit_rmse_v_px": summary.get("fit_rmse_v"),
        "cyl_verify_rmse_total_px": summary.get("cyl_verify_rmse_total"),
        "baseline_rmse_total_px": summary.get("baseline_rmse_total"),
        "manual_points_pass": summary.get("manual_points_pass"),
        "cyl_verify_pass": summary.get("cyl_verify_pass"),
        "fit_pass": summary.get("fit_pass"),
        "auto_accept_pass": summary.get("auto_accept_pass"),
        "annotation_minutes": summary.get("annotation_minutes"),
        "processing_minutes": summary.get("processing_minutes"),
        "has_lwir": bool(source_assets["hdr_path"] and source_assets["bsq_path"]),
        "has_lidar": bool(source_assets["las_path"]),
        "has_correspondence_artifact": bool(source_assets["corresp_path"]),
        "has_cyl": bool(source_assets["cyl_path"]),
        "has_projected_depth_labels": result_presence["has_fitted_rigid_projection_npz"],
        "has_summary_json": result_presence["has_summary_json"],
        "has_manual_projection_plot": result_presence["has_manual_projection_comparison_png"],
        "has_cyl_verification_plot": result_presence["has_cyl_verification_overlay_png"],
        "has_manual_residual_csv": result_presence["has_manual_projection_residuals_csv"],
        "has_manual_las_points_csv": result_presence["has_manual_las_points_csv"],
        "artifacts_complete": artifacts_complete,
        "qc_num_reviews": qc.get("qc_num_reviews"),
        "qc_vote_good": qc.get("qc_vote_good"),
        "qc_vote_usable_with_caution": qc.get("qc_vote_usable_with_caution"),
        "qc_vote_bad": qc.get("qc_vote_bad"),
        "qc_mean_seconds": qc.get("qc_mean_seconds"),
        "qc_notes": override.get("qc_notes") or qc.get("qc_notes"),
        "corresp_path": source_assets["corresp_path"],
        "cyl_path": source_assets["cyl_path"],
        "hdr_path": source_assets["hdr_path"],
        "bsq_path": source_assets["bsq_path"],
        "las_path": source_assets["las_path"],
        "summary_path": base["summary_path"],
    }
    return row


def to_markdown_table(df: pd.DataFrame) -> str:
    columns = [
        "scene_label",
        "collection",
        "path",
        "step",
        "release_decision",
        "summary_verdict",
        "qc_majority_verdict",
        "num_picked_pairs",
        "fit_rmse_total_px",
        "cyl_verify_rmse_total_px",
        "artifacts_complete",
        "exclusion_reason",
    ]
    table = df[columns].copy().sort_values(["collection", "path", "step", "scene_label"])
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in table.iterrows():
        vals = []
        for col in columns:
            value = row[col]
            if pd.isna(value):
                value = ""
            vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    data_root = Path(args.data_root) if args.data_root else None
    qc_reviews = load_qc_reviews(Path(args.qc_reviews_csv)) if args.qc_reviews_csv else {}
    overrides = load_overrides(Path(args.overrides_csv)) if args.overrides_csv else {}

    rows = find_results_rows(results_root)
    manifest_rows = []
    for base in rows:
        manifest_rows.append(
            build_row(
                base=base,
                qc=qc_reviews.get(base["scene_label"]),
                override=overrides.get(base["scene_label"]),
                data_root=data_root,
            )
        )

    if not manifest_rows:
        raise SystemExit(f"No scene summaries found under {results_root}")

    df = pd.DataFrame(manifest_rows).sort_values(
        by=["collection", "path", "step", "scene_label"]
    ).reset_index(drop=True)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    if args.output_markdown:
        output_md = Path(args.output_markdown)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(to_markdown_table(df))

    counts = df["release_decision"].value_counts().to_dict()
    print(f"Saved frozen manifest to {output_csv}")
    print(f"Scene count: {len(df)}")
    print(f"Decision counts: {counts}")


if __name__ == "__main__":
    main()
