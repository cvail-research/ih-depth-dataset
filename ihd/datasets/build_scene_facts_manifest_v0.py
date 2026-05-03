from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import spectral as spy


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build the factual 306-row scene manifest for IH-Depth.")
    ap.add_argument(
        "--with-prior-manifest",
        default="manifests/archive/legacy_pool_manifests_v0/01_with_prior_cyl_n60.csv",
        help="Scene roster for the legacy with-prior-.cyl pool.",
    )
    ap.add_argument(
        "--without-prior-manifest",
        default="manifests/archive/legacy_pool_manifests_v0/02_without_prior_cyl_n246.csv",
        help="Scene roster for the without-prior-.cyl pool.",
    )
    ap.add_argument(
        "--fitted-manifest",
        default="manifests/archive/legacy_pool_manifests_v0/03_unified_own_fitted_cyl_n232.csv",
        help="Manifest of scenes with our fitted cylindrical camera outputs.",
    )
    ap.add_argument(
        "--output-csv",
        default="manifests/01_scene_facts_n306.csv",
        help="Output scene-facts CSV.",
    )
    ap.add_argument(
        "--output-summary-json",
        default="manifests/01_scene_facts_n306_summary.json",
        help="Output summary JSON.",
    )
    return ap.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def scene_join_key(collection: str, path: str, step: str) -> str:
    return f"{collection}|{path}|{step}"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def infer_sensor_metadata(hdr_path: str | None) -> tuple[str | None, int | None]:
    if hdr_path is None or str(hdr_path).strip() == "":
        return None, None

    path = Path(hdr_path)
    name = path.name.upper()
    sensor_id: str | None = None
    if "LWHSI1" in name:
        sensor_id = "LWHSI1"
    elif "LWHSI2" in name:
        sensor_id = "LWHSI2"

    sensor_num_bands: int | None = None
    if path.exists():
        try:
            img = spy.open_image(str(path))
            sensor_num_bands = int(len(img.metadata.get("wavelength", [])))
            if sensor_id is None:
                if sensor_num_bands == 256:
                    sensor_id = "LWHSI1"
                elif sensor_num_bands == 250:
                    sensor_id = "LWHSI2"
        except Exception:
            pass

    return sensor_id, sensor_num_bands


def unique_scene_file(scene_dir: str | None, pattern: str) -> str | None:
    if not scene_dir:
        return None
    root = Path(scene_dir)
    if not root.exists():
        return None
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        return None
    return str(matches[0])


def fallback_scene_dir(scene_dir: str | None, hdr_path: str | None, las_path: str | None) -> str | None:
    if scene_dir:
        return scene_dir
    for candidate in [hdr_path, las_path]:
        if isinstance(candidate, str) and candidate.strip():
            return str(Path(candidate).parent)
    return None


def resolve_workspace_scene_paths(collection: str, path: str, step: str) -> tuple[dict[str, Any], dict[str, Any]]:
    nocyl_scene = Path("analysis/annotation_workspace_nocyl") / collection / path / step / "scene.json"
    legacy_scene = Path("analysis/annotation_workspace") / collection / path / step / "scene.json"
    return load_json(nocyl_scene) or {}, load_json(legacy_scene) or {}


def resolve_fit_outputs(row: pd.Series, collection: str, path: str, step: str) -> tuple[str | None, str | None, float | None, int | None]:
    fit_json = row.get("fit_json")
    if pd.isna(fit_json) or not str(fit_json).strip():
        candidate = Path("analysis/annotation_workspace_nocyl") / collection / path / step / "fit.json"
        if candidate.exists():
            fit_json = str(candidate)
    if pd.isna(fit_json) or not str(fit_json).strip():
        return None, None, row.get("fit_rmse_total_px"), row.get("num_picked_pairs")

    fit_json_str = str(fit_json)
    fitted_cyl = None
    fit_rmse_total_px = row.get("fit_rmse_total_px")
    num_picked_pairs = row.get("num_picked_pairs")
    fit_json_path = Path(fit_json_str)
    if fit_json_path.exists():
        candidate = fit_json_path.with_name("fitted.cyl")
        if candidate.exists():
            fitted_cyl = str(candidate)
        data = load_json(fit_json_path)
        if data:
            if fitted_cyl is None and isinstance(data.get("fitted_cyl"), str) and str(data["fitted_cyl"]).strip():
                fitted_cyl = str(data["fitted_cyl"])
            if pd.isna(fit_rmse_total_px) and data.get("fit_rmse_total") is not None:
                fit_rmse_total_px = float(data["fit_rmse_total"])
            if pd.isna(num_picked_pairs) and isinstance(data.get("picked_generated_points"), list):
                num_picked_pairs = len(data["picked_generated_points"])
    return fit_json_str, fitted_cyl, fit_rmse_total_px, num_picked_pairs


def workspace_artifact_path(root: str, collection: str, path: str, step: str, name: str) -> str | None:
    candidate = Path(root) / collection / path / step / name
    if candidate.exists():
        return str(candidate)
    return None


def build_scene_facts(with_prior: pd.DataFrame, without_prior: pd.DataFrame, fitted: pd.DataFrame) -> pd.DataFrame:
    roster = pd.concat([with_prior, without_prior], ignore_index=True)
    roster = roster.rename(columns={"pool": "source_pool"})
    roster["join_key"] = roster.apply(
        lambda row: scene_join_key(str(row["collection"]).strip(), str(row["path"]).strip(), str(row["step"]).strip()),
        axis=1,
    )
    if roster["join_key"].duplicated().any():
        dupes = roster.loc[roster["join_key"].duplicated(), "join_key"].tolist()
        raise ValueError(f"Duplicate scenes in roster manifests: {dupes[:5]}")

    fitted = fitted.copy()
    fitted["join_key"] = fitted.apply(
        lambda row: scene_join_key(str(row["collection"]).strip(), str(row["path"]).strip(), str(row["step"]).strip()),
        axis=1,
    )

    merged = roster.merge(
        fitted[
            [
                "join_key",
                "source_pool",
                "fit_rmse_total_px",
                "num_picked_pairs",
                "fitted_cyl",
                "fit_json",
            ]
        ],
        on="join_key",
        how="left",
        suffixes=("", "_fitted"),
    )

    rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        series = pd.Series(row._asdict())
        nocyl_scene, legacy_scene = resolve_workspace_scene_paths(series["collection"], series["path"], series["step"])
        nocyl_source = nocyl_scene.get("source_paths", {}) if isinstance(nocyl_scene, dict) else {}
        legacy_source = legacy_scene.get("source_paths", {}) if isinstance(legacy_scene, dict) else {}

        scene_dir = nocyl_source.get("scene_dir") or legacy_source.get("scene_dir")
        hdr_path = nocyl_source.get("hsi_hdr") or legacy_source.get("hsi_hdr")
        bsq_path = None
        if isinstance(hdr_path, str) and hdr_path.strip():
            candidate = Path(hdr_path).with_suffix(".bsq")
            if candidate.exists():
                bsq_path = str(candidate)

        las_path = nocyl_source.get("raw_las") or legacy_source.get("raw_las")
        scene_dir = fallback_scene_dir(scene_dir, hdr_path, las_path)
        if las_path is None:
            las_path = unique_scene_file(scene_dir, "*HiResLIDAR*.las")
        original_cyl_path = legacy_source.get("cyl")
        if original_cyl_path is None and str(series["source_pool"]).strip().lower() == "with_prior_cyl":
            original_cyl_path = unique_scene_file(scene_dir, "*.cyl")
        original_corresp_path = legacy_source.get("corresp_txt")
        if original_corresp_path is None and str(series["source_pool"]).strip().lower() == "with_prior_cyl":
            original_corresp_path = unique_scene_file(scene_dir, "*_corresp.txt")

        manual_picks_path = workspace_artifact_path(
            "analysis/annotation_workspace_nocyl", series["collection"], series["path"], series["step"], "picks.json"
        )
        benchmark_corresp_path = workspace_artifact_path(
            "analysis/annotation_workspace_nocyl", series["collection"], series["path"], series["step"], "generated_corresp.txt"
        )

        fit_json, fitted_cyl, fit_rmse_total_px, num_picked_pairs = resolve_fit_outputs(
            series, series["collection"], series["path"], series["step"]
        )
        if fitted_cyl is None:
            fitted_cyl = series.get("fitted_cyl") or nocyl_source.get("fitted_cyl")

        session_json = None
        nocyl_session = Path("analysis/annotation_workspace_nocyl") / series["collection"] / series["path"] / series["step"] / "session.json"
        if nocyl_session.exists():
            session_json = str(nocyl_session)
        has_fit_json = bool(isinstance(fit_json, str) and fit_json.strip())
        has_session_json = bool(session_json)

        sensor_id, sensor_num_bands = infer_sensor_metadata(hdr_path)

        benchmark_cyl_path = fitted_cyl

        rows.append(
            {
                "scene_id": series["scene_id"],
                "scene": None,
                "collection": series["collection"],
                "path": series["path"],
                "step": series["step"],
                "path_name": series["path_name"],
                "step_name": series["step_name"],
                "source_pool": series["source_pool"],
                "scene_dir": scene_dir,
                "hdr_path": hdr_path,
                "bsq_path": bsq_path,
                "original_cyl_path": original_cyl_path,
                "original_corresp_path": original_corresp_path,
                "manual_picks_path": manual_picks_path,
                "benchmark_corresp_path": benchmark_corresp_path,
                "benchmark_cyl_path": benchmark_cyl_path,
                "las_path": las_path,
                "sensor_id": sensor_id,
                "sensor_num_bands": sensor_num_bands,
                "fit_rmse_total_px": fit_rmse_total_px,
                "num_picked_pairs": num_picked_pairs,
                "fit_json": fit_json,
                "session_json": session_json,
                "has_fit_json": has_fit_json,
                "has_session_json": has_session_json,
                "join_key": series["join_key"],
            }
        )

    facts = pd.DataFrame(rows)
    facts["scene"] = facts["scene"].fillna(
        facts.apply(lambda row: f"{row['collection']} / {row['path']} / {row['step']}", axis=1)
    )
    return facts.sort_values(["collection", "path", "step"]).reset_index(drop=True)


def build_summary(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "scene_count": int(len(df)),
        "source_pool_counts": df["source_pool"].fillna("unknown").value_counts(dropna=False).to_dict(),
        "hdr_path_nonempty_count": int(df["hdr_path"].fillna("").astype(str).str.len().gt(0).sum()),
        "bsq_path_nonempty_count": int(df["bsq_path"].fillna("").astype(str).str.len().gt(0).sum()),
        "original_cyl_path_nonempty_count": int(df["original_cyl_path"].fillna("").astype(str).str.len().gt(0).sum()),
        "benchmark_cyl_path_nonempty_count": int(df["benchmark_cyl_path"].fillna("").astype(str).str.len().gt(0).sum()),
        "las_path_nonempty_count": int(df["las_path"].fillna("").astype(str).str.len().gt(0).sum()),
        "fit_json_nonempty_count": int(df["fit_json"].fillna("").astype(str).str.len().gt(0).sum()),
        "manual_picks_path_nonempty_count": int(df["manual_picks_path"].fillna("").astype(str).str.len().gt(0).sum()),
        "benchmark_corresp_path_nonempty_count": int(df["benchmark_corresp_path"].fillna("").astype(str).str.len().gt(0).sum()),
        "sensor_id_counts": df["sensor_id"].fillna("unknown").value_counts(dropna=False).to_dict(),
    }


def main() -> None:
    args = parse_args()
    with_prior = load_csv(Path(args.with_prior_manifest))
    without_prior = load_csv(Path(args.without_prior_manifest))
    fitted = load_csv(Path(args.fitted_manifest))

    facts = build_scene_facts(with_prior, without_prior, fitted)
    if len(facts) != 306:
        raise ValueError(f"Expected 306 rows in scene facts manifest, found {len(facts)}")

    output_csv = Path(args.output_csv)
    output_summary_json = Path(args.output_summary_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    facts.to_csv(output_csv, index=False)
    output_summary_json.write_text(json.dumps(build_summary(facts), indent=2, sort_keys=True) + "\n")

    print(f"Saved scene facts manifest to {output_csv}")
    print(f"Saved summary to {output_summary_json}")
    print(f"Scene count: {len(facts)}")


if __name__ == "__main__":
    main()
