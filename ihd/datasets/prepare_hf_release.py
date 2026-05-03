from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare a Hugging Face dataset release package.")
    ap.add_argument(
        "--frozen-manifest",
        default="manifests/06_frozen_manifest_v0.csv",
        help="Frozen release manifest CSV.",
    )
    ap.add_argument(
        "--output-dir",
        default="analysis/huggingface_release",
        help="Directory to write Hugging Face release artifacts into.",
    )
    ap.add_argument(
        "--repo-id",
        default="cvail-research/ih-depth-dataset",
        help="Target Hugging Face dataset repo id for the draft card.",
    )
    return ap.parse_args()


def write_readme(path: Path, summary: dict, repo_id: str) -> None:
    include_count = summary["include_count"]
    defer_count = summary["defer_count"]
    exclude_count = summary["exclude_count"]
    scene_count = summary["scene_count"]
    text = f"""---
pretty_name: "IH-Depth"
license: "cc-by-4.0"
task_categories:
  - depth-estimation
tags:
  - mlcroissant
  - croissant
  - lwir
  - lidar
  - depth-estimation
  - dataset
size_categories:
  - n<1K
---

# IH-Depth

Curated LWIR-LiDAR dataset expansion for supervised thermal depth estimation on DARPA Invisible Headlights scenes.

## Dataset Summary

- Repository: `{repo_id}`
- Frozen release version: `v0`
- Total scenes in frozen manifest: `{scene_count}`
- Included scenes: `{include_count}`
- Deferred scenes: `{defer_count}`
- Excluded scenes: `{exclude_count}`

## Contents

This release package is prepared from the frozen manifest and is intended to be uploaded to the Hugging Face Hub as a dataset repository. The Hub dataset viewer can then expose Croissant metadata for the uploaded dataset where supported.

## Responsible Use

This dataset is derived from DARPA Invisible Headlights scenes and includes only scenes that passed the current release gate. The manifest retains deferred and excluded scenes for auditability.
"""
    path.write_text(text)


def infer_croissant_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "sc:Boolean"
    if pd.api.types.is_integer_dtype(series):
        return "sc:Integer"
    if pd.api.types.is_float_dtype(series):
        return "sc:Float"
    return "sc:Text"


def write_croissant(path: Path, df: pd.DataFrame, repo_id: str) -> None:
    file_object_id = "frozen-manifest-csv"
    record_set_id = "frozen-manifest"
    columns = [
        "collection",
        "path",
        "step",
        "scene",
        "release_decision",
        "release_reason",
        "verdict",
        "annotation_status",
        "fit_rmse_total_px",
        "distance_max_error_percent_of_picked_range",
        "distance_all_points_pass_le_5pct",
        "candidate_rmse5_distance5_current",
        "cleanup_status",
        "cleanup_region_count",
        "removed_points",
        "kept_points",
    ]
    field_nodes = []
    for col in columns:
        if col not in df.columns:
            continue
        field_nodes.append(
            {
                "@type": "cr:Field",
                "@id": f"{record_set_id}/{col}",
                "name": col,
                "dataType": infer_croissant_type(df[col]),
                "source": {
                    "fileObject": {"@id": file_object_id},
                    "extract": {"column": col},
                },
            }
        )

    croissant = {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "sc": "https://schema.org/",
            "dct": "http://purl.org/dc/terms/",
            "data": {"@id": "cr:data", "@type": "@json"},
            "distribution": "sc:distribution",
        },
        "@type": "sc:Dataset",
        "@id": repo_id,
        "name": "IH-Depth",
        "description": (
            "Curated LWIR-LiDAR dataset expansion for supervised thermal depth "
            "estimation on DARPA Invisible Headlights scenes."
        ),
        "url": f"https://huggingface.co/datasets/{repo_id}",
        "distribution": [
            {
                "@type": "cr:FileObject",
                "@id": file_object_id,
                "name": "06_frozen_manifest_v0.csv",
                "contentUrl": "06_frozen_manifest_v0.csv",
                "encodingFormat": "text/csv",
            }
        ],
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": record_set_id,
                "name": "frozen_manifest",
                "description": "Frozen IH-Depth v0 release manifest.",
                "key": [
                    {"@id": f"{record_set_id}/collection"},
                    {"@id": f"{record_set_id}/path"},
                    {"@id": f"{record_set_id}/step"},
                ],
                "field": field_nodes,
            }
        ],
    }
    path.write_text(json.dumps(croissant, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    frozen_path = Path(args.frozen_manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(frozen_path)
    summary = {
        "frozen_manifest": str(frozen_path),
        "scene_count": int(len(df)),
        "include_count": int((df["release_decision"] == "include").sum()),
        "defer_count": int((df["release_decision"] == "defer").sum()),
        "exclude_count": int((df["release_decision"] == "exclude").sum()),
        "candidate_count": int((df["candidate_rmse5_distance5_current"] == True).sum()),
        "cleanup_reviewed_count": int((df["has_cleanup_review"] == True).sum()),
    }

    # Keep the canonical frozen manifest as the upload artifact so the Hub copy
    # points directly at the release source of truth.
    legacy_hf_manifest = output_dir / "ih_depth_frozen_v0_hf_manifest.csv"
    if legacy_hf_manifest.exists():
        legacy_hf_manifest.unlink()
    shutil.copyfile(frozen_path, output_dir / "06_frozen_manifest_v0.csv")
    legacy_croissant = output_dir / "croissant_v0.jsonld"
    if legacy_croissant.exists():
        legacy_croissant.unlink()
    write_croissant(output_dir / "croissant_v0.json", df, args.repo_id)

    (output_dir / "release_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_readme(output_dir / "README.md", summary, args.repo_id)

    print(f"Wrote Hugging Face release package to {output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
