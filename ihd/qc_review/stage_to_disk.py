import argparse
import csv
import re
import shutil
from pathlib import Path

from ihd.qc_review.scene_service import (
    ANALYSIS_ROOT,
    QC_ROOT,
    build_reference_preview,
    discover_qc_scenes,
    resolve_scene_dir,
    resolve_hsi_hdr,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Populate per-scene QC PNGs into /disk scene folders using dataset-style "
            "filenames derived from the LWHSI source name."
        )
    )
    ap.add_argument(
        "--results-root",
        default=str(ANALYSIS_ROOT / "lidar_labeling"),
        help="Primary results root; annotation workspace roots are discovered next to it.",
    )
    ap.add_argument(
        "--data-root",
        default="/disk",
        help="Dataset root containing the shared scene folders.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing staged PNGs in /disk scene folders.",
    )
    ap.add_argument(
        "--manifest-out",
        default=str(QC_ROOT / "staged_to_disk_manifest.csv"),
        help="CSV manifest summarizing staged outputs.",
    )
    return ap.parse_args()


def derive_output_names(hdr_path: Path) -> tuple[str, str]:
    stem = hdr_path.stem
    if "_LWHSI1_collect" in stem:
        prefix, collect_suffix = stem.split("_LWHSI1_", 1)
        ref_name = f"{prefix}_PseudoBB_{collect_suffix}.png"
        overlay_name = f"{prefix}_DepthOverlay_{collect_suffix}.png"
        return ref_name, overlay_name
    match = re.fullmatch(r"(?P<prefix>.+)_LWHSI1_+DistStA", stem)
    if match:
        prefix = match.group("prefix")
        return (
            f"{prefix}_PseudoBB_DistStA.png",
            f"{prefix}_DepthOverlay_DistStA.png",
        )
    if "_LWHSI2_collect" in stem:
        prefix, collect_suffix = stem.split("_LWHSI2_", 1)
        ref_name = f"{prefix}_PseudoBB_{collect_suffix}.png"
        overlay_name = f"{prefix}_DepthOverlay_{collect_suffix}.png"
        return ref_name, overlay_name
    match = re.fullmatch(r"(?P<prefix>.+)_LWHSI2_+DistStA", stem)
    if match:
        prefix = match.group("prefix")
        return (
            f"{prefix}_PseudoBB_DistStA.png",
            f"{prefix}_DepthOverlay_DistStA.png",
        )
    raise ValueError(f"Could not derive QC output names from HDR filename: {hdr_path.name}")


def ensure_reference_png(scene, ref_target: Path) -> str:
    if scene.reference_png_path is not None and scene.reference_png_path.exists():
        shutil.copy2(scene.reference_png_path, ref_target)
        return str(scene.reference_png_path)
    if scene.reference_hdr_path is None:
        raise FileNotFoundError(f"Missing reference source for scene {scene.scene_label}")
    build_reference_preview(scene.reference_hdr_path, ref_target)
    return str(scene.reference_hdr_path)


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    data_root = Path(args.data_root)
    cache_root = QC_ROOT / "cache"
    scenes = discover_qc_scenes(results_root=results_root, data_root=data_root, cache_root=cache_root)

    manifest_rows: list[dict[str, str]] = []
    for scene in scenes:
        scene_dir = resolve_scene_dir(scene.collection, scene.path_key, scene.step_dir, data_root)
        if scene_dir is None:
            manifest_rows.append(
                {
                    "scene_label": scene.scene_label,
                    "collection": scene.collection,
                    "path_key": scene.path_key,
                    "step_dir": scene.step_dir,
                    "status": "skip_missing_scene_dir",
                    "reference_src": "",
                    "reference_dst": "",
                    "overlay_src": str(scene.overlay_path),
                    "overlay_dst": "",
                }
            )
            continue

        hdr_path = scene.reference_hdr_path or resolve_hsi_hdr(scene_dir, scene.collection, scene.path_key, scene.step_dir)
        if hdr_path is None:
            manifest_rows.append(
                {
                    "scene_label": scene.scene_label,
                    "collection": scene.collection,
                    "path_key": scene.path_key,
                    "step_dir": scene.step_dir,
                    "status": "skip_missing_hdr",
                    "reference_src": "",
                    "reference_dst": "",
                    "overlay_src": str(scene.overlay_path),
                    "overlay_dst": "",
                }
            )
            continue

        try:
            ref_name, overlay_name = derive_output_names(hdr_path)
        except ValueError:
            manifest_rows.append(
                {
                    "scene_label": scene.scene_label,
                    "collection": scene.collection,
                    "path_key": scene.path_key,
                    "step_dir": scene.step_dir,
                    "status": "skip_bad_name_template",
                    "reference_src": str(hdr_path),
                    "reference_dst": "",
                    "overlay_src": str(scene.overlay_path),
                    "overlay_dst": "",
                }
            )
            continue

        ref_target = scene_dir / ref_name
        overlay_target = scene_dir / overlay_name
        if not args.overwrite and ref_target.exists() and overlay_target.exists():
            manifest_rows.append(
                {
                    "scene_label": scene.scene_label,
                    "collection": scene.collection,
                    "path_key": scene.path_key,
                    "step_dir": scene.step_dir,
                    "status": "skip_exists",
                    "reference_src": str(scene.reference_png_path or hdr_path),
                    "reference_dst": str(ref_target),
                    "overlay_src": str(scene.overlay_path),
                    "overlay_dst": str(overlay_target),
                }
            )
            continue

        ref_target.parent.mkdir(parents=True, exist_ok=True)
        reference_src = ensure_reference_png(scene, ref_target)
        shutil.copy2(scene.overlay_path, overlay_target)
        manifest_rows.append(
            {
                "scene_label": scene.scene_label,
                "collection": scene.collection,
                "path_key": scene.path_key,
                "step_dir": scene.step_dir,
                "status": "staged",
                "reference_src": reference_src,
                "reference_dst": str(ref_target),
                "overlay_src": str(scene.overlay_path),
                "overlay_dst": str(overlay_target),
            }
        )

    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scene_label",
        "collection",
        "path_key",
        "step_dir",
        "status",
        "reference_src",
        "reference_dst",
        "overlay_src",
        "overlay_dst",
    ]
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    staged = sum(1 for row in manifest_rows if row["status"] == "staged")
    skipped = len(manifest_rows) - staged
    print(f"Scenes considered: {len(manifest_rows)}")
    print(f"Staged: {staged}")
    print(f"Skipped: {skipped}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
