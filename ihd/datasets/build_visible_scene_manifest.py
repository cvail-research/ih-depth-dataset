from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


BBVIS_FILE_RE = re.compile(
    r"IHTest_(?P<year>\d+)_Path(?P<path>\d+)_Step(?P<step>\d+)(?:_Collect\d+)?_BBVIS(?P<variant>[^.]*)_DistStA\.hdr$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build visible-scene resolution manifest for split definition v0.")
    ap.add_argument(
        "--split-manifest",
        default="manifests/07_split_definition_v0/scene_splits.csv",
        help="Input split CSV with collection/path/step columns.",
    )
    ap.add_argument(
        "--disk-root",
        default="/disk",
        help="Root directory that contains IHTest collections.",
    )
    ap.add_argument(
        "--output-csv",
        default="manifests/07_split_definition_v0/scene_splits_visible_resolution_v0.csv",
        help="Output CSV with selected visible assets and categorization placeholders.",
    )
    ap.add_argument(
        "--missing-csv",
        default="manifests/07_split_definition_v0/scene_splits_visible_missing_v0.csv",
        help="Output CSV with unresolved scenes that need BBVIS download.",
    )
    ap.add_argument(
        "--summary-json",
        default="manifests/07_split_definition_v0/scene_splits_visible_resolution_summary_v0.json",
        help="Output JSON with resolution counters.",
    )
    ap.add_argument(
        "--navigation-csv",
        default="manifests/07_split_definition_v0/scene_visible_navigation_manifest_v0.csv",
        help="Compact scene-to-visible mapping CSV for easy sharing/navigation.",
    )
    ap.add_argument(
        "--preferred-side",
        default="LEFT",
        choices=["LEFT", "RIGHT"],
        help="Preferred BBVIS side when both LEFT and RIGHT exist.",
    )
    ap.add_argument(
        "--scene-spot-mapping",
        default="manifests/05_scene_spot_mapping_v0.csv",
        help="Optional collection/path to scene_spot_id mapping for cross-path fallback.",
    )
    return ap.parse_args()


def canonical_collection_from_year(year: str) -> str:
    if year == "202204":
        return "IHTest_202204_DistStA-20221110"
    return f"IHTest_{year}_DistStA"


def parse_step_num(step: str) -> int:
    m = re.search(r"step(\d+)$", step)
    if not m:
        raise ValueError(f"Could not parse step number from: {step}")
    return int(m.group(1))


def parse_path_num(path_name: str) -> int:
    m = re.search(r"path(\d+)$", path_name)
    if not m:
        raise ValueError(f"Could not parse path number from: {path_name}")
    return int(m.group(1))


def infer_side(variant: str) -> str:
    text = variant.upper()
    if "LEFT" in text:
        return "LEFT"
    if "RIGHT" in text:
        return "RIGHT"
    return ""


def build_index(disk_root: Path) -> tuple[dict[tuple[str, str, str], list[tuple[str, Path]]], dict[tuple[str, str], set[int]]]:
    keyed_assets: dict[tuple[str, str, str], list[tuple[str, Path]]] = defaultdict(list)
    step_sets: dict[tuple[str, str], set[int]] = defaultdict(set)

    for candidate in disk_root.glob("IHTest_*_DistStA*/**/*.hdr"):
        if candidate.suffix.lower() != ".hdr":
            continue
        hdr = candidate
        m = BBVIS_FILE_RE.match(hdr.name)
        if not m:
            continue
        year = m.group("year")
        path_num = int(m.group("path"))
        step_num = int(m.group("step"))
        variant = m.group("variant") or ""
        side = infer_side(variant)

        collection = canonical_collection_from_year(year)
        path_name = f"path{path_num}"
        step_name = f"path{path_num}_step{step_num}"

        key = (collection, path_name, step_name)
        keyed_assets[key].append((side, hdr))
        step_sets[(collection, path_name)].add(step_num)

    return keyed_assets, step_sets


def choose_asset(candidates: list[tuple[str, Path]], preferred_side: str) -> tuple[str, Path] | None:
    if not candidates:
        return None
    for side, path in candidates:
        if side == preferred_side:
            return side, path
    for side, path in candidates:
        if side == "LEFT":
            return side, path
    for side, path in candidates:
        if side == "RIGHT":
            return side, path
    return sorted(candidates, key=lambda item: str(item[1]))[0]


def nearest_step(candidates: set[int], target: int) -> int | None:
    if not candidates:
        return None
    return min(candidates, key=lambda v: (abs(v - target), v))


def load_spot_map(path: Path) -> dict[tuple[str, str], str]:
    if not path.exists():
        return {}
    mapping: dict[tuple[str, str], str] = {}
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            collection = str(row.get("collection", "")).strip()
            path_name = str(row.get("path", "")).strip()
            spot_id = str(row.get("scene_spot_id", "")).strip()
            if collection and path_name and spot_id:
                mapping[(collection, path_name)] = spot_id
    return mapping


def main() -> None:
    args = parse_args()
    split_manifest = Path(args.split_manifest)
    disk_root = Path(args.disk_root)
    output_csv = Path(args.output_csv)
    missing_csv = Path(args.missing_csv)
    summary_json = Path(args.summary_json)
    navigation_csv = Path(args.navigation_csv)
    scene_spot_mapping = Path(args.scene_spot_mapping)

    if not split_manifest.exists():
        raise FileNotFoundError(split_manifest)
    if not disk_root.exists():
        raise FileNotFoundError(disk_root)

    keyed_assets, step_sets = build_index(disk_root)
    spot_map = load_spot_map(scene_spot_mapping)
    spot_assets: dict[tuple[str, str], list[tuple[str, int, str]]] = defaultdict(list)
    for collection, path_name, step_name in keyed_assets:
        spot_id = spot_map.get((collection, path_name))
        if not spot_id:
            continue
        spot_assets[(collection, spot_id)].append((path_name, parse_step_num(step_name), step_name))

    with split_manifest.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        input_columns = list(reader.fieldnames or [])

    resolved_rows: list[dict[str, str]] = []
    missing_rows: list[dict[str, str]] = []
    navigation_rows: list[dict[str, str]] = []

    counts = {
        "total_scenes": 0,
        "exact": 0,
        "fallback_nearest": 0,
        "fallback_same_spot": 0,
        "missing": 0,
    }

    for row in rows:
        counts["total_scenes"] += 1

        collection = str(row["collection"]).strip()
        path_name = str(row["path"]).strip()
        step_name = str(row["step"]).strip()
        target_step = parse_step_num(step_name)
        path_num = parse_path_num(path_name)

        key = (collection, path_name, step_name)
        selected_status = "missing"
        selected_step_name = ""
        selected_step_num = ""
        selected_side = ""
        selected_hdr = ""
        selected_raw = ""
        fallback_distance = ""

        if key in keyed_assets:
            pick = choose_asset(keyed_assets[key], args.preferred_side)
            if pick is not None:
                side, hdr_path = pick
                selected_status = "exact"
                selected_step_name = step_name
                selected_step_num = str(target_step)
                selected_side = side
                selected_hdr = str(hdr_path)
                raw_path = hdr_path.with_suffix(".raw")
                bsq_path = hdr_path.with_suffix(".bsq")
                selected_raw = str(raw_path if raw_path.exists() else bsq_path)
        else:
            nearest = nearest_step(step_sets.get((collection, path_name), set()), target_step)
            if nearest is not None:
                fallback_step_name = f"path{path_num}_step{nearest}"
                fallback_key = (collection, path_name, fallback_step_name)
                pick = choose_asset(keyed_assets.get(fallback_key, []), args.preferred_side)
                if pick is not None:
                    side, hdr_path = pick
                    selected_status = "fallback_nearest"
                    selected_step_name = fallback_step_name
                    selected_step_num = str(nearest)
                    selected_side = side
                    selected_hdr = str(hdr_path)
                    raw_path = hdr_path.with_suffix(".raw")
                    bsq_path = hdr_path.with_suffix(".bsq")
                    selected_raw = str(raw_path if raw_path.exists() else bsq_path)
                    fallback_distance = str(abs(nearest - target_step))
            if selected_status == "missing":
                spot_id = spot_map.get((collection, path_name))
                candidates = spot_assets.get((collection, spot_id), []) if spot_id else []
                candidates = [c for c in candidates if c[0] != path_name]
                if candidates:
                    chosen_path, chosen_step_num, chosen_step_name = min(
                        candidates,
                        key=lambda c: (abs(c[1] - target_step), c[0], c[1]),
                    )
                    chosen_key = (collection, chosen_path, chosen_step_name)
                    pick = choose_asset(keyed_assets.get(chosen_key, []), args.preferred_side)
                    if pick is not None:
                        side, hdr_path = pick
                        selected_status = "fallback_same_spot"
                        selected_step_name = chosen_step_name
                        selected_step_num = str(chosen_step_num)
                        selected_side = side
                        selected_hdr = str(hdr_path)
                        raw_path = hdr_path.with_suffix(".raw")
                        bsq_path = hdr_path.with_suffix(".bsq")
                        selected_raw = str(raw_path if raw_path.exists() else bsq_path)
                        fallback_distance = str(abs(chosen_step_num - target_step))

        if selected_status == "exact":
            counts["exact"] += 1
        elif selected_status == "fallback_nearest":
            counts["fallback_nearest"] += 1
        elif selected_status == "fallback_same_spot":
            counts["fallback_same_spot"] += 1
        else:
            counts["missing"] += 1
            missing_rows.append(
                {
                    "collection": collection,
                    "path": path_name,
                    "step": step_name,
                    "target_step_num": str(target_step),
                    "download_priority": "high",
                    "missing_reason": "no_exact_or_same_path_fallback_bbvis",
                    "s3_prefix_hint": f"s3://ihdataset-01/{collection}/Path{path_num}_DistStA/",
                }
            )

        out = dict(row)
        out.update(
            {
                "visible_status": selected_status,
                "visible_selected_step": selected_step_name,
                "visible_selected_step_num": selected_step_num,
                "visible_selected_side": selected_side,
                "visible_selected_hdr_path": selected_hdr,
                "visible_selected_raw_path": selected_raw,
                "visible_fallback_distance_steps": fallback_distance,
                "scene_category": "",
                "scene_category_notes": "",
            }
        )
        resolved_rows.append(out)
        navigation_rows.append(
            {
                "scene": str(row.get("scene", "")).strip(),
                "scene_id_for_split": str(row.get("scene_id_for_split", "")).strip(),
                "collection": collection,
                "path": path_name,
                "step": step_name,
                "split": str(row.get("split", "")).strip(),
                "source_hdr_path": str(row.get("hdr_path", "")).strip(),
                "visible_status": selected_status,
                "bbvis_hdr_path": selected_hdr,
                "bbvis_image_path": selected_raw,
                "bbvis_selected_side": selected_side,
                "bbvis_selected_step": selected_step_name,
                "bbvis_fallback_distance_steps": fallback_distance,
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    missing_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    navigation_csv.parent.mkdir(parents=True, exist_ok=True)

    output_columns = input_columns + [
        "visible_status",
        "visible_selected_step",
        "visible_selected_step_num",
        "visible_selected_side",
        "visible_selected_hdr_path",
        "visible_selected_raw_path",
        "visible_fallback_distance_steps",
        "scene_category",
        "scene_category_notes",
    ]

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(resolved_rows)

    with missing_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "collection",
                "path",
                "step",
                "target_step_num",
                "download_priority",
                "missing_reason",
                "s3_prefix_hint",
            ],
        )
        writer.writeheader()
        writer.writerows(missing_rows)

    with navigation_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene",
                "scene_id_for_split",
                "collection",
                "path",
                "step",
                "split",
                "source_hdr_path",
                "visible_status",
                "bbvis_hdr_path",
                "bbvis_image_path",
                "bbvis_selected_side",
                "bbvis_selected_step",
                "bbvis_fallback_distance_steps",
            ],
        )
        writer.writeheader()
        writer.writerows(navigation_rows)

    summary = {
        "version": "v0",
        "split_manifest": str(split_manifest),
        "disk_root": str(disk_root),
        "preferred_side": args.preferred_side,
        "counts": counts,
    }
    with summary_json.open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
