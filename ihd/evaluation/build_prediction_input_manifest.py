from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build an HDR/label manifest for baseline prediction jobs.")
    ap.add_argument("--scene-manifest", required=True, help="Frozen or accepted-scene CSV.")
    ap.add_argument("--depth-label-root", default="analysis/depth_labels/platform_sphere_r2p5")
    ap.add_argument("--disk-root", default="/disk")
    ap.add_argument("--out-csv", default="analysis/evaluation/baseline_prediction_inputs.csv")
    ap.add_argument("--limit", type=int)
    return ap.parse_args()


def step_number(step: str) -> int:
    match = re.search(r"step(\d+)", str(step), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse step number from {step}")
    return int(match.group(1))


def path_number(path: str) -> int:
    match = re.search(r"path(\d+)", str(path), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse path number from {path}")
    return int(match.group(1))


def collection_tag(collection: str) -> str:
    match = re.match(r"^(IHTest_\d{6})_DistStA", collection)
    if not match:
        return collection.split("_DistStA")[0]
    return match.group(1)


def find_hdr(row: pd.Series, disk_root: Path) -> str | None:
    if "hdr_path" in row and pd.notna(row["hdr_path"]) and Path(str(row["hdr_path"])).exists():
        return str(row["hdr_path"])
    if "disk_reference" in row and pd.notna(row["disk_reference"]):
        scene_dir = Path(str(row["disk_reference"])).parent
        hdrs = sorted(scene_dir.glob("*LWHSI1*.hdr"))
        if hdrs:
            collect0 = [p for p in hdrs if "collect0" in p.name]
            return str((collect0 or hdrs)[0])

    collection = str(row["collection"])
    pnum = path_number(str(row["path"]))
    snum = step_number(str(row["step"]))
    tag = collection_tag(collection)
    path_dir = disk_root / collection / f"Path{pnum}_DistStA"
    candidates = [
        path_dir / f"Path{pnum}_Step{snum}" / f"{tag}_Path{pnum}_Step{snum}_LWHSI1_DistStA.hdr",
        path_dir / f"Path{pnum}_Step{snum}_DistStA" / f"{tag}_Path{pnum}_Step{snum}_LWHSI1_collect0_DistStA.hdr",
        path_dir / f"Path{pnum}_Step{snum}_DistStA" / f"{tag}_Path{pnum}_Step{snum}_LWHSI1_DistStA.hdr",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    step_dirs = sorted(path_dir.glob(f"Path{pnum}_Step{snum}*"))
    for step_dir in step_dirs:
        hdrs = sorted(step_dir.glob("*LWHSI1*.hdr"))
        if hdrs:
            collect0 = [p for p in hdrs if "collect0" in p.name]
            return str((collect0 or hdrs)[0])
    return None


def label_path(row: pd.Series, depth_label_root: Path) -> str | None:
    if "label_path" in row and pd.notna(row["label_path"]) and Path(str(row["label_path"])).exists():
        return str(row["label_path"])
    p = depth_label_root / str(row["collection"]) / str(row["path"]) / str(row["step"]) / "projected_lidar_depth_label.npz"
    return str(p) if p.exists() else None


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.scene_manifest)
    rows = []
    for row in df.sort_values(["collection", "path", "step"]).itertuples(index=False):
        series = pd.Series(row._asdict())
        hdr = find_hdr(series, Path(args.disk_root))
        label = label_path(series, Path(args.depth_label_root))
        if not hdr or not label:
            continue
        scene = series.get("scene") or series.get("scene_id") or f"{series['collection']} / {series['path']} / {series['step']}"
        rows.append(
            {
                "scene": scene,
                "collection": series["collection"],
                "path": series["path"],
                "step": series["step"],
                "hdr_path": hdr,
                "label_path": label,
            }
        )
        if args.limit and len(rows) >= args.limit:
            break
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {len(rows)} prediction inputs to {out}")


if __name__ == "__main__":
    main()

