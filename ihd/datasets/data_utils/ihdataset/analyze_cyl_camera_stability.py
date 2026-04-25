import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ihd.datasets.cylindrical_camera import read_cam


CYL_RE = re.compile(
    r"(?P<collection>IHTest_[^/]+)/"
    r"(?P<path>Path\d+_[^/]+)/"
    r"(?P<step>Path\d+_Step\d+[^/]*)/"
    r"(?P<filename>[^/]+\.cyl)$"
)


@dataclass
class CylRecord:
    path: str
    collection: str
    path_name: str
    step_name: str
    sensor: str
    collect_num: str
    rotation: np.ndarray
    translation: np.ndarray
    R: float
    w: float
    f: float
    j0: float
    y: float


def parse_sensor(filename: str) -> tuple[str, str]:
    sensor_match = re.search(r"(LWHSI\d+)", filename)
    collect_match = re.search(r"(collect\d+)", filename)
    sensor = sensor_match.group(1) if sensor_match else "unknown"
    collect_num = collect_match.group(1) if collect_match else "collect0"
    return sensor, collect_num


def parse_step_number(step_name: str) -> int:
    match = re.search(r"Step(\d+)", step_name)
    return int(match.group(1)) if match else -1


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    rel = R_a.T @ R_b
    trace = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def load_records(data_roots: list[Path], sensor_filter: str) -> list[CylRecord]:
    records: list[CylRecord] = []
    for root in data_roots:
        for cyl_path in sorted(root.rglob("*.cyl")):
            rel = str(cyl_path.relative_to(root.parent))
            match = CYL_RE.match(rel)
            if not match:
                continue
            sensor, collect_num = parse_sensor(match.group("filename"))
            if sensor != sensor_filter:
                continue
            cam = read_cam(str(cyl_path))
            records.append(
                CylRecord(
                    path=str(cyl_path),
                    collection=match.group("collection"),
                    path_name=match.group("path"),
                    step_name=match.group("step"),
                    sensor=sensor,
                    collect_num=collect_num,
                    rotation=np.asarray(cam.Rot, dtype=float),
                    translation=np.asarray(cam.t, dtype=float),
                    R=float(cam.R),
                    w=float(cam.w),
                    f=float(cam.f),
                    j0=float(cam.j0),
                    y=float(cam.y),
                )
            )
    return records


def summarize_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for group_key, group in df.groupby(group_cols, dropna=False):
        group = group.sort_values("step_num")
        first = group.iloc[0]

        rot_deltas = [rotation_angle_deg(first["rotation"], rot) for rot in group["rotation"]]
        trans_deltas = [
            float(np.linalg.norm(t - first["translation"])) for t in group["translation"]
        ]

        row = {}
        if isinstance(group_key, tuple):
            for col, value in zip(group_cols, group_key):
                row[col] = value
        else:
            row[group_cols[0]] = group_key

        row.update(
            {
                "n_files": int(len(group)),
                "first_step": first["step_name"],
                "last_step": group.iloc[-1]["step_name"],
                "rotation_delta_deg_max": float(np.max(rot_deltas)),
                "rotation_delta_deg_mean": float(np.mean(rot_deltas)),
                "translation_delta_l2_max": float(np.max(trans_deltas)),
                "translation_delta_l2_mean": float(np.mean(trans_deltas)),
            }
        )

        for name in ("R", "w", "f", "j0", "y"):
            values = group[name].to_numpy(dtype=float)
            row[f"{name}_min"] = float(np.min(values))
            row[f"{name}_max"] = float(np.max(values))
            row[f"{name}_range"] = float(np.max(values) - np.min(values))
            row[f"{name}_std"] = float(np.std(values))

        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def build_dataframe(records: list[CylRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append(
            {
                "path": r.path,
                "collection": r.collection,
                "path_name": r.path_name,
                "step_name": r.step_name,
                "step_num": parse_step_number(r.step_name),
                "sensor": r.sensor,
                "collect_num": r.collect_num,
                "rotation": r.rotation,
                "translation": r.translation,
                "tx": float(r.translation[0]),
                "ty": float(r.translation[1]),
                "tz": float(r.translation[2]),
                "R": r.R,
                "w": r.w,
                "f": r.f,
                "j0": r.j0,
                "y": r.y,
            }
        )
    return pd.DataFrame(rows)


def print_summary(title: str, summary: pd.DataFrame, max_rows: int = 12) -> None:
    print(f"\n=== {title} ===")
    if summary.empty:
        print("No data.")
        return
    cols = [
        c for c in summary.columns
        if c in {
            "collection", "path_name", "n_files", "first_step", "last_step",
            "rotation_delta_deg_max", "translation_delta_l2_max",
            "R_range", "w_range", "f_range", "j0_range", "y_range",
        }
    ]
    print(summary[cols].head(max_rows).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze how cylindrical camera parameters vary across scenes."
    )
    ap.add_argument(
        "--data-root",
        action="append",
        required=True,
        help="Dataset root to scan, e.g. /disk/IHTest_202104_DistStA",
    )
    ap.add_argument("--sensor", default="LWHSI1", help="Sensor tag to analyze.")
    ap.add_argument(
        "--out-dir",
        default="analysis/cyl_stability",
        help="Directory for CSV summaries.",
    )
    args = ap.parse_args()

    roots = [Path(p) for p in args.data_root]
    records = load_records(roots, sensor_filter=args.sensor)
    if not records:
        raise SystemExit(f"No {args.sensor} .cyl files found in the requested roots.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataframe(records).sort_values(["collection", "path_name", "step_num"]).reset_index(drop=True)
    by_path = summarize_group(df, ["collection", "path_name"])
    by_collection = summarize_group(df, ["collection"])

    raw_out = out_dir / f"{args.sensor.lower()}_raw.csv"
    path_out = out_dir / f"{args.sensor.lower()}_by_path.csv"
    collection_out = out_dir / f"{args.sensor.lower()}_by_collection.csv"
    df.drop(columns=["rotation", "translation"]).to_csv(raw_out, index=False)
    by_path.to_csv(path_out, index=False)
    by_collection.to_csv(collection_out, index=False)

    print(f"Analyzed {len(df)} {args.sensor} .cyl files.")
    print(f"Collections: {sorted(df['collection'].unique().tolist())}")
    print(f"Paths: {sorted(df['path_name'].unique().tolist())}")
    print(f"Raw table: {raw_out}")
    print(f"Path summary: {path_out}")
    print(f"Collection summary: {collection_out}")

    print_summary("By Collection", by_collection)
    print_summary(
        "By Path (sorted by max rotation delta)",
        by_path.sort_values("rotation_delta_deg_max", ascending=False),
        max_rows=20,
    )


if __name__ == "__main__":
    main()
