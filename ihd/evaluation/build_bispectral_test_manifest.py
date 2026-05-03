from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build a small scene manifest for a reproducible bispectral benchmark test."
    )
    ap.add_argument("--frozen-manifest", default="manifests/06_frozen_manifest_v0.csv")
    ap.add_argument("--split-manifest", default="manifests/07_split_definition_v0/scene_splits.csv")
    ap.add_argument("--out-csv", default="analysis/evaluation/manifests/bispectral_lwhsi1_test_manifest.csv")
    ap.add_argument("--sensor-id", default="LWHSI1", choices=["LWHSI1", "LWHSI2"])
    ap.add_argument("--release-decision", default="include")
    ap.add_argument("--split", default=None, help="Optional split filter: train, val, or test.")
    ap.add_argument(
        "--scene",
        default=None,
        help="Optional exact scene string, e.g. 'IHTest_202009_DistStA / path1 / path1_step2'.",
    )
    ap.add_argument("--limit", type=int, default=1)
    return ap.parse_args()


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def _key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("collection", ""), row.get("path", ""), row.get("step", ""))


def main() -> None:
    args = parse_args()

    frozen_rows = _read_csv(args.frozen_manifest)
    split_rows = _read_csv(args.split_manifest)

    frozen_by_key = {_key(row): row for row in frozen_rows}

    selected: list[dict[str, str]] = []
    for split_row in sorted(split_rows, key=lambda r: (r.get("collection", ""), r.get("path", ""), r.get("step", ""))):
        if args.release_decision and split_row.get("release_decision") != args.release_decision:
            continue
        if args.split and split_row.get("split") != args.split:
            continue
        if args.scene and split_row.get("scene") != args.scene:
            continue

        frozen_row = frozen_by_key.get(_key(split_row))
        if frozen_row is None:
            continue

        if frozen_row.get("sensor_id") != args.sensor_id:
            continue

        hdr_path = frozen_row.get("hdr_path", "")
        label_path = frozen_row.get("projected_depth_label_path_current", "")
        if not hdr_path or not label_path:
            continue

        selected.append(
            {
                "scene": split_row.get("scene", ""),
                "collection": split_row.get("collection", ""),
                "path": split_row.get("path", ""),
                "step": split_row.get("step", ""),
                "split": split_row.get("split", ""),
                "release_decision": split_row.get("release_decision", ""),
                "sensor_id": frozen_row.get("sensor_id", ""),
                "sensor_num_bands": frozen_row.get("sensor_num_bands", ""),
                "hdr_path": hdr_path,
                "label_path": label_path,
                "benchmark_cyl_path": frozen_row.get("benchmark_cyl_path", ""),
                "benchmark_corresp_path": frozen_row.get("benchmark_corresp_path", ""),
            }
        )
        if args.limit and len(selected) >= args.limit:
            break

    if not selected:
        raise SystemExit("No scenes matched the requested bispectral test filters.")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(selected[0].keys()))
        writer.writeheader()
        writer.writerows(selected)

    print(f"Saved {len(selected)} scene(s) to {out_path}")
    for row in selected:
        print(
            f"- {row['scene']} | split={row['split']} | sensor={row['sensor_id']} "
            f"| hdr={row['hdr_path']}"
        )


if __name__ == "__main__":
    main()
