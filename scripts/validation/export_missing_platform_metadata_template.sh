#!/bin/bash
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"

SRC="${1:-analysis/lidar_preprocessing/platform_sphere_r4p0_missing_metadata_paths.csv}"
OUT="${2:-analysis/lidar_preprocessing/platform_sphere_r4p0_missing_metadata_template.csv}"

uv run python - <<'PY' "${SRC}" "${OUT}"
import csv
import sys
from pathlib import Path

src = Path(sys.argv[1])
out = Path(sys.argv[2])

if not src.exists():
    raise SystemExit(f"Missing input CSV: {src}")

with src.open("r", newline="") as f:
    rows = list(csv.DictReader(f))

fieldnames = [
    "collection",
    "scope",
    "path_name",
    "platform_center_x",
    "platform_center_y",
    "platform_center_z",
    "platform_sphere_radius_m",
    "projection_voxel_m",
    "sor_k",
    "sor_std_ratio",
    "method",
    "representative_step",
    "notes",
]

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "collection": row["collection"],
                "scope": "path",
                "path_name": row["path_name"],
                "platform_center_x": "",
                "platform_center_y": "",
                "platform_center_z": "",
                "platform_sphere_radius_m": "4.0",
                "projection_voxel_m": "0.03",
                "sor_k": "50",
                "sor_std_ratio": "2.0",
                "method": "manual",
                "representative_step": row["representative_step"],
                "notes": "",
            }
        )

print(f"Wrote {len(rows)} template rows to {out}")
PY
