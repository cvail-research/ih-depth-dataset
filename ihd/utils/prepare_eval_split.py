from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path


def _safe_relpath(value: str, *, field: str) -> Path:
    if not value:
        raise ValueError(f"Missing required manifest field: {field}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Manifest field {field} must be a relative path inside the dataset root: {value}")
    return path


def _depth_relpath(row: dict[str, str]) -> Path:
    if row.get("depth_png_relpath"):
        return _safe_relpath(row["depth_png_relpath"], field="depth_png_relpath")

    raw_lwhsi_stem = row.get("raw_lwhsi_stem")
    if not raw_lwhsi_stem:
        raise ValueError("Manifest must contain depth_png_relpath or raw_lwhsi_stem.")

    if row.get("raw_scene_relpath"):
        scene_relpath = _safe_relpath(row["raw_scene_relpath"], field="raw_scene_relpath")
    else:
        missing = [field for field in ("collection", "path", "step") if not row.get(field)]
        if missing:
            raise ValueError(
                "Manifest must contain depth_png_relpath, raw_scene_relpath, "
                f"or collection/path/step. Missing: {', '.join(missing)}."
            )
        scene_relpath = Path(row["collection"]) / row["path"] / row["step"]

    return scene_relpath / f"{raw_lwhsi_stem}_depth.png"


def read_split_depth_paths(split_csv: str | Path) -> list[Path]:
    with Path(split_csv).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Split CSV is empty: {split_csv}")
    return [_depth_relpath(row) for row in rows]


def prepare_eval_split(
    raw_ih_root: str | Path,
    gt_dir: str | Path,
    split_csv: str | Path,
    *,
    symlink: bool = False,
    overwrite: bool = False,
) -> list[Path]:
    raw_root = Path(raw_ih_root)
    out_root = Path(gt_dir)
    if not raw_root.is_dir():
        raise ValueError(f"RAW_IH_ROOT does not exist or is not a directory: {raw_root}")

    relpaths = read_split_depth_paths(split_csv)
    problems: list[str] = []
    written: list[Path] = []

    for relpath in relpaths:
        src = raw_root / relpath
        dst = out_root / relpath
        if not src.is_file():
            problems.append(f"Missing depth PNG: {src}")
            continue
        if dst.exists() or dst.is_symlink():
            if not overwrite:
                problems.append(f"Destination already exists: {dst}")
                continue
            if dst.is_dir() and not dst.is_symlink():
                problems.append(f"Destination is a directory: {dst}")
                continue
            dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)
        if symlink:
            os.symlink(os.path.relpath(src.resolve(), start=dst.parent.resolve()), dst)
        else:
            shutil.copy2(src, dst)
        written.append(dst)

    if problems:
        raise ValueError("\n".join(problems))
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an IH-Depth GT_DIR from an overlaid raw IH root and a split CSV. "
            "Only benchmark depth PNGs are copied or linked; .cyl and correspondence files are not needed by evaluation."
        )
    )
    parser.add_argument("raw_ih_root", metavar="RAW_IH_ROOT")
    parser.add_argument("gt_dir", metavar="GT_DIR")
    parser.add_argument("--split_csv", required=True, help="Path to scenes_test.csv or another IH-Depth split CSV.")
    parser.add_argument("--symlink", action="store_true", help="Symlink depth PNGs instead of copying them.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing files in GT_DIR.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = prepare_eval_split(
        args.raw_ih_root,
        args.gt_dir,
        args.split_csv,
        symlink=args.symlink,
        overwrite=args.overwrite,
    )
    action = "Symlinked" if args.symlink else "Copied"
    print(f"{action} {len(written)} depth PNGs into {Path(args.gt_dir)}.")


if __name__ == "__main__":
    main()
