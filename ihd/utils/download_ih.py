from __future__ import annotations

import argparse
import csv
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError


DEFAULT_BUCKET = "ihdataset-01"
DEFAULT_REGION = "us-east-2"


def _safe_relpath(value: str, *, field: str) -> Path:
    if not value:
        raise ValueError(f"Missing required manifest field: {field}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Manifest field {field} must be a relative path inside the dataset root: {value}")
    return path


def _scene_relpath(row: dict[str, str]) -> Path:
    if row.get("raw_scene_relpath"):
        return _safe_relpath(row["raw_scene_relpath"], field="raw_scene_relpath")

    missing = [field for field in ("collection", "path", "step") if not row.get(field)]
    if missing:
        raise ValueError(
            "Manifest must contain raw_scene_relpath or collection/path/step. "
            f"Missing: {', '.join(missing)}."
        )
    return Path(row["collection"]) / row["path"] / row["step"]


def _raw_file_relpaths(row: dict[str, str]) -> tuple[Path, Path]:
    raw_lwhsi_stem = row.get("raw_lwhsi_stem")
    if not raw_lwhsi_stem:
        raise ValueError("Manifest must contain raw_lwhsi_stem.")
    scene_relpath = _scene_relpath(row)
    return (
        scene_relpath / f"{raw_lwhsi_stem}.hdr",
        scene_relpath / f"{raw_lwhsi_stem}.bsq",
    )


def read_raw_relpaths(manifest_csv: str | Path) -> list[Path]:
    with Path(manifest_csv).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest CSV is empty: {manifest_csv}")

    relpaths: list[Path] = []
    seen: set[str] = set()
    for row in rows:
        for relpath in _raw_file_relpaths(row):
            key = relpath.as_posix()
            if key in seen:
                continue
            seen.add(key)
            relpaths.append(relpath)
    return relpaths


def _s3_key(relpath: Path, prefix: str) -> str:
    base = relpath.as_posix()
    if not prefix:
        return base
    return f"{prefix.strip('/')}/{base}"


def download_ih(
    output_root: str | Path,
    manifest_csv: str | Path,
    *,
    bucket: str = DEFAULT_BUCKET,
    region: str = DEFAULT_REGION,
    prefix: str = "",
    overwrite: bool = False,
) -> tuple[list[Path], list[Path]]:
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    relpaths = read_raw_relpaths(manifest_csv)
    client = boto3.client("s3", region_name=region, config=Config(signature_version=UNSIGNED))

    downloaded: list[Path] = []
    skipped: list[Path] = []
    problems: list[str] = []

    for relpath in relpaths:
        dst = out_root / relpath
        if dst.exists():
            if not overwrite:
                skipped.append(dst)
                continue
            if dst.is_dir():
                problems.append(f"Destination is a directory: {dst}")
                continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            client.download_file(bucket, _s3_key(relpath, prefix), str(dst))
        except (BotoCoreError, ClientError) as exc:
            problems.append(f"Failed to download s3://{bucket}/{_s3_key(relpath, prefix)} -> {dst}: {exc}")
            continue
        downloaded.append(dst)

    if problems:
        raise ValueError("\n".join(problems))
    return downloaded, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download only the raw IH .hdr/.bsq files referenced by an IH-Depth manifest. "
            "The original IH folder structure and filenames are preserved."
        )
    )
    parser.add_argument("output_root", metavar="RAW_IH_ROOT", help="Destination root for the raw IH scene files.")
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to scenes_manifest.csv, scenes_train.csv, or scenes_test.csv from the IH-Depth release.",
    )
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help=f"S3 bucket name. Default: {DEFAULT_BUCKET}.")
    parser.add_argument("--region", default=DEFAULT_REGION, help=f"AWS region. Default: {DEFAULT_REGION}.")
    parser.add_argument("--prefix", default="", help="Optional S3 key prefix inside the bucket.")
    parser.add_argument("--overwrite", action="store_true", help="Re-download files that already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    downloaded, skipped = download_ih(
        args.output_root,
        args.manifest,
        bucket=args.bucket,
        region=args.region,
        prefix=args.prefix,
        overwrite=args.overwrite,
    )
    print(f"Downloaded {len(downloaded)} raw IH files into {Path(args.output_root)}.")
    if skipped:
        print(f"Skipped {len(skipped)} existing files.")


if __name__ == "__main__":
    main()
