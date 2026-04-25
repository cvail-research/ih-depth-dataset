import argparse
import os
import subprocess
from pathlib import Path

import pandas as pd


RELEVANT_SUFFIXES = (".bsq", ".hdr", ".txt", ".cyl", ".las")


def is_relevant(filename: str) -> bool:
    if not filename.endswith(RELEVANT_SUFFIXES):
        return False
    return ("LWHSI1" in filename) or ("HiResLIDAR" in filename)


def load_desired_rows(manifest_path: Path, collection: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    df = df[df["collect"] == collection].copy()
    df = df[df["filename"].fillna("").map(is_relevant)].copy()
    return df


def ensure_download(row: pd.Series, dst_root: Path, bucket: str, dry_run: bool) -> tuple[bool, bool]:
    rel_path = Path(row["s3_key"])
    dst_path = dst_root / rel_path
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    expected_size = int(row["size_bytes"])
    if dst_path.is_file() and dst_path.stat().st_size == expected_size:
        return False, True

    src = f"s3://{bucket}/{row['s3_key']}"
    cmd = ["aws", "s3", "cp", "--no-sign-request", "--no-progress", src, str(dst_path)]
    if dry_run:
        print("DRYRUN download:", " ".join(cmd))
    else:
        subprocess.run(cmd, check=True)
    return True, False


def prune_extra_files(collection_root: Path, desired_paths: set[Path], dry_run: bool) -> int:
    removed = 0
    if not collection_root.exists():
        return removed

    for path in sorted(collection_root.rglob("*")):
        if not path.is_file():
            continue
        if not is_relevant(path.name):
            continue
        if path not in desired_paths:
            removed += 1
            if dry_run:
                print(f"DRYRUN remove: {path}")
            else:
                path.unlink()

    # Clean empty directories deepest-first.
    for path in sorted(collection_root.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass
    return removed


def main() -> None:
    ap = argparse.ArgumentParser(description="Prune and materialize one collection from the comprehensive IH manifest.")
    ap.add_argument("--manifest", required=True, help="Path to ihdataset_comprehensive.csv")
    ap.add_argument("--collection", required=True, help="Collection name, e.g. IHTest_202104_DistStA")
    ap.add_argument("--disk-root", default="/disk", help="Root directory that stores collections")
    ap.add_argument("--bucket", default="ihdataset-01", help="S3 bucket name")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without changing files")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    disk_root = Path(args.disk_root)
    collection_root = disk_root / args.collection

    desired = load_desired_rows(manifest_path, args.collection)
    desired_paths = {disk_root / Path(s3_key) for s3_key in desired["s3_key"].tolist()}

    print(f"Collection: {args.collection}")
    print(f"Manifest: {manifest_path}")
    print(f"Collection root: {collection_root}")
    print(f"Desired files: {len(desired_paths)}")

    removed = prune_extra_files(collection_root, desired_paths, dry_run=args.dry_run)
    print(f"Pruned extra relevant files: {removed}")

    downloaded = 0
    skipped = 0
    for _, row in desired.iterrows():
        did_download, already_ok = ensure_download(row, disk_root, args.bucket, dry_run=args.dry_run)
        downloaded += int(did_download)
        skipped += int(already_ok)

    print(f"Downloaded files: {downloaded}")
    print(f"Already present with expected size: {skipped}")


if __name__ == "__main__":
    main()
