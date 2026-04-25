"""
Download LWIRHSI products from the DARPA Invisible Headlights S3 bucket (anonymous).

This script is intentionally robust to naming differences by:
- listing objects under the scene prefix, then
- filtering by substring (e.g. "lwhsi") and extension (e.g. ".hdr", ".bsq"),
- downloading matched keys while preserving the S3 directory structure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import boto3
from botocore import UNSIGNED
from botocore.client import Config


BUCKET = "ihdataset-01"


def build_scene_prefix(test: str, path: str, step: str) -> str:
    """
    Scene prefix used by the dataset for DistStA, e.g.
      IHTest_202104_DistStA/Path5_DistStA/Path5_Step1_DistStA/
    """
    return f"IHTest_{test}_DistStA/Path{path}_DistStA/Path{path}_Step{step}_DistStA/"


def iter_keys(s3_client, prefix: str) -> Iterable[str]:
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            yield key


def match_key(key: str, contains: List[str], exts: List[str]) -> bool:
    k = key.lower()
    if contains and not any(c.lower() in k for c in contains):
        return False
    if exts and not any(k.endswith(e.lower()) for e in exts):
        return False
    return True


def download_key(s3_client, key: str, dest_root: Path, dry_run: bool) -> None:
    out_path = dest_root / key
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"SKIP (exists): {out_path}")
        return
    print(f"GET  s3://{BUCKET}/{key} -> {out_path}", flush=True)
    if dry_run:
        return
    s3_client.download_file(Bucket=BUCKET, Key=key, Filename=str(out_path))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test", default="202104")
    p.add_argument("--path", default="5")
    p.add_argument("--step", default="1")
    p.add_argument(
        "--prefix",
        default=None,
        help="Override S3 prefix (otherwise constructed from --test/--path/--step)",
    )
    p.add_argument("--dest", type=Path, default=Path("/disk/raw"))
    p.add_argument(
        "--contains",
        nargs="*",
        default=["LWHSI"],
        help="Only download keys whose path contains ANY of these substrings (case-insensitive).",
    )
    p.add_argument(
        "--ext",
        nargs="*",
        default=[".hdr", ".bsq"],
        help="Only download keys ending with these extensions (case-insensitive).",
    )
    p.add_argument("--dry-run", action="store_true", help="List planned downloads without downloading.")
    args = p.parse_args()

    prefix = args.prefix or build_scene_prefix(args.test, args.path, args.step)
    print(f"Bucket:  {BUCKET}")
    print(f"Prefix:  {prefix}")
    print(f"Dest:    {args.dest}")
    print(f"Filter:  contains={args.contains}  ext={args.ext}")
    print(f"Dry run: {args.dry_run}")

    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    keys = [k for k in iter_keys(s3_client, prefix) if match_key(k, args.contains, args.ext)]
    if not keys:
        print("No keys matched. Try loosening filters, e.g. --contains LWHSI1 LWHSI --ext .hdr .bsq .json")
        return

    print(f"Matched {len(keys)} files:")
    for k in keys:
        print(f"  {k}")

    for k in keys:
        download_key(s3_client, k, args.dest, args.dry_run)

    print("Done.")


if __name__ == "__main__":
    main()

