import argparse
import csv
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.client import Config


RELEVANT_SUFFIXES = {".bsq", ".hdr", ".txt", ".cyl", ".las"}


def is_relevant_key(key: str) -> bool:
    lower = key.lower()
    if not any(lower.endswith(suffix) for suffix in RELEVANT_SUFFIXES):
        return False
    return ("lwhsi1" in lower) or ("hireslidar" in lower)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a current S3 manifest for relevant IH LWHSI1/HiResLIDAR files.")
    ap.add_argument("--bucket", default="ihdataset-01")
    ap.add_argument("--prefix", default="", help="Optional S3 prefix filter, e.g. IHTest_202104_DistStA/")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")

    rows: list[dict[str, str | int]] = []
    for page in paginator.paginate(Bucket=args.bucket, Prefix=args.prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            if not is_relevant_key(key):
                continue

            parts = key.split("/")
            collect = parts[0] if len(parts) > 0 else ""
            path = parts[1] if len(parts) > 1 else ""
            step = parts[2] if len(parts) > 2 else ""
            filename = parts[-1]
            rows.append(
                {
                    "s3_key": key,
                    "collect": collect,
                    "path": path,
                    "step": step,
                    "filename": filename,
                    "size_bytes": int(obj["Size"]),
                    "last_modified": obj["LastModified"].isoformat(),
                    "category": "raw",
                }
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "s3_key",
                "collect",
                "path",
                "step",
                "filename",
                "size_bytes",
                "last_modified",
                "category",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} manifest rows to {out_path}")


if __name__ == "__main__":
    main()
