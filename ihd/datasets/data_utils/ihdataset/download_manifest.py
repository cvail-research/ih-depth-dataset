import argparse
from pathlib import Path

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.client import Config


def download_from_manifest(args):
    """
    Reads a manifest CSV and downloads the specified files from S3.
    """
    try:
        manifest_df = pd.read_csv(args.manifest)
    except FileNotFoundError:
        print(f"❌ Error: Manifest file not found at '{args.manifest}'")
        return

    # Initialize the S3 client
    # Ensure your environment is configured (e.g., via `aws configure`)
    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    
    download_path = Path(args.output_path)
    download_path.mkdir(exist_ok=True)

    print(f"🚀 Starting download from bucket 'ihdataset-01'...")
    print(f"   Manifest: {args.manifest}")
    print(f"   Destination: {args.output_path}")

    # Iterate over each file in the manifest
    for index, row in manifest_df.iterrows():
        s3_key = row["s3_key"]
        local_file_path = download_path / s3_key

        # Create the local parent directory if it doesn't exist
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading: {s3_key}")
        try:
            s3_client.download_file(
                Bucket='ihdataset-01',
                Key=s3_key,
                Filename=str(local_file_path)
            )
        except Exception as e:
            print(f"   ⚠️  Could not download {s3_key}. Error: {e}")

    print(f"\n✅ Download complete. Files are located in the '{args.output_path}' directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the ihdataset from a manifest csv using the s3_key to download each file."
    )
    parser.add_argument("--manifest", type=str, default="data/ihdataset_manifest.csv", help="Path to the main input manifest CSV.")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to download all dataset files.")

    download_from_manifest(parser.parse_args())
