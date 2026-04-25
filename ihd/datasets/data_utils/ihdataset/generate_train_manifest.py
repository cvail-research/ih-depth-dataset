import argparse
import glob
import os


def create_manifest(data_root, out_txt):
    pattern = os.path.join(data_root, "*.hdr")
    hdr_files = sorted(glob.glob(pattern))

    names = [os.path.basename(f) for f in hdr_files]

    with open(out_txt, "w") as f:
        for n in names:
            f.write(n + "\n")

    print(f"Found {len(names)} HDR files.")
    print(f"Manifest saved to: {out_txt}")


def main():
    parser = argparse.ArgumentParser(
        description="Create training manifest from flattened hyperspectral dataset."
    )

    parser.add_argument(
        "--data-root",
        required=True,
        help="Path to the flat folder containing HDR/BSQ files."
    )

    parser.add_argument(
        "--out",
        default="manifest.txt",
        help="Output text file for the manifest."
    )

    args = parser.parse_args()

    create_manifest(args.data_root, args.out)


if __name__ == "__main__":
    main()
