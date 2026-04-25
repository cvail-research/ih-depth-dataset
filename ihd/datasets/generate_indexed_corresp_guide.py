import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spectral as spy

from calibration_lidar_cylindrical import read_corresp


def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate a guide image with zero-based indices for correspondence points."
    )
    ap.add_argument("--corresp", required=True, help="Correspondence .txt file with i j X Y Z rows")
    ap.add_argument("--hsi-hdr", required=True, help="ENVI .hdr file for the scene")
    ap.add_argument("--out", required=True, help="Output PNG path")
    return ap.parse_args()


def load_gray_image(hdr_path: Path):
    bsq_path = hdr_path.with_suffix(".bsq")
    img = spy.envi.open(str(hdr_path), str(bsq_path))
    cube = img.load()
    gray = cube.sum(axis=-1).astype("float64")
    gray -= gray.min()
    if gray.max() > 0:
        gray /= gray.max()
    return gray


def main():
    args = parse_args()

    corr_i, corr_j, _ = read_corresp(args.corresp)
    gray = load_gray_image(Path(args.hsi_hdr))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 4), dpi=180)
    ax.imshow(gray, cmap="gray", interpolation="nearest")
    ax.scatter(corr_i, corr_j, s=44, facecolors="none", edgecolors="red", linewidths=1.5)

    for idx, (ii, jj) in enumerate(zip(corr_i, corr_j)):
        ax.text(
            ii + 8,
            jj - 6,
            str(idx),
            color="yellow",
            fontsize=9,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.55, edgecolor="none"),
        )

    ax.set_title("Correspondence points with zero-based indices")
    ax.set_xlim(0, gray.shape[1] - 1)
    ax.set_ylim(gray.shape[0] - 1, 0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved guide: {out_path}")
    print(f"Indexed points: {len(corr_i)}")


if __name__ == "__main__":
    main()
