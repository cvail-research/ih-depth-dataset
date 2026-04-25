import argparse
import glob
import os

import matplotlib.pyplot as plt
import spectral as spy


def pseudo_broadband(hsi):
    """
    Fast pseudo-broadband image.
    - hsi: numpy array (H, W, B)
    """
    img = hsi.sum(axis=2)      # sum all bands
    img = img.astype(float)
    img = img - img.min()
    img = img / (img.max() + 1e-6)
    return img


def find_hdr_files(root):
    """Recursively collect all .hdr files under root."""
    hdr_files = sorted(glob.glob(os.path.join(root, "**", "*.hdr"), recursive=True))
    filtered_hdr_files = [file for file in hdr_files if not str(file).__contains__("corrected")]
    return filtered_hdr_files


def inspect_hyperspectral(folder, out_file="kept_scenes.txt"):
    # list hdr files (each corresponds to a .bsq)
    hdr_files = find_hdr_files(folder)

    if not hdr_files:
        print("No .hdr files found recursively.")
        return

    kept = []

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))

    for fname in hdr_files:
        hdr_path = os.path.join(folder, fname)

        # Load hyperspectral cube
        hsi = spy.open_image(hdr_path).load()  # loads fully into RAM
        img = pseudo_broadband(hsi)

        # Display
        ax.clear()
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{fname}\n[k] keep | [d] discard | [q] quit")
        ax.axis("off")
        fig.canvas.draw()

        print(f"Showing {fname}...")

        key = input("k/d/q: ").strip().lower()

        if key == "k":
            kept.append(fname)
        elif key == "d":
            pass
        elif key == "q":
            break
        else:
            print("Invalid key, use k/d/q.")

    # Save kept list
    with open(out_file, "w") as f:
        for k in kept:
            f.write(k + "\n")

    print(f"Done. Kept {len(kept)} scenes → saved to {out_file}")


def main():
    parser = argparse.ArgumentParser(description='Visually inspect all .hdr ' \
    'scenes in a folder creating a pseudo-broadband image by summing all bands')
    parser.add_argument('--folder')
    args = parser.parse_args()
    inspect_hyperspectral(folder=args.folder)


if __name__ == "__main__":
    main()
