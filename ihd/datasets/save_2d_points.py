import cv2
import numpy as np
import spectral as spy
from pathlib import Path
import argparse

# === Function Definitions ===

def redraw_image():
    """Redraws all current points on a fresh copy of the base image."""
    global img_for_display
    img_for_display = base_img.copy()
    for i, pt in enumerate(points_2d):
        cv2.circle(img_for_display, tuple(pt), 3, (0, 255, 0), -1)
        # Use the original index if available, otherwise use current position
        display_idx = point_indices[i] if point_indices is not None else i
        cv2.putText(img_for_display, str(display_idx), (pt[0] + 8, pt[1] + 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
    cv2.imshow('Click on image', img_for_display)

def skip_current_point():
    """Skip the current point by adding a placeholder (-1, -1)."""
    global points_2d, points_3d, point_indices
    
    current_point_idx = len(points_2d)
    if current_point_idx >= len(points_3d):
        print("All 3D points already processed. Press 'q' to save.")
        return
    
    # Use the original index if available, otherwise use current position
    display_idx = point_indices[current_point_idx] if point_indices is not None else current_point_idx
    print(f'Skipped point {display_idx}')
    points_2d.append([-1, -1])  # Placeholder for skipped point
    
    # Redraw the image (skipped points won't be displayed)
    redraw_image()
    
    # Update console prompt for the next point
    if (current_point_idx + 1) < len(points_3d):
        xyz = points_3d[current_point_idx + 1]
        next_display_idx = point_indices[current_point_idx + 1] if point_indices is not None else (current_point_idx + 1)
        print(f"\nNext, click the 2D point for 3D point {next_display_idx}: ({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})")
    else:
        print("\nAll points processed! Press 'q' to quit and save.")

def click_event(event, x, y, flags, param):
    """Mouse callback to record clicked points."""
    global points_2d, points_3d, point_indices
    
    if event == cv2.EVENT_LBUTTONDOWN:
        current_point_idx = len(points_2d)
        if current_point_idx >= len(points_3d):
            print("All 3D points already have a corresponding 2D point. Press 'q' to save.")
            return

        # Use the original index if available, otherwise use current position
        display_idx = point_indices[current_point_idx] if point_indices is not None else current_point_idx
        print(f'Clicked point {display_idx}: ({x}, {y})')
        points_2d.append([x, y])
        
        # Redraw the image with the new point
        redraw_image()
        
        # Update console prompt for the next point
        if (current_point_idx + 1) < len(points_3d):
            xyz = points_3d[current_point_idx + 1]
            next_display_idx = point_indices[current_point_idx + 1] if point_indices is not None else (current_point_idx + 1)
            print(f"\nNext, click the 2D point for 3D point {next_display_idx}: ({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})")
        else:
            print("\nAll points selected! Press 'q' to quit and save.")


# === Main Execution ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Click 2D points on a hyperspectral image to correspond with a list of 3D points, then save the combined data."
    )
    parser.add_argument('--hdr', type=Path, required=True, help='Path to the input .hdr hyperspectral image file.')
    parser.add_argument('--xyz', type=Path, required=True, help='Path to the input .txt file with 3D (x,y,z) coordinates, comma-separated, no header.')
    parser.add_argument('--output', type=Path, required=True, help='Path for the output .txt file with merged (u,v,x,y,z) data.')
    args = parser.parse_args()

    # --- Validate paths and load data ---
    if not args.hdr.exists():
        raise FileNotFoundError(f"HDR file not found: {args.hdr}")
    if not args.xyz.exists():
        raise FileNotFoundError(f"XYZ file not found: {args.xyz}")

    bsq_path = args.hdr.with_suffix('.bsq')
    if not bsq_path.exists():
        raise FileNotFoundError(f"Matching BSQ file not found at: {bsq_path}")

    # Load the raw data from the XYZ file
    points_3d_raw = np.loadtxt(args.xyz, delimiter=',')
    
    # Handle files with only a single line, which load as 1D arrays
    if points_3d_raw.ndim == 1:
        points_3d_raw = points_3d_raw.reshape(1, -1)

    # Check for 4-column format (index, x, y, z) and adapt
    if points_3d_raw.shape[1] == 4:
        print("Detected 4 columns in XYZ file. Using original indices from first column.")
        point_indices = points_3d_raw[:, 0].astype(int)
        points_3d = points_3d_raw[:, 1:]
    elif points_3d_raw.shape[1] == 3:
        print("Detected 3 columns in XYZ file. Using default indexing (0, 1, 2, ...).")
        point_indices = None
        points_3d = points_3d_raw
    else:
        raise ValueError(
            f"XYZ file must have 3 (x,y,z) or 4 (idx,x,y,z) columns, but found {points_3d_raw.shape[1]}."
        )

    print(f"Loaded {len(points_3d)} 3D points from {args.xyz.name}")

    # --- Prepare image for display ---
    spy_img = spy.envi.open(str(args.hdr), str(bsq_path))
    hsi = spy_img.load()
    hsi_broadband = np.sum(hsi, axis=-1)
    hsi_broadband_norm = (hsi_broadband - hsi_broadband.min()) / (hsi_broadband.max() - hsi_broadband.min())
    
    base_img = (hsi_broadband_norm * 255).astype(np.uint8)
    base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    img_for_display = base_img.copy()

    # --- Interactive Point Selection ---
    points_2d = []
    cv2.imshow('Click on image', img_for_display)
    cv2.setMouseCallback('Click on image', click_event)

    print("\n--- Point Selection ---")
    xyz_0 = points_3d[0]
    display_idx_0 = point_indices[0] if point_indices is not None else 0
    print(f"Click the 2D point for 3D point {display_idx_0}: ({xyz_0[0]:.3f}, {xyz_0[1]:.3f}, {xyz_0[2]:.3f})")
    print("Press 'z' to undo the last point. Press 's' to skip the current point. Press 'q' to quit.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('z'): # UNDO functionality
            if points_2d:
                last_point = points_2d.pop()
                current_point_idx = len(points_2d)
                last_display_idx = point_indices[current_point_idx] if point_indices is not None else current_point_idx
                print(f"\nUndid point {last_display_idx}: {last_point}")
                redraw_image()
                
                # Update console prompt to ask for the removed point again
                xyz = points_3d[current_point_idx]
                display_idx = point_indices[current_point_idx] if point_indices is not None else current_point_idx
                print(f"Next, click the 2D point for 3D point {display_idx}: ({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})")
            else:
                print("No points to undo.")
        elif key == ord('s'): # SKIP functionality
            skip_current_point()

    cv2.destroyAllWindows()

    # --- Save to file ---
    if not points_2d:
        print("\nNo points were clicked. Nothing to save.")
    elif len(points_2d) != len(points_3d):
        print(f"\n⚠️ Warning: Number of clicked 2D points ({len(points_2d)}) does not match loaded 3D points ({len(points_3d)}).")
        print("Saving the matched pairs only.")
        min_len = min(len(points_2d), len(points_3d))
        points_2d = points_2d[:min_len]
        points_3d = points_3d[:min_len]

    if points_2d:
        points_2d_arr = np.array(points_2d)
        combined_points = np.hstack((points_2d_arr, points_3d))
        
        # Filter out skipped points (marked as [-1, -1]) before saving
        valid_mask = ~((points_2d_arr[:, 0] == -1) & (points_2d_arr[:, 1] == -1))
        combined_points = combined_points[valid_mask]
        
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            args.output, 
            combined_points, 
            fmt=['%d', '%d', '%.6f', '%.6f', '%.6f'], 
            header='u v x y z', 
            comments=''
        )
        print(f"\n✅ Saved {len(combined_points)} merged points to {args.output}")
        
        skipped_count = len(points_2d) - len(combined_points)
        if skipped_count > 0:
            print(f"Note: {skipped_count} points were skipped and not saved.")
