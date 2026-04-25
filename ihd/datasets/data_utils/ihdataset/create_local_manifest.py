import argparse
import re
from pathlib import Path

import pandas as pd


def generate_scene_id(row):
    """Generates a unique numeric ID from collection, path, step, and lwhsi_collect_num."""
    try:
        # --- 1) extract numeric collection code (e.g. 202104) ---
        m_col = re.search(r'IHTest_(\d+)', row['collection'])
        if not m_col:
            return -1
        collection_num = m_col.group(1)

        # --- 2) extract path and step numbers ---
        path_match = re.search(r'(\d+)', row['path'])
        # Correctly find the number associated with "Step"
        step_match = re.search(r'Step(\d+)', row['step'])
        if not path_match or not step_match:
            return -1
        path_num = f"{int(path_match.group(1)):02d}"
        step_num = f"{int(step_match.group(1)):02d}"

        # --- 3) extract collect number (default to 0) ---
        collect_num_str = row.get('lwhsi_collect_num')
        if pd.isna(collect_num_str) or collect_num_str is None:
            collect_num = "00"
        else:
            m_coll = re.search(r'(\d+)', collect_num_str)
            collect_num = f"{int(m_coll.group(1)):02d}" if m_coll else "00"

        # --- 4) build and return integer ID ---
        scene_id_str = f"{collection_num}{path_num}{step_num}{collect_num}"
        return int(scene_id_str)

    except Exception:
        return -1


def main(args):
    """
    Scans a local directory, creates a manifest of all relevant files,
    assigns a stable scene_id to each path/step group, and saves it to a CSV file.
    """
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"❌ Error: Data root directory not found at '{data_root}'")
        return

    # Scan for all relevant files recursively
    print(f"Scanning for files in {data_root}...")
    all_files = list(data_root.rglob("*.las")) + \
                list(data_root.rglob("*.bsq")) + \
                list(data_root.rglob("*.hdr")) + \
                list(data_root.rglob("*.cyl")) + \
                list(data_root.rglob("*.txt"))
    
    print(f"Found {len(all_files)} total files.")

    records = []
    for f in all_files:
        # Skip temporary or backup correspondence files
        if '_ann_points' in f.name:
            continue

        relative_path = f.relative_to(data_root)
        parts = relative_path.parts
        
        collection = next((p for p in parts if p.startswith("IHTest_")), "Unknown")
        path = next((p for p in parts if p.startswith("Path")), None)
        step = next((p for p in parts if "Step" in p), None)
        
        sensor = "Unknown"
        if "HiResLIDAR" in f.name:
            sensor = "LiDAR"
        elif "LWHSI" in f.name:
            sensor = "LWHSI"
        elif f.suffix == '.txt':
            sensor = "Annotation"

        # Extract LWHSI collection number (e.g., collect0, collect1)
        lwhsi_collect_match = re.search(r'LWHSI\d+_(collect\d+)', f.name)
        lwhsi_collect_num = lwhsi_collect_match.group(1) if lwhsi_collect_match else None

        records.append({
            "file_path": str(f),
            "collection": collection,
            "path": path,
            "step": step,
            "sensor": sensor,
            "lwhsi_collect_num": lwhsi_collect_num,
            "file_name": f.name,
            "extension": f.suffix
        })
        # break

    if not records:
        print("No valid records found. Exiting.")
        return

    df = pd.DataFrame(records)

    # Drop rows where path or step could not be determined
    df.dropna(subset=['path', 'step'], inplace=True)

    # --- Associate LiDAR with all LWHSI collects in the same step ---
    df['scene_group_id'] = df['collection'] + '/' + df['path'] + '/' + df['step']

    # Create a map from each scene_group_id to its list of unique LWHSI collect numbers
    collect_map = df.dropna(subset=['lwhsi_collect_num']) \
                    .groupby('scene_group_id')['lwhsi_collect_num'] \
                    .unique() \
                    .to_dict()

    # Separate LiDAR from other files
    lidar_df = df[df['sensor'] == 'LiDAR'].copy()
    non_lidar_df = df[df['sensor'] != 'LiDAR'].copy()

    # For each LiDAR file, find its corresponding list of collects and "explode" it,
    # creating a row for each LiDAR-collect association.
    lidar_df['lwhsi_collect_num'] = lidar_df['scene_group_id'].map(collect_map)
    
    # If a scene has no 'collect' numbers (like in 202108), default to 'collect0'
    # This ensures that LiDAR files from such scenes are not dropped.
    lidar_df['lwhsi_collect_num'] = lidar_df['lwhsi_collect_num'].apply(
        lambda x: x if isinstance(x, list) or not pd.isna(x) else ['collect0']
    )
    non_lidar_df['lwhsi_collect_num'] = non_lidar_df['lwhsi_collect_num'].apply(
        lambda x: x if isinstance(x, list) or not pd.isna(x) else 'collect0'
    )

    # Drop LiDAR files that are in scenes with no LWHSI files at all
    lidar_df.dropna(subset=['lwhsi_collect_num'], inplace=True)
    
    lidar_df = lidar_df.explode('lwhsi_collect_num').reset_index(drop=True)

    # Combine the exploded LiDAR data back with the HSI data
    df = pd.concat([lidar_df, non_lidar_df], ignore_index=True)

    # --- Generate Scene IDs ---
    # A scene is now defined by collection, path, step, AND lwhsi_collect_num
    df['scene_id'] = df.apply(generate_scene_id, axis=1)

    # Sort by the new scene_id to ensure stable ordering
    df = df.sort_values(by=['scene_id', 'file_path']).reset_index(drop=True)

    # --- Save Manifest ---
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n✅ Successfully created and saved manifest to '{output_path}'")
    print(f"   Total files in manifest: {len(df)}")
    print(f"   Total unique scenes found: {df['scene_id'].nunique()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a local manifest file from a directory of IHDataset files."
    )
    parser.add_argument(
        "--data_root", 
        type=str, 
        default="/home/guillermo/ssd/datasets/interim/ihdataset", 
        help="The root directory of your local IHDataset."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/utils/local_manifest.csv", 
        help="The path to save the output manifest CSV file."
    )
    main(parser.parse_args())
