import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import math
import re

def generate_scene_id(row):
    """Generates a unique numeric ID from collection, path, step, and lwhsi_collect_num."""
    try:
        # --- 1) extract numeric collection code (e.g. 202104) ---
        m_col = re.search(r'IHTest_(\d+)', row['collect'])
        if not m_col: return -1
        collection_num = m_col.group(1)

        # --- 2) extract path and step numbers ---
        path_match = re.search(r'(\d+)', row['path'])
        step_match = re.search(r'Step(\d+)', row['step'])
        if not path_match or not step_match: return -1
        path_num = f"{int(path_match.group(1)):02d}"
        step_num = f"{int(step_match.group(1)):02d}"

        # --- 3) extract collect number (default to 0) ---
        m_coll = re.search(r'(\d+)', row['lwhsi_collect_num'])
        collect_num = f"{int(m_coll.group(1)):02d}" if m_coll else "00"

        # --- 4) build and return integer ID ---
        scene_id_str = f"{collection_num}{path_num}{step_num}{collect_num}"
        return int(scene_id_str)
    except (ValueError, TypeError):
        return -1

def format_size(size_bytes):
    """Converts a size in bytes to a human-readable format (KB, MB, GB)."""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

def main(args):
    """
    Loads a manifest, identifies strict and comprehensive data steps,
    calculates their total size, and saves them to separate, correctly filtered CSV files.
    """
    # Load your existing manifest
    manifest = pd.read_csv(args.manifest)

    # Check if the required 'ssize_bytesize' column exists
    if 'size_bytes' not in manifest.columns:
        print("❌ Error: Manifest must contain a 'size_bytes' column to compute file sizes. Exiting.")
        return

    # --- Data Pre-processing ---
    # Extract common identifiers from the s3_key
    manifest[['collect', 'path', 'step']] = manifest['s3_key'].str.extract(
        r'^(IHTest_[^/]+)/([^/]+)/([^/]+_Step\d+[^/]*)'
    ).fillna('') # Use fillna to handle non-matching patterns gracefully
    
    # Create a unique identifier for each step
    manifest['step_id'] = manifest['collect'] + '/' + manifest['path'] + '/' + manifest['step']
    
    # Drop rows where a step_id could not be formed
    manifest.dropna(subset=['step_id'], inplace=True)

    # Categorize files based on keywords in the path
    manifest['file_type'] = np.where(
        manifest['s3_key'].str.contains('HiResLIDAR'), 'HiResLIDAR',
        np.where(manifest['s3_key'].str.contains('LWHSI'), 'LWHSI', 'other')
    )
    
    # Extract LWHSI collection number (e.g., collect0, collect1) for comprehensive analysis
    manifest['lwhsi_collect_num'] = manifest['filename'].str.extract(r'LWHSI\d+_(collect\d+)').fillna('collect0')


    # --- Step Analysis ---
    # Define required file extensions for different levels of completeness
    REQUIRED_BASIC_LWHSI = {'.bsq', '.hdr'}
    REQUIRED_FULL_LWHSI = {'.bsq', '.hdr', '.cyl', '.txt'}

    def analyze_step_group(group):
        """
        Analyzes files within a step, identifying LiDAR presence, all LWHSI extensions,
        and which specific LWHSI collections are complete.
        """
        analysis = {
            'has_lidar': False,
            'all_lwhsi_extensions': set(),
            'complete_lwhsi_collects': [] # Will store names of complete collections, e.g., ['collect0']
        }
        analysis['has_lidar'] = any((group['file_type'] == 'HiResLIDAR') & (group['filename'].str.lower().str.endswith('.las')))
        
        lwhsi_files = group[group['file_type'] == 'LWHSI']
        if lwhsi_files.empty:
            return pd.Series(analysis)

        # Group LWHSI files by collection number to check for completeness
        collect_exts = lwhsi_files.groupby('lwhsi_collect_num')['filename'].apply(lambda f: {Path(name).suffix.lower() for name in f})
        
        # Identify which collections are complete
        for collect_num, exts in collect_exts.items():
            if REQUIRED_FULL_LWHSI.issubset(exts):
                analysis['complete_lwhsi_collects'].append(collect_num)
        
        # Store all unique extensions found in the step for the strict check
        analysis['all_lwhsi_extensions'] = set.union(*collect_exts.values) if not collect_exts.empty else set()

        return pd.Series(analysis)

    # Group by step_id and apply the analysis function
    step_analysis = manifest.groupby('step_id', group_keys=False).apply(analyze_step_group, include_groups=False)

    # --- Identify Valid Steps ---

    # **Strict Steps:** Must have LiDAR and the basic LWHSI files (.bsq, .hdr) anywhere in the step.
    # This uses the successful logic from your 'previous' script.
    strict_mask = (step_analysis['has_lidar']) & \
                  (step_analysis['all_lwhsi_extensions'].apply(lambda x: REQUIRED_BASIC_LWHSI.issubset(x)))
    strict_step_ids = step_analysis[strict_mask].index

    # **Comprehensive Steps:** Must have LiDAR and at least one fully complete LWHSI collection (.bsq, .hdr, .cyl, .txt).
    comprehensive_mask = (step_analysis['has_lidar']) & \
                         (step_analysis['complete_lwhsi_collects'].apply(lambda x: len(x) > 0))
    comprehensive_step_ids = step_analysis[comprehensive_mask].index


    # --- Create and Save Filtered Manifests ---
    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True)

    # 1. Create the STRICT manifest
    if not strict_step_ids.empty:
        strict_manifest = manifest[manifest['step_id'].isin(strict_step_ids)].copy()
        
        # **Crucial Improvement:** Filter to ONLY include the required file types.
        file_mask = (
            (strict_manifest['file_type'] == 'HiResLIDAR') & (strict_manifest['filename'].str.lower().str.endswith('.las'))
        ) | (
            (strict_manifest['file_type'] == 'LWHSI') & (strict_manifest['filename'].apply(lambda f: Path(f).suffix.lower()).isin(REQUIRED_BASIC_LWHSI))
        )
        strict_manifest = strict_manifest[file_mask]

        # Calculate total size
        total_size_strict = strict_manifest['size_bytes'].sum()

        # --- Add scene_id ---
        strict_manifest['scene_id'] = strict_manifest.apply(generate_scene_id, axis=1)

        strict_manifest.to_csv(args.strict, index=False)
        print(f"✅ Saved strict manifest with {len(strict_manifest)} entries from {len(strict_step_ids)} steps to {args.strict}")
        print(f"   Total size of files: {format_size(total_size_strict)}")
        print("File type distribution in strict manifest:")
        print(strict_manifest['file_type'].value_counts().to_dict())

    # 2. Create the COMPREHENSIVE manifest
    if not comprehensive_step_ids.empty:
        comprehensive_manifest = manifest[manifest['step_id'].isin(comprehensive_step_ids)].copy()

        # Filter to include only LiDAR and files from COMPLETE LWHSI collections.
        complete_collects_map = step_analysis.loc[comprehensive_step_ids, 'complete_lwhsi_collects']
        comprehensive_manifest['valid_collects'] = comprehensive_manifest['step_id'].map(complete_collects_map)

        def is_valid_comprehensive_row(row):
            if row['file_type'] == 'HiResLIDAR':
                return True
            if row['file_type'] == 'LWHSI':
                return row['lwhsi_collect_num'] in row['valid_collects']
            return False
        
        comprehensive_manifest = comprehensive_manifest[comprehensive_manifest.apply(is_valid_comprehensive_row, axis=1)].drop(columns=['valid_collects'])

        # Calculate total size
        total_size_comprehensive = comprehensive_manifest['size_bytes'].sum()

        # --- Add scene_id ---
        comprehensive_manifest['scene_id'] = comprehensive_manifest.apply(generate_scene_id, axis=1)

        comprehensive_manifest.to_csv(args.comprehensive, index=False)
        print(f"\n✅ Saved comprehensive manifest with {len(comprehensive_manifest)} entries from {len(comprehensive_step_ids)} steps to {args.comprehensive}")
        print(f"   Total size of files: {format_size(total_size_comprehensive)}")
        print("File type distribution in comprehensive manifest:")
        print(comprehensive_manifest['file_type'].value_counts().to_dict())

    # --- Summary ---
    print("\n--- Dataset Summary ---")
    print(f"Total unique steps found in manifest: {len(step_analysis)}")
    print(f"Steps with any HiResLIDAR file: {step_analysis['has_lidar'].sum()}")
    print(f"Steps qualifying for STRICT manifest: {len(strict_step_ids)}")
    print(f"Steps qualifying for COMPREHENSIVE manifest: {len(comprehensive_step_ids)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create strict and comprehensive manifests from a main dataset manifest. "
                    "Strict pairs contain HiResLIDAR (.las) and basic LWHSI (.bsq, .hdr). "
                    "Comprehensive pairs contain HiResLIDAR and a full LWHSI collection (.bsq, .hdr, .txt, .cyl)."
                    "Calculates total file size for each generated manifest."
    )
    parser.add_argument("--manifest", type=str, default="data/ihdataset_manifest.csv", help="Path to the main input manifest CSV.")
    parser.add_argument("--strict", type=str, default="ihd/datasets/manifests/ihdataset_strict.csv", help="Output path for the strict pairs manifest.")
    parser.add_argument("--comprehensive", type=str, default="ihd/datasets/manifests/ihdataset_comprehensive.csv", help="Output path for the comprehensive pairs manifest.")

    main(parser.parse_args())
