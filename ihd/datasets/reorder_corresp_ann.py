"""
Script to reorder correspondence annotation files.

This script should be applied only in STEP ONE that will have all the 
annotated points across the path. It takes a correspondence annotation 
file and reorders the first column (indices) to be sequential starting 
from 0, while preserving the original order of the rows.

Example:
    Original file with indices [1, 3, 4, 5] becomes [0, 1, 2, 3]
"""

import argparse
import numpy as np
from pathlib import Path


def reorder_correspondence_file(input_file, output_file):
    """
    Reorder the first column of a correspondence annotation file.
    
    Args:
        input_file (Path): Path to input correspondence file
        output_file (Path): Path to output reordered file
    """
    
    # Load the data
    data = np.loadtxt(input_file, delimiter=',', dtype=np.float64)
    
    # Handle single row case
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    print(f"Loaded {len(data)} points from {input_file.name}")
    
    # Check if file has indices (4+ columns) or just coordinates (2-3 columns)
    if data.shape[1] >= 4:
        print("Detected file with indices in first column")
        original_indices = data[:, 0].astype(int)
        coordinates = data[:, 1:]
        
        print(f"Original indices: {original_indices.tolist()}")
        
        # Create new sequential indices starting from 0
        new_indices = np.arange(len(data))
        print(f"New indices: {new_indices.tolist()}")
        
        # Combine new indices with original coordinates
        reordered_data = np.column_stack((new_indices, coordinates))
        
        # Determine format string based on number of columns
        if data.shape[1] == 4:  # u, v, depth or similar
            fmt = ['%d', '%.12f', '%.12f', '%.12f']
        #     header = 'idx u v depth'
        elif data.shape[1] == 5:  # u, v, x, y, z
            fmt = ['%d', '%d', '%d', '%.12f', '%.12f', '%.12f']
        #     header = 'idx u v x y z'
        else:  # Generic format
            fmt = ['%d'] + ['%.12f'] * (data.shape[1] - 1)
            # header = f'idx ' + ' '.join([f'col{i}' for i in range(1, data.shape[1])])
            
    else:
        print("Detected file without indices - adding sequential indices")
        
        # Add sequential indices as first column
        new_indices = np.arange(len(data))
        reordered_data = np.column_stack((new_indices, data))
        
        # Determine format based on original columns
        if data.shape[1] == 2:  # u, v
            fmt = ['%d', '%d', '%d']
        #     header = 'idx u v'
        elif data.shape[1] == 3:  # u, v, depth or x, y, z
            fmt = ['%d', '%.12f', '%.12f', '%.12f']
        #     header = 'idx col1 col2 col3'
        else:  # Generic
            fmt = ['%d'] + ['%.12f'] * data.shape[1]
        #     header = f'idx ' + ' '.join([f'col{i}' for i in range(1, data.shape[1] + 1)])
    
    # Save the reordered data
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_file,
        reordered_data,
        fmt=fmt,
        delimiter=',',
        # header=None,
        comments='',    
    )
    
    print(f"✅ Saved reordered file with {len(reordered_data)} points to {output_file}")
    
    return reordered_data


def main():
    parser = argparse.ArgumentParser(
        description="Reorder correspondence annotation file indices to be sequential starting from 0."
    )
    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Path to input correspondence file (comma-separated)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Path to output reordered file (default: input_file_reordered.txt)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # Set default output path if not provided
    if args.output is None:
        args.output = args.input.parent / f"{args.input.stem}_reordered{args.input.suffix}"
    
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    
    # Process the file
    reorder_correspondence_file(args.input, args.output)


if __name__ == "__main__":
    main()
