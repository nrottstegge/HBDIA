#!/usr/bin/env python3
"""
Combine CSV data from parameter sweep results into single files for plotting.
"""

import os
import pandas as pd
import re
from pathlib import Path

def extract_parameters_from_dirname(dirname):
    """Extract noise and threshold values from directory name."""
    # Pattern: 3D27Stencil_32_32_32_{noise}_{timestamp}
    pattern = r"3D27Stencil_32_32_32_(\d+\.\d+)_\d{8}_\d{6}_\d+"
    match = re.match(pattern, dirname)
    if match:
        noise = float(match.group(1))
        return noise
    return None

def combine_csv_files():
    """Combine all CSV files from the benchmark runs."""
    data_dir = Path("/users/nrottste/HBDIA/benchmarking/data/storage_32_32_32_noise_comparism")
    
    matrix_info_data = []
    histogram_data = []
    
    # Iterate through all subdirectories
    for subdir in data_dir.iterdir():
        if not subdir.is_dir():
            continue
            
        noise = extract_parameters_from_dirname(subdir.name)
        if noise is None:
            continue
            
        # Read matrix_info.csv
        matrix_info_file = subdir / "matrix_info.csv"
        if matrix_info_file.exists():
            df = pd.read_csv(matrix_info_file)
            # Add directory info for tracking
            df['directory'] = subdir.name
            df['extracted_noise'] = noise
            matrix_info_data.append(df)
            
        # Read histogram.csv
        histogram_file = subdir / "histogram.csv"
        if histogram_file.exists():
            df = pd.read_csv(histogram_file)
            # Add directory info for tracking
            df['directory'] = subdir.name
            df['extracted_noise'] = noise
            histogram_data.append(df)
    
    # Combine all data
    if matrix_info_data:
        combined_matrix_info = pd.concat(matrix_info_data, ignore_index=True)
        output_file = data_dir.parent / "combined_matrix_info.csv"
        combined_matrix_info.to_csv(output_file, index=False)
        print(f"Combined matrix info data saved to: {output_file}")
        print(f"Total matrix info records: {len(combined_matrix_info)}")
        
    if histogram_data:
        combined_histogram = pd.concat(histogram_data, ignore_index=True)
        output_file = data_dir.parent / "combined_histogram.csv"
        combined_histogram.to_csv(output_file, index=False)
        print(f"Combined histogram data saved to: {output_file}")
        print(f"Total histogram records: {len(combined_histogram)}")
    
    # Print summary statistics
    if matrix_info_data:
        print("\nMatrix Info Summary:")
        print(f"Noise values: {sorted(combined_matrix_info['extracted_noise'].unique())}")
        print(f"Threshold values: {sorted(combined_matrix_info['threshold'].unique())}")
        print(f"Iterations per combination: {len(combined_matrix_info) // (len(combined_matrix_info['extracted_noise'].unique()) * len(combined_matrix_info['threshold'].unique()))}")

if __name__ == "__main__":
    combine_csv_files()
