#!/usr/bin/env python3
"""
Script to combine HBDIA benchmark results into comprehensive CSV files.
Creates separate files for cuSPARSE and HBDIA results with noise levels.
"""

import os
import pandas as pd
import re
import glob
from pathlib import Path

def extract_noise_from_folder(folder_name):
    """Extract noise level from folder name like '3D27Stencil_128_128_128_0.001000_20250821_163209_715'"""
    match = re.search(r'_(\d+\.\d+)_\d{8}_\d{6}_\d{3}$', folder_name)
    if match:
        return float(match.group(1))
    return None

def combine_measurements(base_path, output_dir):
    """Combine all measurement files into comprehensive CSV files"""
    
    # Initialize lists to store combined data
    cusparse_data = []
    hbdia_data = []
    
    # Get all measurement directories
    data_dirs = glob.glob(os.path.join(base_path, "3D27Stencil_128_128_128_*"))
    data_dirs.sort()  # Sort for consistent ordering
    
    print(f"Found {len(data_dirs)} measurement directories")
    
    for data_dir in data_dirs:
        folder_name = os.path.basename(data_dir)
        noise_level = extract_noise_from_folder(folder_name)
        
        if noise_level is None:
            print(f"Warning: Could not extract noise level from {folder_name}")
            continue
            
        print(f"Processing noise level: {noise_level}")
        
        # Process cuSPARSE measurements
        cusparse_file = os.path.join(data_dir, "cusparse_measurements.csv")
        if os.path.exists(cusparse_file):
            try:
                df_cusparse = pd.read_csv(cusparse_file)
                df_cusparse['noise_level'] = noise_level
                df_cusparse['folder'] = folder_name
                cusparse_data.append(df_cusparse)
            except Exception as e:
                print(f"Error reading {cusparse_file}: {e}")
        else:
            print(f"Warning: {cusparse_file} not found")
            
        # Process HBDIA measurements
        hbdia_file = os.path.join(data_dir, "hbdia_measurements.csv")
        if os.path.exists(hbdia_file):
            try:
                df_hbdia = pd.read_csv(hbdia_file)
                df_hbdia['noise_level'] = noise_level
                df_hbdia['folder'] = folder_name
                hbdia_data.append(df_hbdia)
            except Exception as e:
                print(f"Error reading {hbdia_file}: {e}")
        else:
            print(f"Warning: {hbdia_file} not found")
    
    # Combine all data
    if cusparse_data:
        combined_cusparse = pd.concat(cusparse_data, ignore_index=True)
        # Reorder columns to put noise_level first
        cols = ['noise_level', 'iteration', 'time_ms', 'folder']
        combined_cusparse = combined_cusparse[cols]
        
        # Sort by noise level and iteration
        combined_cusparse = combined_cusparse.sort_values(['noise_level', 'iteration'])
        
        # Save combined cuSPARSE data
        cusparse_output = os.path.join(output_dir, "combined_cusparse_results_128x128x128.csv")
        combined_cusparse.to_csv(cusparse_output, index=False)
        print(f"Saved combined cuSPARSE results to: {cusparse_output}")
        print(f"Total cuSPARSE measurements: {len(combined_cusparse)}")
        
        # Print summary statistics
        print("\ncuSPARSE Summary by Noise Level:")
        summary_cusparse = combined_cusparse.groupby('noise_level')['time_ms'].agg(['count', 'mean', 'std', 'min', 'max'])
        print(summary_cusparse)
    
    if hbdia_data:
        combined_hbdia = pd.concat(hbdia_data, ignore_index=True)
        # Reorder columns to put noise_level first
        cols = ['noise_level', 'iteration', 'time_ms', 'folder']
        combined_hbdia = combined_hbdia[cols]
        
        # Sort by noise level and iteration
        combined_hbdia = combined_hbdia.sort_values(['noise_level', 'iteration'])
        
        # Save combined HBDIA data
        hbdia_output = os.path.join(output_dir, "combined_hbdia_results_128x128x128.csv")
        combined_hbdia.to_csv(hbdia_output, index=False)
        print(f"Saved combined HBDIA results to: {hbdia_output}")
        print(f"Total HBDIA measurements: {len(combined_hbdia)}")
        
        # Print summary statistics
        print("\nHBDIA Summary by Noise Level:")
        summary_hbdia = combined_hbdia.groupby('noise_level')['time_ms'].agg(['count', 'mean', 'std', 'min', 'max'])
        print(summary_hbdia)
    
    # Create comparison summary
    if cusparse_data and hbdia_data:
        print("\nCreating comparison summary...")
        
        # Calculate mean times per noise level
        cusparse_means = combined_cusparse.groupby('noise_level')['time_ms'].mean().reset_index()
        cusparse_means.columns = ['noise_level', 'cusparse_mean_ms']
        
        hbdia_means = combined_hbdia.groupby('noise_level')['time_ms'].mean().reset_index()
        hbdia_means.columns = ['noise_level', 'hbdia_mean_ms']
        
        # Merge for comparison
        comparison = pd.merge(cusparse_means, hbdia_means, on='noise_level', how='outer')
        comparison['speedup'] = comparison['cusparse_mean_ms'] / comparison['hbdia_mean_ms']
        comparison['hbdia_faster_by_percent'] = ((comparison['cusparse_mean_ms'] - comparison['hbdia_mean_ms']) / comparison['cusparse_mean_ms'] * 100)
        
        # Save comparison
        comparison_output = os.path.join(output_dir, "performance_comparison_128x128x128.csv")
        comparison.to_csv(comparison_output, index=False)
        print(f"Saved performance comparison to: {comparison_output}")
        
        print("\nPerformance Comparison:")
        print(comparison)

if __name__ == "__main__":
    base_path = "/users/nrottste/HBDIA/benchmarking/data/runtime_128_128_128_comparism_GPUonly"
    output_dir = "/users/nrottste/HBDIA/benchmarking/data"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting data combination process...")
    combine_measurements(base_path, output_dir)
    print("\nData combination complete!")
