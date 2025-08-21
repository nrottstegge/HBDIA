#!/usr/bin/env python3
"""
Plot HBDIA storage requirements with confidence intervals using bootstrap.
Follows scientific benchmarking best practices from Hoefler et al.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

def bootstrap_median_ci(data, confidence_level=0.95, n_resamples=10000):
    """Calculate bootstrap confidence interval for median."""
    np.random.seed(42)  # For reproducibility
    bootstrap_medians = []
    
    for _ in range(n_resamples):
        # Resample with replacement
        resampled = np.random.choice(data, size=len(data), replace=True)
        bootstrap_medians.append(np.median(resampled))
    
    # Calculate confidence interval
    alpha = (1 - confidence_level) / 2
    ci_low = np.percentile(bootstrap_medians, alpha * 100)
    ci_high = np.percentile(bootstrap_medians, (1 - alpha) * 100)
    
    return ci_low, ci_high

def parametric_mean_ci(data, confidence_level=0.95):
    """Calculate parametric confidence interval for mean (assumes normal distribution)."""
    n = len(data)
    mean_val = np.mean(data)
    std_err = stats.sem(data)  # Standard error of the mean
    
    # t-distribution critical value
    alpha = 1 - confidence_level
    t_crit = stats.t.ppf(1 - alpha/2, n - 1)
    
    margin_error = t_crit * std_err
    ci_low = mean_val - margin_error
    ci_high = mean_val + margin_error
    
    return ci_low, ci_high

def check_normality(data, name):
    """Check normality using D'Agostino and Pearson's test."""
    if len(data) < 8:  # Need at least 8 samples
        return False, np.nan
    stat, p_value = stats.normaltest(data)
    is_normal = p_value > 0.05
    print(f"{name}: p-value={p_value:.4f}, {'Normal' if is_normal else 'Not normal'}")
    return is_normal, p_value

def main():
    # Load data
    df = pd.read_csv('/users/nrottste/HBDIA/benchmarking/data/storage_32_32_32_noise_comparism/combined_matrix_info.csv')
    
    # Convert storage to MB
    df['hbdia_storage_mb'] = df['hbdia_cost_bytes'] / (1024**2)
    
    # Get unique noise values and thresholds
    noise_values = sorted(df['extracted_noise'].unique())
    thresholds = sorted(df['threshold'].unique())
    
    print(f"Noise values: {noise_values}")
    print(f"Thresholds: {thresholds}")
    
    # Prepare data for plotting
    results = {}
    
    # First pass: Check if ANY condition is non-normal
    all_normal = True
    normality_results = {}
    
    for threshold in thresholds:
        threshold_data = df[df['threshold'] == threshold]
        for noise in noise_values:
            data = threshold_data[threshold_data['extracted_noise'] == noise]['hbdia_storage_mb']
            if len(data) > 0:
                is_normal, p_val = check_normality(data.values, f"Threshold {threshold}, Noise {noise}")
                normality_results[(threshold, noise)] = is_normal
                if not is_normal:
                    all_normal = False
    
    # Determine which statistic to use for ALL conditions (for comparability)
    if all_normal:
        use_mean = True
        print("\n=== ALL conditions are normal → Using MEAN + parametric CI for all ===")
    else:
        use_mean = False
        print(f"\n=== Some conditions are non-normal → Using MEDIAN + bootstrap CI for all (comparability) ===")
    
    # Second pass: Apply consistent statistic to all conditions
    for threshold in thresholds:
        threshold_data = df[df['threshold'] == threshold]
        central_values, ci_lows, ci_highs = [], [], []
        
        for noise in noise_values:
            data = threshold_data[threshold_data['extracted_noise'] == noise]['hbdia_storage_mb']
            
            if len(data) > 0:
                if use_mean:
                    # Use mean with parametric CI for all (all conditions are normal)
                    central_val = np.mean(data)
                    ci_low, ci_high = parametric_mean_ci(data.values)
                else:
                    # Use median with bootstrap CI for all (ensures comparability)
                    central_val = np.median(data)
                    ci_low, ci_high = bootstrap_median_ci(data.values)
                
                central_values.append(central_val)
                ci_lows.append(central_val - ci_low)
                ci_highs.append(ci_high - central_val)
            else:
                central_values.append(np.nan)
                ci_lows.append(0)
                ci_highs.append(0)
        
        results[threshold] = {
            'central_values': central_values,
            'ci_lows': ci_lows, 
            'ci_highs': ci_highs
        }
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Set up bar positions
    x = np.arange(len(noise_values))
    width = 0.12
    colors = plt.cm.viridis(np.linspace(0, 1, len(thresholds)))
    
    for i, threshold in enumerate(thresholds):
        offset = (i - len(thresholds)/2 + 0.5) * width
        
        plt.bar(x + offset, results[threshold]['central_values'], width, 
               yerr=[results[threshold]['ci_lows'], results[threshold]['ci_highs']], 
               label=f'Threshold {threshold}', color=colors[i], 
               capsize=3, alpha=0.8)
    
    # Find and annotate the threshold with minimum central value for each noise level
    for j, noise in enumerate(noise_values):
        min_central = float('inf')
        best_threshold = None
        max_height = 0
        
        for threshold in thresholds:
            central_val = results[threshold]['central_values'][j]
            if not np.isnan(central_val) and central_val < min_central:
                min_central = central_val
                best_threshold = threshold
            
            # Find the maximum bar height for this noise level (for annotation positioning)
            if not np.isnan(central_val):
                bar_top = central_val + results[threshold]['ci_highs'][j]
                max_height = max(max_height, bar_top)
        
        # Add annotation above the highest bar for this noise level
        if best_threshold is not None:
            plt.text(j, max_height + max_height * 0.02, f'T{best_threshold}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Format x-axis labels with appropriate precision
    def format_percentage(n):
        if n == 0:
            return "0%"
        elif n == 0.0001:
            return f"{n*100:.2f}%"
        elif n == 0.001:
            return f"{n*100:.1f}%"
        else:
            return f"{n*100:.0f}%"
    
    plt.xlabel('Noise Level (%)', fontsize=12)
    plt.ylabel('HBDIA Storage (MB)', fontsize=12)
    
    # Dynamic title based on which statistic was used
    statistic_used = "Mean ± 95% Parametric CI" if use_mean else "Median ± 95% Bootstrap CI"
    plt.title(f'HBDIA Storage Requirements vs Noise Level\n({statistic_used})', fontsize=14)
    
    plt.xticks(x, [format_percentage(n) for n in noise_values])
    plt.yscale('log')
    plt.legend(loc='upper left', framealpha=0.9, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('/users/nrottste/HBDIA/benchmarking/Plotting/hbdia_storage_analysis.pdf', 
                bbox_inches='tight', format='pdf')
    plt.show()
    
    # Print summary statistics
    statistic_name = "mean" if use_mean else "median"
    print(f"\nSummary Statistics (MB) - Using {statistic_name} for all conditions (comparability):")
    for threshold in thresholds:
        threshold_data = df[df['threshold'] == threshold]['hbdia_storage_mb']
        
        if use_mean:
            central_val = np.mean(threshold_data)
        else:
            central_val = np.median(threshold_data)
            
        print(f"Threshold {threshold}: {statistic_name}={central_val:.2f}, "
              f"IQR=[{np.percentile(threshold_data, 25):.2f}, {np.percentile(threshold_data, 75):.2f}]")

if __name__ == "__main__":
    main()
