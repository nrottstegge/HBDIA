#!/usr/bin/env python3
"""
HBDIA vs cuSPARSE Runtime Comparison Analysis
Following Hoefler et al. scientific benchmarking guidelines.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'lines.linewidth': 2,
    'lines.markersize': 8
})

def load_and_validate_data():
    """Load and validate the benchmark data"""
    try:
        cusparse_df = pd.read_csv('/users/nrottste/HBDIA/benchmarking/data/combined_cusparse_results_128x128x128.csv')
        hbdia_df = pd.read_csv('/users/nrottste/HBDIA/benchmarking/data/combined_hbdia_results_128x128x128.csv')
        
        print(f"Loaded cuSPARSE data: {len(cusparse_df)} measurements")
        print(f"Loaded HBDIA data: {len(hbdia_df)} measurements")
        
        return cusparse_df, hbdia_df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None

def calculate_confidence_interval(data, confidence=0.95):
    """Calculate confidence interval using scipy.stats"""
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)  # Standard error of the mean
    
    # Use t-distribution for small samples, normal for large
    if n < 30:
        # t-distribution
        t_val = stats.t.ppf((1 + confidence) / 2, df=n-1)
        margin_error = t_val * std_err
    else:
        # Normal distribution (Central Limit Theorem)
        z_val = stats.norm.ppf((1 + confidence) / 2)
        margin_error = z_val * std_err
    
    return mean - margin_error, mean + margin_error

def bootstrap_ci_scipy(data, n_bootstrap=10000, confidence=0.95):
    """Calculate bootstrap confidence interval using scipy"""
    from scipy.stats import bootstrap
    
    # Prepare data for scipy bootstrap
    data_tuple = (data,)
    
    # Define statistic function
    def mean_statistic(x):
        return np.mean(x)
    
    # Use scipy bootstrap
    rng = np.random.default_rng(42)
    res = bootstrap(data_tuple, mean_statistic, n_resamples=n_bootstrap, 
                   confidence_level=confidence, random_state=rng)
    
    return res.confidence_interval.low, res.confidence_interval.high

def statistical_comparison(cusparse_data, hbdia_data):
    """Perform statistical comparison following Hoefler et al. guidelines using scipy"""
    
    # Test for normality using Shapiro-Wilk test (scipy implementation)
    cusparse_normal = stats.shapiro(cusparse_data).pvalue > 0.05
    hbdia_normal = stats.shapiro(hbdia_data).pvalue > 0.05
    
    # Choose appropriate test
    if cusparse_normal and hbdia_normal:
        # Parametric test - check for equal variances first
        equal_var_pvalue = stats.levene(cusparse_data, hbdia_data).pvalue
        equal_var = equal_var_pvalue > 0.05
        
        if equal_var:
            stat, p_value = stats.ttest_ind(cusparse_data, hbdia_data, equal_var=True)
            test_name = "Independent t-test (equal variance)"
        else:
            stat, p_value = stats.ttest_ind(cusparse_data, hbdia_data, equal_var=False)
            test_name = "Welch's t-test (unequal variance)"
    else:
        # Non-parametric test (Mann-Whitney U)
        stat, p_value = stats.mannwhitneyu(cusparse_data, hbdia_data, alternative='two-sided')
        test_name = "Mann-Whitney U test"
    
    # Effect size (Cohen's d) using scipy
    cohens_d = (np.mean(cusparse_data) - np.mean(hbdia_data)) / \
               np.sqrt(((len(cusparse_data) - 1) * np.var(cusparse_data, ddof=1) + 
                       (len(hbdia_data) - 1) * np.var(hbdia_data, ddof=1)) / 
                      (len(cusparse_data) + len(hbdia_data) - 2))
    
    # Effect size interpretation
    if abs(cohens_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"
    
    return {
        'test_name': test_name,
        'test_statistic': stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'effect_size': effect_size,
        'significant': p_value < 0.05,
        'cusparse_normal': cusparse_normal,
        'hbdia_normal': hbdia_normal
    }

def analyze_noise_level(cusparse_df, hbdia_df, noise_level):
    """Analyze data for a specific noise level"""
    # Filter data for this noise level
    cusparse_data = cusparse_df[cusparse_df['noise_level'] == noise_level]['time_ms'].values
    hbdia_data = hbdia_df[hbdia_df['noise_level'] == noise_level]['time_ms'].values
    
    if len(cusparse_data) == 0 or len(hbdia_data) == 0:
        return None
    
    # DO NOT remove outliers - keep all data for scientific integrity
    cusparse_clean = cusparse_data
    hbdia_clean = hbdia_data
    
    # Calculate statistics
    cusparse_mean = np.mean(cusparse_clean)
    hbdia_mean = np.mean(hbdia_clean)
    speedup = cusparse_mean / hbdia_mean
    
    # Calculate confidence intervals - try bootstrap first, fallback to parametric
    try:
        cusparse_ci = bootstrap_ci_scipy(cusparse_clean)
        hbdia_ci = bootstrap_ci_scipy(hbdia_clean)
        ci_method = "bootstrap"
    except:
        # Fallback to parametric CI
        cusparse_ci = calculate_confidence_interval(cusparse_clean)
        hbdia_ci = calculate_confidence_interval(hbdia_clean)
        ci_method = "parametric"
    
    # For speedup CI, we need to bootstrap the ratio
    try:
        from scipy.stats import bootstrap
        
        def speedup_statistic(cusparse_sample, hbdia_sample):
            return np.mean(cusparse_sample) / np.mean(hbdia_sample)
        
        rng = np.random.default_rng(42)
        data_tuple = (cusparse_clean, hbdia_clean)
        
        res = bootstrap(data_tuple, speedup_statistic, n_resamples=10000, 
                       confidence_level=0.95, random_state=rng)
        speedup_ci = (res.confidence_interval.low, res.confidence_interval.high)
    except:
        # Simple fallback using delta method approximation
        cusparse_var = np.var(cusparse_clean, ddof=1)
        hbdia_var = np.var(hbdia_clean, ddof=1)
        
        # Delta method for ratio of means
        ratio_var = (cusparse_var / (len(cusparse_clean) * hbdia_mean**2)) + \
                   (cusparse_mean**2 * hbdia_var / (len(hbdia_clean) * hbdia_mean**4))
        ratio_se = np.sqrt(ratio_var)
        
        z_val = stats.norm.ppf(0.975)  # 95% CI
        speedup_ci = (speedup - z_val * ratio_se, speedup + z_val * ratio_se)
    
    # Statistical test using scipy
    stats_result = statistical_comparison(cusparse_clean, hbdia_clean)
    
    return {
        'noise_level': noise_level,
        'cusparse_mean': cusparse_mean,
        'cusparse_ci': cusparse_ci,
        'cusparse_n': len(cusparse_clean),
        'hbdia_mean': hbdia_mean,
        'hbdia_ci': hbdia_ci,
        'hbdia_n': len(hbdia_clean),
        'speedup': speedup,
        'speedup_ci': speedup_ci,
        'stats': stats_result,
        'ci_method': ci_method
    }

def create_runtime_comparison_plot():
    """Create the main runtime comparison plot"""
    cusparse_df, hbdia_df = load_and_validate_data()
    if cusparse_df is None or hbdia_df is None:
        print("Failed to load data")
        return
    
    # Get unique noise levels
    noise_levels = sorted(cusparse_df['noise_level'].unique())
    
    # Analyze each noise level
    results = []
    for noise in noise_levels:
        result = analyze_noise_level(cusparse_df, hbdia_df, noise)
        if result:
            results.append(result)
    
    if not results:
        print("No valid results found")
        return
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Runtime comparison with confidence intervals
    noise_vals = [r['noise_level'] for r in results]
    cusparse_means = [r['cusparse_mean'] for r in results]
    hbdia_means = [r['hbdia_mean'] for r in results]
    
    # Calculate CI error bars (distance from mean to CI bounds)
    cusparse_ci_lower = [r['cusparse_mean'] - r['cusparse_ci'][0] for r in results]
    cusparse_ci_upper = [r['cusparse_ci'][1] - r['cusparse_mean'] for r in results]
    cusparse_errors = [cusparse_ci_lower, cusparse_ci_upper]
    
    hbdia_ci_lower = [r['hbdia_mean'] - r['hbdia_ci'][0] for r in results]
    hbdia_ci_upper = [r['hbdia_ci'][1] - r['hbdia_mean'] for r in results]
    hbdia_errors = [hbdia_ci_lower, hbdia_ci_upper]
    
    x = np.arange(len(noise_vals))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, cusparse_means, width, 
                    yerr=cusparse_errors, 
                    label='cuSPARSE', alpha=0.8, capsize=5,
                    color='tab:blue')
    bars2 = ax1.bar(x + width/2, hbdia_means, width,
                    yerr=hbdia_errors,
                    label='HBDIA', alpha=0.8, capsize=5,
                    color='tab:orange')
    
    ax1.set_xlabel('Noise Level')
    ax1.set_ylabel('Runtime (ms)')
    ax1.set_title('Runtime Comparison with 95% Bootstrap CI')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{n:.1%}' if n > 0 else '0%' for n in noise_vals])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup with confidence intervals
    speedups = [r['speedup'] for r in results]
    speedup_ci_lower = [r['speedup'] - r['speedup_ci'][0] for r in results]
    speedup_ci_upper = [r['speedup_ci'][1] - r['speedup'] for r in results]
    speedup_errors = [speedup_ci_lower, speedup_ci_upper]
    
    ax2.errorbar(x, speedups, yerr=speedup_errors, 
                 fmt='o-', linewidth=2, markersize=8, capsize=5,
                 color='tab:green', label='HBDIA Speedup')
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
    ax2.set_xlabel('Noise Level')
    ax2.set_ylabel('Speedup Factor (cuSPARSE/HBDIA)')
    ax2.set_title('HBDIA Speedup vs Noise Level with 95% Bootstrap CI')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{n:.1%}' if n > 0 else '0%' for n in noise_vals])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plots
    output_dir = '/users/nrottste/HBDIA/benchmarking/data'
    plt.savefig(f'{output_dir}/hbdia_cusparse_runtime_comparison.pdf', 
                bbox_inches='tight', format='pdf')
    plt.savefig(f'{output_dir}/hbdia_cusparse_runtime_comparison.png', 
                bbox_inches='tight', dpi=300)
    
    # Print summary statistics
    print("\n=== RUNTIME COMPARISON SUMMARY ===")
    print(f"{'Noise Level':<12} {'cuSPARSE (ms)':<15} {'HBDIA (ms)':<15} {'Speedup':<10} {'CI Method':<12} {'Significant':<12}")
    print("-" * 80)
    
    for r in results:
        sig_status = "Yes" if r['stats']['significant'] else "No"
        print(f"{r['noise_level']:<12.1%} "
              f"{r['cusparse_mean']:<15.3f} "
              f"{r['hbdia_mean']:<15.3f} "
              f"{r['speedup']:<10.2f} "
              f"{r.get('ci_method', 'unknown'):<12} "
              f"{sig_status:<12}")
    
    avg_speedup = np.mean([r['speedup'] for r in results])
    print(f"\nAverage speedup: {avg_speedup:.2f}x")
    print(f"Maximum speedup: {max(r['speedup'] for r in results):.2f}x")
    print(f"Minimum speedup: {min(r['speedup'] for r in results):.2f}x")
    
    # Print statistical method summary
    print(f"\nStatistical Methods Used:")
    for r in results:
        print(f"  Noise {r['noise_level']:.1%}: {r['stats']['test_name']}, "
              f"Effect size: {r['stats']['effect_size']} (Cohen's d = {r['stats']['cohens_d']:.3f})")
    
    plt.show()

if __name__ == "__main__":
    create_runtime_comparison_plot()
