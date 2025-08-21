#!/usr/bin/env python3
"""
Concise HBDIA vs cuSPARSE Runtime Comparison
Following Hoefler et al. scientific benchmarking principles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def bootstrap_ci(data, statistic=np.mean, n_bootstrap=10000, confidence=0.95):
    """Calculate bootstrap confidence interval using scipy"""
    from scipy.stats import bootstrap
    
    def stat_func(x):
        return statistic(x)
    
    rng = np.random.default_rng(42)
    res = bootstrap((data,), stat_func, n_resamples=n_bootstrap, 
                   confidence_level=confidence, random_state=rng)
    
    return res.confidence_interval.low, res.confidence_interval.high

def main():
    # Load data (skip first measurement as warm-up)
    cusparse_df = pd.read_csv('/users/nrottste/HBDIA/benchmarking/data/combined_cusparse_results_128x128x128.csv')
    hbdia_df = pd.read_csv('/users/nrottste/HBDIA/benchmarking/data/combined_hbdia_results_128x128x128.csv')
    
    # Remove warm-up iterations (iteration 0)
    cusparse_df = cusparse_df[cusparse_df['iteration'] > 0]
    hbdia_df = hbdia_df[hbdia_df['iteration'] > 0]
    
    noise_levels = sorted(cusparse_df['noise_level'].unique())
    results = {}
    
    # Test for normality and determine statistical approach
    use_bootstrap = False
    for noise in noise_levels:
        cusparse_data = cusparse_df[cusparse_df['noise_level'] == noise]['time_ms'].values
        hbdia_data = hbdia_df[hbdia_df['noise_level'] == noise]['time_ms'].values
        
        # Shapiro-Wilk normality test
        _, p_cusparse = stats.shapiro(cusparse_data)
        _, p_hbdia = stats.shapiro(hbdia_data)
        
        if p_cusparse < 0.05 or p_hbdia < 0.05:
            use_bootstrap = True
            break
    
    method = "Bootstrap CI" if use_bootstrap else "Parametric CI"
    print(f"Using {method}")
    
    # Analyze each noise level
    for noise in noise_levels:
        cusparse_data = cusparse_df[cusparse_df['noise_level'] == noise]['time_ms'].values
        hbdia_data = hbdia_df[hbdia_df['noise_level'] == noise]['time_ms'].values
        
        if use_bootstrap:
            # Bootstrap approach
            cusparse_mean = np.mean(cusparse_data)
            hbdia_mean = np.mean(hbdia_data)
            cusparse_ci = bootstrap_ci(cusparse_data, np.mean)
            hbdia_ci = bootstrap_ci(hbdia_data, np.mean)
            
            # Bootstrap speedup CI
            speedup_bootstrap = []
            rng = np.random.default_rng(42)
            for _ in range(10000):
                c_sample = rng.choice(cusparse_data, size=len(cusparse_data), replace=True)
                h_sample = rng.choice(hbdia_data, size=len(hbdia_data), replace=True)
                speedup_bootstrap.append(np.mean(c_sample) / np.mean(h_sample))
            
            speedup = cusparse_mean / hbdia_mean
            speedup_ci = (np.percentile(speedup_bootstrap, 2.5), 
                         np.percentile(speedup_bootstrap, 97.5))
        else:
            # Parametric approach
            cusparse_mean = np.mean(cusparse_data)
            hbdia_mean = np.mean(hbdia_data)
            
            # t-distribution CI
            n_c = len(cusparse_data)
            n_h = len(hbdia_data)
            t_c = stats.t.ppf(0.975, n_c-1)
            t_h = stats.t.ppf(0.975, n_h-1)
            
            cusparse_ci = (cusparse_mean - t_c * stats.sem(cusparse_data),
                          cusparse_mean + t_c * stats.sem(cusparse_data))
            hbdia_ci = (hbdia_mean - t_h * stats.sem(hbdia_data),
                       hbdia_mean + t_h * stats.sem(hbdia_data))
            
            speedup = cusparse_mean / hbdia_mean
            # Approximate speedup CI using delta method
            cv_c = stats.sem(cusparse_data) / cusparse_mean
            cv_h = stats.sem(hbdia_data) / hbdia_mean
            speedup_se = speedup * np.sqrt(cv_c**2 + cv_h**2)
            speedup_ci = (speedup - 1.96*speedup_se, speedup + 1.96*speedup_se)
        
        results[noise] = {
            'cusparse_mean': cusparse_mean,
            'cusparse_ci': cusparse_ci,
            'hbdia_mean': hbdia_mean,
            'hbdia_ci': hbdia_ci,
            'speedup': speedup,
            'speedup_ci': speedup_ci
        }
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(noise_levels))
    width = 0.35
    
    # Runtime comparison
    cusparse_means = [results[n]['cusparse_mean'] for n in noise_levels]
    cusparse_cis = [results[n]['cusparse_ci'] for n in noise_levels]
    hbdia_means = [results[n]['hbdia_mean'] for n in noise_levels]
    hbdia_cis = [results[n]['hbdia_ci'] for n in noise_levels]
    
    cusparse_errs = [[m - ci[0] for m, ci in zip(cusparse_means, cusparse_cis)],
                     [ci[1] - m for m, ci in zip(cusparse_means, cusparse_cis)]]
    hbdia_errs = [[m - ci[0] for m, ci in zip(hbdia_means, hbdia_cis)],
                  [ci[1] - m for m, ci in zip(hbdia_means, hbdia_cis)]]
    
    ax1.bar(x - width/2, cusparse_means, width, yerr=cusparse_errs,
            label='cuSPARSE', capsize=5, alpha=0.8, color='#2E86AB')
    ax1.bar(x + width/2, hbdia_means, width, yerr=hbdia_errs,
            label='HBDIA', capsize=5, alpha=0.8, color='#A23B72')
    
    ax1.set_xlabel('Noise Level (%)')
    ax1.set_ylabel('Runtime (ms)')
    ax1.set_title(f'Runtime Comparison (95% {method})')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{n*100:.1f}%" if n < 0.01 else f"{n*100:.0f}%" for n in noise_levels])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Speedup plot
    speedups = [results[n]['speedup'] for n in noise_levels]
    speedup_cis = [results[n]['speedup_ci'] for n in noise_levels]
    speedup_errs = [[s - ci[0] for s, ci in zip(speedups, speedup_cis)],
                    [ci[1] - s for s, ci in zip(speedups, speedup_cis)]]
    
    ax2.bar(x, speedups, yerr=speedup_errs, capsize=5, alpha=0.8, color='#F18F01')
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
    ax2.set_xlabel('Noise Level (%)')
    ax2.set_ylabel('Speedup (cuSPARSE/HBDIA)')
    ax2.set_title(f'HBDIA Performance Advantage (95% {method})')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{n*100:.1f}%" if n < 0.01 else f"{n*100:.0f}%" for n in noise_levels])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output = '/users/nrottste/HBDIA/benchmarking/Plotting/hbdia_vs_cusparse_runtime'
    plt.savefig(f'{output}.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{output}.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    print(f"Average speedup: {np.mean(speedups):.2f}x")
    print(f"Max speedup: {max(speedups):.2f}x")

if __name__ == "__main__":
    main()
