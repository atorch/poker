#!/usr/bin/env python3
"""
Analyze results from hyperparameter grid search.

Reads results from grid_search_results/results.jsonl and generates:
- Summary statistics
- Best configurations by each metric
- Sensitivity analysis (which hyperparameters matter most)
- Visualizations (if matplotlib available)
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np


RESULTS_FILE = Path("grid_search_results/results.jsonl")


def load_results():
    """Load all results from JSONL file into a pandas DataFrame."""
    if not RESULTS_FILE.exists():
        print(f"Error: Results file not found: {RESULTS_FILE}")
        print("Run grid_search.py first to generate results.")
        return None

    results = []
    with open(RESULTS_FILE, 'r') as f:
        for line in f:
            results.append(json.loads(line))

    print(f"Loaded {len(results)} results from {RESULTS_FILE}")

    # Filter to completed runs only
    completed = [r for r in results if r.get('status') == 'completed']
    failed = [r for r in results if r.get('status') == 'failed']

    print(f"  Completed: {len(completed)}")
    print(f"  Failed: {len(failed)}")

    if failed:
        print("\nFailed configurations:")
        for r in failed:
            print(f"  {r['config_id']}: {r.get('error', 'Unknown error')}")

    # Convert to DataFrame
    # Flatten config dict into columns
    rows = []
    for r in completed:
        row = {
            'config_id': r['config_id'],
            **r['config'],  # Unpack config dict
            'final_frozen_win_rate_vs_random': r['final_frozen_win_rate_vs_random'],
            'final_frozen_win_rate_vs_skillful': r['final_frozen_win_rate_vs_skillful'],
            'final_prob_bet_pocket_aces': r['final_prob_bet_pocket_aces'],
            'final_sanity_checks_passed': r['final_sanity_checks_passed'],
            'robustness_std_win_rate': r['robustness_std_win_rate'],
            'episodes_to_40pct_win_rate': r.get('episodes_to_40pct_win_rate'),
            'timestamp': r['timestamp'],
        }
        rows.append(row)

    if not rows:
        print("No completed results to analyze!")
        return None

    df = pd.DataFrame(rows)

    # Convert network_size from list to string for groupby operations
    if 'network_size' in df.columns:
        df['network_size'] = df['network_size'].apply(lambda x: str(tuple(x)) if isinstance(x, list) else str(x))

    return df


def print_summary(df):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    metrics = [
        'final_frozen_win_rate_vs_random',
        'final_frozen_win_rate_vs_skillful',
        'final_prob_bet_pocket_aces',
        'robustness_std_win_rate',
    ]

    for metric in metrics:
        values = df[metric]
        print(f"\n{metric}:")
        print(f"  Mean:   {values.mean():.4f}")
        print(f"  Std:    {values.std():.4f}")
        print(f"  Min:    {values.min():.4f}")
        print(f"  Max:    {values.max():.4f}")
        print(f"  Median: {values.median():.4f}")


def print_best_configs(df, metric, n=5):
    """Print top N configurations by a specific metric."""
    print(f"\n{'=' * 80}")
    print(f"TOP {n} CONFIGURATIONS BY: {metric}")
    print(f"{'=' * 80}")

    # Sort by metric (descending for most metrics, ascending for robustness_std)
    ascending = (metric == 'robustness_std_win_rate')
    sorted_df = df.sort_values(by=metric, ascending=ascending)

    top_n = sorted_df.head(n)

    for idx, (_, row) in enumerate(top_n.iterrows(), 1):
        print(f"\n{idx}. Config ID: {row['config_id']}")
        print(f"   {metric}: {row[metric]:.4f}")
        print(f"   Learning rate: {row['learning_rate']}")
        print(f"   Network size: {row['network_size']}")
        print(f"   N episodes: {row['n_episodes']}")
        print(f"   Epsilon decay: {row['epsilon_decay_rate']}")
        print(f"   Temperature: {row['temperature']}")
        print(f"   Metrics:")
        print(f"     - Win rate (random):  {100*row['final_frozen_win_rate_vs_random']:.1f}%")
        print(f"     - Win rate (skillful): {100*row['final_frozen_win_rate_vs_skillful']:.1f}%")
        print(f"     - P(bet|AA): {100*row['final_prob_bet_pocket_aces']:.1f}%")
        print(f"     - Sanity checks passed: {row['final_sanity_checks_passed']}")
        print(f"     - Robustness (std): {100*row['robustness_std_win_rate']:.1f}%")
        if row['episodes_to_40pct_win_rate'] is not None:
            print(f"     - Episodes to 40% win rate: {row['episodes_to_40pct_win_rate']}")


def analyze_hyperparameter_sensitivity(df):
    """Analyze which hyperparameters have the most impact on performance."""
    print(f"\n{'=' * 80}")
    print("HYPERPARAMETER SENSITIVITY ANALYSIS")
    print(f"{'=' * 80}")

    hyperparams = ['learning_rate', 'network_size', 'n_episodes', 'epsilon_decay_rate', 'temperature']
    metric = 'final_frozen_win_rate_vs_random'

    print(f"\nImpact on {metric}:")
    print(f"(Mean win rate for each hyperparameter value)\n")

    for hp in hyperparams:
        if hp not in df.columns:
            continue

        grouped = df.groupby(hp)[metric].agg(['mean', 'std', 'count'])
        print(f"\n{hp}:")
        print(grouped.to_string())

        # Calculate range (max - min) as a measure of sensitivity
        value_range = grouped['mean'].max() - grouped['mean'].min()
        print(f"  → Range: {100*value_range:.1f}% (larger = more sensitive)")


def plot_results(df):
    """Generate visualizations of grid search results."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        print("\nMatplotlib not available - skipping visualizations")
        print("Install with: pip install matplotlib seaborn")
        return

    print(f"\n{'=' * 80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'=' * 80}")

    output_dir = Path("grid_search_results/plots")
    output_dir.mkdir(exist_ok=True)

    # 1. Learning rate vs win rate
    if 'learning_rate' in df.columns and len(df['learning_rate'].unique()) > 1:
        plt.figure(figsize=(10, 6))
        grouped = df.groupby('learning_rate')['final_frozen_win_rate_vs_random'].agg(['mean', 'std'])
        # Replace NaN std with 0 for plotting (happens when only one sample per group)
        std_values = grouped['std'].fillna(0)
        plt.errorbar(grouped.index, grouped['mean'], yerr=std_values, marker='o', capsize=5)
        plt.xlabel('Learning Rate')
        plt.ylabel('Frozen Win Rate vs Random')
        plt.title('Learning Rate vs Performance')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.333, color='r', linestyle='--', label='Fair share (33.3%)')
        plt.axhline(y=0.40, color='g', linestyle='--', label='Target (40%)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'learning_rate_vs_win_rate.png', dpi=150)
        print(f"  Saved: {output_dir / 'learning_rate_vs_win_rate.png'}")
        plt.close()

    # 2. P(bet|AA) distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['final_prob_bet_pocket_aces'], bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('P(bet|AA)')
    plt.ylabel('Count')
    plt.title('Distribution of P(bet|AA) Across Configurations')
    plt.axvline(x=0.85, color='g', linestyle='--', label='Target (85%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'prob_bet_aa_distribution.png', dpi=150)
    print(f"  Saved: {output_dir / 'prob_bet_aa_distribution.png'}")
    plt.close()

    # 3. Win rate vs P(bet|AA) scatter
    plt.figure(figsize=(10, 6))
    plt.scatter(
        df['final_prob_bet_pocket_aces'],
        df['final_frozen_win_rate_vs_random'],
        alpha=0.6,
        s=100
    )
    plt.xlabel('P(bet|AA)')
    plt.ylabel('Frozen Win Rate vs Random')
    plt.title('Win Rate vs P(bet|AA) - Correlation Analysis')
    plt.axhline(y=0.40, color='g', linestyle='--', alpha=0.5, label='Target win rate')
    plt.axvline(x=0.85, color='g', linestyle='--', alpha=0.5, label='Target P(bet|AA)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'win_rate_vs_prob_bet_aa.png', dpi=150)
    print(f"  Saved: {output_dir / 'win_rate_vs_prob_bet_aa.png'}")
    plt.close()

    print(f"\nAll plots saved to: {output_dir}/")


def main():
    """Main analysis function."""
    df = load_results()

    if df is None or len(df) == 0:
        return

    print_summary(df)

    # Best configs by each metric
    print_best_configs(df, 'final_frozen_win_rate_vs_random', n=3)
    print_best_configs(df, 'final_prob_bet_pocket_aces', n=3)
    print_best_configs(df, 'robustness_std_win_rate', n=3)

    # Sensitivity analysis
    analyze_hyperparameter_sensitivity(df)

    # Visualizations
    plot_results(df)

    # Save summary to CSV for further analysis
    csv_path = Path("grid_search_results/results_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n{'=' * 80}")
    print(f"Summary saved to: {csv_path}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
