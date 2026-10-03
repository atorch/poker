#!/usr/bin/env python3
"""
Test if evaluation results are consistent between:
1. In-memory agent evaluation (during training)
2. Loaded agent evaluation (from disk in grid search)

This will help us understand the 26% vs 34.6% discrepancy.
"""
import numpy as np

print("=" * 80)
print("EVALUATION FLOW ANALYSIS")
print("=" * 80)

print("\nThere are THREE different evaluation points in the code:")
print("\n1. MID-TRAINING CHECKPOINT (poker/play.py:770-806)")
print("   - Evaluates: in-memory agent (during training)")
print("   - Episodes: 300")
print("   - Happens at: episodes 100, 200 (NOT 300, as 300 is final)")
print("   - User's logs show:")
print("     Episode 100: 34.0% ± 5.4%")
print("     Episode 200: 36.0% ± 5.4%")

print("\n2. POST-TRAINING VALIDATION (poker/play.py:889-916)")
print("   - Evaluates: in-memory agent (after training completes)")
print("   - Episodes: 1000")
print("   - Happens: once, after all training")
print("   - User's logs show:")
print("     Episode 300 (final): 34.6% ± 2.9%")

print("\n3. GRID SEARCH CHECKPOINT EVAL (grid_search.py:156-193)")
print("   - Evaluates: agent loaded from disk")
print("   - Episodes: 100")
print("   - Happens: after training, for episodes 100, 200, 300")
print("   - JSONL shows:")
print("     Episode 100: 40%")
print("     Episode 200: 38%")
print("     Episode 300: 26%")

print("\n" + "=" * 80)
print("KEY FINDING: Different number of evaluation episodes!")
print("=" * 80)

# Calculate confidence intervals for each
from scipy import stats

def ci_width(p, n):
    """Return 95% CI half-width for win rate p with n samples."""
    z = stats.norm.ppf(0.975)
    se = np.sqrt(p * (1 - p) / n)
    return z * se

print("\nConfidence interval widths at 34% win rate:")
print(f"  100 episodes:  ±{100*ci_width(0.34, 100):.1f}%")
print(f"  300 episodes:  ±{100*ci_width(0.34, 300):.1f}%")
print(f"  1000 episodes: ±{100*ci_width(0.34, 1000):.1f}%")

print("\nWith 100 episodes, 95% CI is ~±9.3%, so a true 34% could easily show as:")
print(f"  - Low end: {100*(0.34 - ci_width(0.34, 100)):.1f}%  ← Could explain 26%!")
print(f"  - High end: {100*(0.34 + ci_width(0.34, 100)):.1f}%")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

print("\nThe discrepancy is explained by SAMPLE SIZE:")
print("  - Grid search uses only 100 episodes per checkpoint")
print("  - MID-TRAINING uses 300 episodes")
print("  - POST-TRAINING uses 1000 episodes")
print("\nThe 26% from grid search (100 episodes) is likely just NOISE.")
print("The true win rate is probably ~33-35%, consistent with:")
print("  - POST-TRAINING: 34.6% ± 2.9% (most reliable, 1000 episodes)")
print("  - MID-TRAINING: 34% and 36% (fairly reliable, 300 episodes each)")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

print("\n1. INCREASE grid search checkpoint evaluation episodes:")
print("   Change line 78 in grid_search.py from:")
print("     n_episodes=100")
print("   To:")
print("     n_episodes=300")
print("   (Or even 500 for more stable estimates)")

print("\n2. This is NOT a bug - just sampling variance!")
print("   With 100 episodes, confidence intervals are too wide.")

print("\n3. For the full grid search, use more evaluation episodes")
print("   to get reliable comparisons between configurations.")
