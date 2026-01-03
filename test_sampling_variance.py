#!/usr/bin/env python3
"""
Test whether the discrepancy between 26% (100 episodes) and 34.6% (1000 episodes)
could be due to sampling variation.
"""
import numpy as np
from scipy import stats

def calculate_ci(win_rate, n_episodes, confidence=0.95):
    """Calculate confidence interval for a win rate."""
    z = stats.norm.ppf((1 + confidence) / 2)
    se = np.sqrt(win_rate * (1 - win_rate) / n_episodes)
    margin = z * se
    return (win_rate - margin, win_rate + margin)

def test_if_same_distribution(p1, n1, p2, n2):
    """
    Test if two observed proportions could come from the same underlying distribution.

    Uses a two-proportion z-test.
    """
    # Pooled proportion
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)

    # Standard error
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

    # Z-statistic
    z = (p1 - p2) / se

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return z, p_value

def simulate_sampling_variance():
    """
    Simulate the scenario: true win rate is unknown, we observe:
    - 26% from 100 samples
    - 34.6% from 1000 samples

    Test if these could plausibly come from the same underlying distribution.
    """
    print("=" * 80)
    print("SAMPLING VARIANCE ANALYSIS")
    print("=" * 80)

    # Observed data
    win_rate_100 = 0.26
    n_100 = 100

    win_rate_1000 = 0.346
    n_1000 = 1000

    # Calculate confidence intervals
    ci_100 = calculate_ci(win_rate_100, n_100)
    ci_1000 = calculate_ci(win_rate_1000, n_1000)

    print("\nObserved data:")
    print(f"  Checkpoint eval (100 episodes):  {100*win_rate_100:.1f}% ± {100*(ci_100[1]-win_rate_100):.1f}% (95% CI: [{100*ci_100[0]:.1f}%, {100*ci_100[1]:.1f}%])")
    print(f"  Post-training eval (1000 episodes): {100*win_rate_1000:.1f}% ± {100*(ci_1000[1]-win_rate_1000):.1f}% (95% CI: [{100*ci_1000[0]:.1f}%, {100*ci_1000[1]:.1f}%])")

    # Check if confidence intervals overlap
    overlap = ci_100[1] >= ci_1000[0] and ci_1000[1] >= ci_100[0]
    print(f"\nConfidence intervals overlap: {overlap}")
    if overlap:
        print("  → The two estimates are statistically consistent (CIs overlap)")
    else:
        print("  → The two estimates are statistically inconsistent (CIs don't overlap)")

    # Two-proportion z-test
    z, p_value = test_if_same_distribution(win_rate_100, n_100, win_rate_1000, n_1000)
    print(f"\nTwo-proportion z-test:")
    print(f"  Z-statistic: {z:.3f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Significance at α=0.05: {'YES - significantly different' if p_value < 0.05 else 'NO - not significantly different'}")

    # Simulate: if true win rate is 34.6%, what's the probability of observing ≤26% in 100 episodes?
    print(f"\n--- Scenario A: True win rate is 34.6% ---")
    true_p = 0.346
    se_100 = np.sqrt(true_p * (1 - true_p) / n_100)
    z_score = (win_rate_100 - true_p) / se_100
    prob_a = stats.norm.cdf(z_score)
    print(f"  If true win rate = 34.6%, probability of observing ≤26% in 100 episodes: {100*prob_a:.2f}%")
    if prob_a > 0.05:
        print(f"  → This is plausible (>{100*0.05:.0f}% chance)")
    else:
        print(f"  → This is unlikely (<{100*0.05:.0f}% chance)")

    # Simulate: if true win rate is 26%, what's the probability of observing ≥34.6% in 1000 episodes?
    print(f"\n--- Scenario B: True win rate is 26% ---")
    true_p = 0.26
    se_1000 = np.sqrt(true_p * (1 - true_p) / n_1000)
    z_score = (win_rate_1000 - true_p) / se_1000
    prob_b = 1 - stats.norm.cdf(z_score)
    print(f"  If true win rate = 26%, probability of observing ≥34.6% in 1000 episodes: {100*prob_b:.2e}%")
    if prob_b > 0.05:
        print(f"  → This is plausible (>{100*0.05:.0f}% chance)")
    else:
        print(f"  → This is VERY unlikely (<{100*0.05:.0f}% chance)")

    # Estimate the most likely true win rate using weighted average
    # Weight by precision (inverse variance)
    var_100 = win_rate_100 * (1 - win_rate_100) / n_100
    var_1000 = win_rate_1000 * (1 - win_rate_1000) / n_1000
    w_100 = 1 / var_100
    w_1000 = 1 / var_1000

    weighted_mean = (w_100 * win_rate_100 + w_1000 * win_rate_1000) / (w_100 + w_1000)
    print(f"\n--- Weighted estimate of true win rate ---")
    print(f"  Precision-weighted mean: {100*weighted_mean:.1f}%")
    print(f"  (This heavily weights the 1000-episode sample, which has ~10x lower variance)")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    if p_value < 0.05:
        print("The difference between 26% and 34.6% is STATISTICALLY SIGNIFICANT.")
        print("This suggests:")
        print("  1. Possible bug in evaluation code (different settings/conditions)")
        print("  2. Agent performance actually changed between evaluations")
        print("  3. There's an issue with model saving/loading")
        print("\nRECOMMENDATION: Investigate for bugs!")
    else:
        print("The difference between 26% and 34.6% is NOT statistically significant.")
        print("This is consistent with sampling variance.")
        print("\nRECOMMENDATION: This is likely just noise. To reduce variance:")
        print("  - Increase checkpoint evaluation episodes (e.g., 300-500 instead of 100)")
        print("  - Or accept that checkpoint evals are noisy indicators")

    return p_value

if __name__ == "__main__":
    p_value = simulate_sampling_variance()
