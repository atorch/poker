#!/usr/bin/env python3
"""
Quick test to verify the new diagnostics are working.
Runs just 110 episodes to see:
- P(bet|AA) in every-10-episodes logging
- Frozen evaluation at episode 100
"""

from poker.play import run_sarsa

if __name__ == "__main__":
    # Run just 110 episodes (enough to see episode 100 checkpoint)
    run_sarsa(
        n_players=3,
        n_episodes=110,
        curriculum_random_episodes=110,  # Stay in random phase
        curriculum_mixed_episodes=0
    )
