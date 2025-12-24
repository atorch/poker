#!/usr/bin/env python3
"""
Hyperparameter grid search for poker agent training.

Searches over learning rate, batch size, network architecture, etc.
Saves results incrementally to avoid losing progress if interrupted.
"""

import json
import os
import itertools
from datetime import datetime
from pathlib import Path
import numpy as np

from poker.play import run_sarsa
from poker.agent import Agent
from poker.random_agent import RandomAgent
from poker.skillful_random_agent import SkillfulRandomAgent
from poker.play import evaluate_frozen_agent, get_pocket_aces_bet_probability


# ============================================================================
# Grid Search Configuration
# ============================================================================

GRID_PARAMS = {
    'learning_rate': [0.0001, 0.0003, 0.001, 0.003],
    'batch_size': [1],  # Start with 1 (online learning), add replay buffer later
    'network_size': [(32, 32), (64, 64), (128, 128), (64, 64, 64)],
    'n_episodes': [300, 500],  # Shorter runs to explore more configurations
    'epsilon_decay_rate': [100, 200, 400],
    'temperature': [0.5, 1.0, 2.0],
}

# For initial exploration, use a smaller grid
QUICK_GRID_PARAMS = {
    'learning_rate': [0.0001, 0.0003, 0.001],
    'batch_size': [1],
    'network_size': [(64, 64)],
    'n_episodes': [300],
    'epsilon_decay_rate': [200],
    'temperature': [1.0],
}

# Metrics to track at each checkpoint (every 100 episodes)
CHECKPOINT_EPISODES = [100, 200, 300, 400, 500]

# Output configuration
RESULTS_DIR = Path("grid_search_results")
RESULTS_FILE = RESULTS_DIR / "results.jsonl"  # JSON Lines format for incremental writing
PROGRESS_FILE = RESULTS_DIR / "progress.json"


# ============================================================================
# Helper Functions
# ============================================================================

def evaluate_agent_at_checkpoint(agent, n_players=3, episode=0):
    """
    Evaluate agent at a checkpoint and return all success metrics.

    Returns:
        dict with keys:
            - frozen_win_rate_vs_random: float
            - frozen_win_rate_vs_skillful: float
            - prob_bet_pocket_aces: float
            - sanity_checks_passed: bool
            - episode: int
    """
    # Metric 1: Frozen win rate vs RandomAgent
    random_opponents = [
        RandomAgent(player_index=i, actions=agent.actions)
        for i in range(1, n_players)
    ]
    results_random = evaluate_frozen_agent(
        agent, random_opponents, n_episodes=100, max_deals=5000
    )

    # Metric 2: Frozen win rate vs SkillfulRandomAgent
    skillful_opponents = [
        SkillfulRandomAgent(player_index=i, actions=agent.actions)
        for i in range(1, n_players)
    ]
    results_skillful = evaluate_frozen_agent(
        agent, skillful_opponents, n_episodes=100, max_deals=5000
    )

    # Metric 3: P(bet|AA)
    prob_bet_aa = get_pocket_aces_bet_probability(agent)

    # Metric 4: Sanity checks (simplified - just check if premium hands prefer betting)
    # We'll use prob_bet_aa as a proxy: >0.85 = pass
    sanity_pass = prob_bet_aa > 0.85

    return {
        'episode': episode,
        'frozen_win_rate_vs_random': results_random['win_rate'],
        'frozen_win_rate_vs_skillful': results_skillful['win_rate'],
        'prob_bet_pocket_aces': prob_bet_aa,
        'sanity_checks_passed': sanity_pass,
    }


def run_single_configuration(config, config_id):
    """
    Run training with a single hyperparameter configuration.

    Args:
        config: dict with hyperparameters
        config_id: unique identifier for this configuration

    Returns:
        dict with results
    """
    print("\n" + "=" * 80)
    print(f"Running configuration {config_id}")
    print(f"  Config: {config}")
    print("=" * 80)

    # Create a unique directory for this configuration's models
    config_dir = RESULTS_DIR / f"config_{config_id}"
    config_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(config_dir / "model.h5")

    # Determine checkpoint episodes based on n_episodes
    n_episodes = config['n_episodes']
    checkpoints = [ep for ep in CHECKPOINT_EPISODES if ep <= n_episodes]

    # Run training with the specified hyperparameters
    try:
        print(f"  Training for {n_episodes} episodes...")
        print(f"    Learning rate: {config['learning_rate']}")
        print(f"    Network size: {config['network_size']}")
        print(f"    Epsilon decay: {config['epsilon_decay_rate']}")
        print(f"    Temperature: {config['temperature']}")

        # Run SARSA training
        run_sarsa(
            n_players=3,
            n_episodes=n_episodes,
            model_path=model_path,
            save_interval=100,  # Save every 100 episodes for checkpoints
            curriculum_random_episodes=n_episodes,  # Stay in random phase only
            curriculum_mixed_episodes=0,
            learning_rate=config['learning_rate'],
            hidden_layers=config['network_size'],
            epsilon_decay_rate=config['epsilon_decay_rate'],
            temperature=config['temperature'],
        )

        # Evaluate at each checkpoint to track policy evolution
        print(f"\n  Evaluating agent at checkpoints: {checkpoints}...")
        checkpoint_results = []

        for checkpoint_ep in checkpoints:
            # Determine which model file to load
            if checkpoint_ep == n_episodes:
                # Final model
                checkpoint_model_path = model_path
            else:
                # Intermediate checkpoint (e.g., model_ep100.h5, model_ep200.h5)
                model_dir = os.path.dirname(model_path)
                model_name = os.path.splitext(os.path.basename(model_path))[0]
                checkpoint_model_path = os.path.join(model_dir, f"{model_name}_ep{checkpoint_ep}.h5")

            # Check if checkpoint exists
            if not os.path.exists(checkpoint_model_path):
                print(f"    ⚠️  Checkpoint at episode {checkpoint_ep} not found: {checkpoint_model_path}")
                continue

            # Load agent with checkpoint weights
            print(f"\n  Loading checkpoint at episode {checkpoint_ep}...")
            agent = Agent(
                player_index=0,
                n_players=3,
                temperature=config['temperature'],
                learning_rate=config['learning_rate'],
                hidden_layers=config['network_size']
            )
            agent.load_model(checkpoint_model_path)

            # Evaluate this checkpoint
            print(f"    Evaluating...")
            metrics = evaluate_agent_at_checkpoint(agent, n_players=3, episode=checkpoint_ep)
            checkpoint_results.append(metrics)
            print(f"    Frozen win rate (random): {100*metrics['frozen_win_rate_vs_random']:.1f}%")
            print(f"    Frozen win rate (skillful): {100*metrics['frozen_win_rate_vs_skillful']:.1f}%")
            print(f"    P(bet|AA): {100*metrics['prob_bet_pocket_aces']:.1f}%")

        # Aggregate metrics
        final_metrics = checkpoint_results[-1] if checkpoint_results else {}

        # Calculate robustness (std dev of frozen win rate across checkpoints)
        win_rates = [m['frozen_win_rate_vs_random'] for m in checkpoint_results]
        robustness = np.std(win_rates) if len(win_rates) > 1 else 0.0

        # Calculate efficiency (episodes to reach 40% frozen win rate)
        episodes_to_40pct = None
        for metrics in checkpoint_results:
            if metrics['frozen_win_rate_vs_random'] >= 0.40:
                episodes_to_40pct = metrics['episode']
                break

        # Compile results
        result = {
            'config_id': config_id,
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'status': 'completed',
            'checkpoint_metrics': checkpoint_results,
            'final_frozen_win_rate_vs_random': final_metrics.get('frozen_win_rate_vs_random', 0.0),
            'final_frozen_win_rate_vs_skillful': final_metrics.get('frozen_win_rate_vs_skillful', 0.0),
            'final_prob_bet_pocket_aces': final_metrics.get('prob_bet_pocket_aces', 0.0),
            'final_sanity_checks_passed': final_metrics.get('sanity_checks_passed', False),
            'robustness_std_win_rate': robustness,
            'episodes_to_40pct_win_rate': episodes_to_40pct,
        }

        print(f"\n✓ Configuration {config_id} completed successfully")
        print(f"  Final frozen win rate (random): {100*result['final_frozen_win_rate_vs_random']:.1f}%")
        print(f"  Robustness (std): {100*robustness:.1f}%")
        print(f"  Episodes to 40% win rate: {episodes_to_40pct}")

        return result

    except Exception as e:
        print(f"\n✗ Configuration {config_id} failed with error: {e}")
        import traceback
        traceback.print_exc()

        return {
            'config_id': config_id,
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'status': 'failed',
            'error': str(e),
        }


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    import numpy as np

    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def save_result(result):
    """Save a single result to JSONL file."""
    # Convert numpy types to Python native types
    result = convert_numpy_types(result)

    with open(RESULTS_FILE, 'a') as f:
        f.write(json.dumps(result) + '\n')
    print(f"  → Result saved to {RESULTS_FILE}")


def load_completed_configs():
    """Load set of completed configuration IDs from results file."""
    if not RESULTS_FILE.exists():
        return set()

    completed = set()
    with open(RESULTS_FILE, 'r') as f:
        for line in f:
            result = json.loads(line)
            if result.get('status') == 'completed':
                completed.add(result['config_id'])

    print(f"Found {len(completed)} completed configurations")
    return completed


def save_progress(current_idx, total):
    """Save progress to allow resuming."""
    progress = {
        'current_idx': current_idx,
        'total': total,
        'timestamp': datetime.now().isoformat(),
    }
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def generate_config_id(config):
    """Generate a unique, readable ID for a configuration."""
    # Create a short hash based on config values
    config_str = json.dumps(config, sort_keys=True)
    import hashlib
    short_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

    # Also include key parameters for readability
    lr = config['learning_rate']
    net = 'x'.join(map(str, config['network_size']))

    return f"lr{lr}_net{net}_{short_hash}"


# ============================================================================
# Main Grid Search
# ============================================================================

def run_grid_search(use_quick_grid=True):
    """
    Run grid search over hyperparameters.

    Args:
        use_quick_grid: If True, use smaller QUICK_GRID_PARAMS for initial exploration
    """
    # Setup
    RESULTS_DIR.mkdir(exist_ok=True)

    # Select grid
    params = QUICK_GRID_PARAMS if use_quick_grid else GRID_PARAMS

    # Generate all configurations
    param_names = sorted(params.keys())
    param_values = [params[name] for name in param_names]
    all_configs = [
        dict(zip(param_names, values))
        for values in itertools.product(*param_values)
    ]

    print(f"\n{'=' * 80}")
    print(f"GRID SEARCH: {'QUICK' if use_quick_grid else 'FULL'} MODE")
    print(f"{'=' * 80}")
    print(f"Total configurations to evaluate: {len(all_configs)}")
    print(f"Results will be saved to: {RESULTS_DIR}")
    print(f"{'=' * 80}\n")

    # Load completed configurations to support resuming
    completed_config_ids = load_completed_configs()

    # Run each configuration
    for idx, config in enumerate(all_configs):
        config_id = generate_config_id(config)

        # Skip if already completed
        if config_id in completed_config_ids:
            print(f"\n[{idx+1}/{len(all_configs)}] Skipping {config_id} (already completed)")
            continue

        print(f"\n[{idx+1}/{len(all_configs)}] Running {config_id}...")

        # Run training and evaluation
        result = run_single_configuration(config, config_id)

        # Save result immediately
        save_result(result)

        # Save progress
        save_progress(idx + 1, len(all_configs))

    print(f"\n{'=' * 80}")
    print(f"GRID SEARCH COMPLETED")
    print(f"{'=' * 80}")
    print(f"Results saved to: {RESULTS_FILE}")
    print(f"\nNext steps:")
    print(f"  1. Analyze results: python analyze_grid_search.py")
    print(f"  2. View best configurations by metric")
    print(f"  3. Run full training with best hyperparameters")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run hyperparameter grid search")
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run full grid search (default: quick mode with smaller grid)'
    )

    args = parser.parse_args()

    run_grid_search(use_quick_grid=not args.full)
