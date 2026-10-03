# Grid Search Implementation TODO

## Status: Framework Complete, Integration Needed

### ✅ What's Done:
1. **grid_search.py** - Main grid search script
   - Defines hyperparameter grid (quick and full modes)
   - Saves results incrementally to JSONL (resumable if interrupted)
   - Evaluates each configuration at checkpoints
   - Tracks all success metrics defined in README

2. **analyze_grid_search.py** - Results analysis script
   - Summary statistics
   - Best configurations by each metric
   - Sensitivity analysis (which hyperparameters matter most)
   - Visualizations (learning rate vs win rate, P(bet|AA) distribution, etc.)

3. **Success metrics tracked:**
   - Frozen evaluation win rate vs RandomAgent
   - Frozen evaluation win rate vs SkillfulRandomAgent
   - P(bet|AA) probability
   - Sanity checks (pass/fail)
   - Robustness (std dev of win rate across checkpoints)
   - Efficiency (episodes to reach 40% win rate)

### ⚠️ What Needs to be Done:

#### 1. Modify `run_sarsa()` to Accept Hyperparameters

Currently `run_sarsa()` in `poker/play.py` has these parameters:
```python
def run_sarsa(
    n_players,
    n_episodes=800,
    model_path="models/player_0_latest.h5",
    save_interval=10,
    max_deals=5_000,
    curriculum_random_episodes=500,
    curriculum_mixed_episodes=100,
):
```

**Need to add:**
```python
def run_sarsa(
    n_players,
    n_episodes=800,
    model_path="models/player_0_latest.h5",
    save_interval=10,
    max_deals=5_000,
    curriculum_random_episodes=500,
    curriculum_mixed_episodes=100,
    # NEW PARAMETERS:
    learning_rate=0.001,
    network_size=(64, 64, 64, 64),  # Tuple of layer sizes
    epsilon_decay_rate=200,
    temperature=1.0,
    # For future: batch_size, experience replay buffer
):
```

**Implementation notes:**
- Pass `learning_rate`, `temperature`, `network_size` to `Agent()` constructor
- Modify `get_model()` in `q_function.py` to accept `network_size` parameter
- Use `epsilon_decay_rate` in epsilon calculation (currently hardcoded as 200)
- For batch_size > 1: Need to implement experience replay buffer (future work)

#### 2. Modify `Agent()` Constructor

Currently:
```python
class Agent:
    def __init__(self, player_index=0, actions=[-1, 0, 1, 2, 3], n_players=3, temperature=1.0, learning_rate=0.001):
```

Already has `temperature` and `learning_rate` ✓

**Need to add:**
```python
class Agent:
    def __init__(
        self,
        player_index=0,
        actions=[-1, 0, 1, 2, 3],
        n_players=3,
        temperature=1.0,
        learning_rate=0.001,
        network_size=(64, 64, 64, 64),  # NEW
    ):
```

Then pass `network_size` to `get_model()`.

#### 3. Modify `get_model()` to Accept Network Architecture

Currently in `poker/q_function.py`:
```python
def get_model(n_actions, n_inputs, learning_rate=0.001):
    # Hardcoded: 4 layers of 64 units each
```

**Change to:**
```python
def get_model(n_actions, n_inputs, learning_rate=0.001, hidden_layers=(64, 64, 64, 64)):
    """
    Create Q-function neural network.

    Args:
        n_actions: Number of actions
        n_inputs: Input dimension
        learning_rate: Learning rate for Adam optimizer
        hidden_layers: Tuple of hidden layer sizes, e.g., (32, 32) or (64, 64, 64)
    """
    # Build layers dynamically based on hidden_layers tuple
```

#### 4. Update Grid Search to Use Real Training

In `grid_search.py`, replace the placeholder code:
```python
# Currently:
print("⚠️  WARNING: run_sarsa doesn't accept all hyperparameters yet")
agent = Agent(...)  # Dummy agent

# Change to:
run_sarsa(
    n_players=3,
    n_episodes=config['n_episodes'],
    model_path=model_path,
    curriculum_random_episodes=config['n_episodes'],
    curriculum_mixed_episodes=0,
    learning_rate=config['learning_rate'],
    network_size=config['network_size'],
    epsilon_decay_rate=config['epsilon_decay_rate'],
    temperature=config['temperature'],
)

# Then load the trained agent for evaluation
agent = Agent(...)
agent.load_model(model_path)
```

## Quick Start (After Integration):

```bash
# Run quick grid search (smaller parameter space)
python grid_search.py

# Run full grid search
python grid_search.py --full

# Analyze results
python analyze_grid_search.py
```

## Expected Output Structure:

```
grid_search_results/
├── results.jsonl              # All results (1 JSON per line, resumable)
├── results_summary.csv         # CSV for spreadsheet analysis
├── progress.json               # Current progress (for resuming)
├── plots/                      # Visualizations
│   ├── learning_rate_vs_win_rate.png
│   ├── prob_bet_aa_distribution.png
│   └── win_rate_vs_prob_bet_aa.png
└── config_<id>/                # Models for each configuration
    └── model.h5
```

## Estimated Runtime:

**Quick grid search** (current config):
- 3 learning rates × 1 network size × 1 n_episodes × 1 epsilon × 1 temp = 3 runs
- Each run: ~300 episodes + evaluations ≈ 5-10 minutes
- Total: ~15-30 minutes

**Full grid search:**
- 4 LR × 4 networks × 2 episodes × 3 epsilon × 3 temp = 288 runs
- Total: ~24-48 hours (can parallelize!)

## Parallelization:

To speed up full grid search, can run multiple configurations in parallel:
```python
# Use multiprocessing or run multiple instances:
python grid_search.py --config-range 0 72   # Process 1
python grid_search.py --config-range 72 144 # Process 2
python grid_search.py --config-range 144 216 # Process 3
python grid_search.py --config-range 216 288 # Process 4
```

(Requires adding `--config-range` parameter to grid_search.py)
