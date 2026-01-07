# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a poker AI research project implementing Deep Q-Learning (SARSA) with curriculum learning. See the README for project details, current achievements, and architecture information.

**Important**: For specific implementation details (hyperparameters, training phases, state representation, etc.), always refer to the actual code and README rather than relying on this document. This file focuses on workflow guidance for Claude Code.

## Development Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install dependencies
uv sync --all-extras

# Run tests
uv run pytest tests/

# Run specific test with verbose output
uv run pytest tests/test_agent.py -v -s

# Run training with default config (best hyperparameters from grid search)
uv run python poker/play.py

# Run training with output logging
uv run python run_training.py

# Play interactively against trained agents
uv run python interactive_play.py              # Normal mode
uv run python interactive_play.py --full-info  # See opponents' cards

# Grid search for hyperparameter tuning
uv run python grid_search.py --mode medium
```

### Git Workflow

**Important**:
- `git add` files as you work on them, but do NOT commit automatically
- Only create commits when explicitly asked by the user
- Commit messages should NOT mention Claude or Anthropic
- Write commit messages that describe the technical changes made

### Testing Workflow

**Critical**: After making changes to any code that has tests, ALWAYS run the relevant test suite with `uv run pytest` to verify the changes don't break existing functionality.

Example:
```bash
# After modifying agent.py
uv run pytest tests/test_agent.py -v

# After modifying state.py
uv run pytest tests/test_state.py -v

# Run all tests
uv run pytest tests/
```

## Code Architecture

For detailed architecture information, see the README and code comments. Key areas to understand:

**Core Components**:
- `poker/state.py`: Game state management, betting rounds, dealer rotation
- `poker/cards.py`, `poker/hands.py`: Card representation and hand evaluation
- `poker/config.py`: Action enums and wealth configuration
  - **CRITICAL**: Changing the `Action` enum requires retraining all models from scratch
- `poker/agent.py`: Deep Q-Network implementation with SARSA learning
- `poker/play.py`: Training pipeline with curriculum learning phases

**Key Concepts** (details in code):
- **Position invariance**: State representation uses relative positioning
- **Curriculum learning**: Multi-phase training (RandomAgent → SkillfulRandomAgent → Self-play)
- **Wealth conservation**: Zero-sum episodes maintain constant total wealth
- **Experience replay**: Batch learning for stability

**Design Principles**:
- Avoid over-training on RandomAgent (causes pathological strategies)
- Never use 100% self-play (causes catastrophic forgetting)
- Maintain opponent diversity throughout training
- See README and training logs for current best practices

## Testing Philosophy

See `tests/` directory for comprehensive test coverage. Key testing principles:

**Always run tests after changes**: Use `uv run pytest tests/` to verify changes don't break functionality

**Test categories** (see individual test files for details):
- Game dynamics and betting logic
- Position invariance and fairness
- Agent behavior and learning
- Hand evaluation and card mechanics

**Validation approach**: Tests include pre-training validation (position fairness), post-training validation (win rates vs different opponents), and sanity checks (Q-value ordering, hand strength preferences)

## Model Persistence

Models are saved as Keras `.h5` weights files in the `models/` directory.

**Important**: State representation changes invalidate old models. See code comments in `agent.py` for current state representation details.

## Additional Documentation

See root directory for research notes and training logs:
- `README.md`: Project overview and current status
- `*.md` files: Design decisions and debugging notes
- `training_output_*.txt`: Historical training run metrics
