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

# Benchmark an agent (duplicate-dealt chips/deal, style stats, M1 check, behaviour fingerprint)
uv run python -m poker.benchmark
uv run python -m poker.benchmark --agent tag --decks 5000 --games 0 --json /tmp/tag.json

# Train the new DQN learner (TICKET_learner_redesign.md); writes runs/<date>_<name>/
uv run python -m poker.train_dqn --name mc_callstation --opponents callstation --decisions 400000

# Summarize / compare training runs (run directories under runs/)
uv run python -m poker.runs list
uv run python -m poker.runs summarize runs/<run>
uv run python -m poker.runs compare runs/<run_a> runs/<run_b>

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
  - **All betting rules live here**: agents and the UI must get legal moves from `State.legal_actions()` / `State.is_legal()` rather than re-deriving them
- `poker/cards.py`, `poker/hands.py`: Card representation and hand evaluation
- `poker/config.py`: Action enums and wealth configuration
  - **CRITICAL**: Changing the `Action` enum requires retraining all models from scratch
- `poker/agent.py`: Deep Q-Network implementation with SARSA learning
- `poker/play.py`: Training pipeline with curriculum learning phases (old learner)
- New learner (TICKET_learner_redesign.md): `poker/features.py` (state -> features), `poker/actions.py`
  (relative action heads), `poker/env.py` (vectorized single-deal environment), `poker/dqn.py`
  (PyTorch Q-network, Monte-Carlo replay, `DQNAgent`), `poker/train_dqn.py` (training loop)
- Evaluation: `poker/benchmark.py`, `poker/diagnostics.py`, `poker/preflop_equity.py`; runs: `poker/runs.py`

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

## Working with Training Runs (context budget)

Training output is large; keep it out of the conversation:
- **New trainers write a run directory** with `poker.runs.RunLogger` (config, `metrics.jsonl`,
  `evals.jsonl`, warnings, checkpoints) and print at most one short line per logging interval.
  Don't add per-episode or per-checkpoint prints; log to `metrics.jsonl` instead.
- **Read runs with `python -m poker.runs summarize|compare|list`** (about 25 lines per run), never by
  `cat`-ing logs or JSONL files. For trends, `python -m poker.runs plot <run>` writes a PNG that can be
  viewed with the Read tool. If a summary is missing something, add it to `poker/runs.py` rather than
  reading raw files.
- **Long runs go in the background** (Bash `run_in_background`), with a budget (steps or minutes), and
  are checked with `summarize`. The old `poker/play.py` / `run_training.py` output is very verbose
  (about 55k tokens per run); grep its log for specific lines if it must be used.
- **Benchmarks:** `poker.benchmark.run_benchmark` returns a JSON-serializable report; `format_report`
  prints about 15 lines. Use `--games 0` to skip the slow, noisy bust-out metric, and the same
  `--seed` for paired comparisons (decks depend only on the seed and opponent).
- **Claims need several seeds** (results varied a lot across seeds in the past). Runs that differ only
  by seed are grouped by `compare`.
- **Subagents running experiments** should return only the `summarize`/`compare` output plus at most
  five lines of interpretation, never raw logs.

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

Two model formats coexist while the learner redesign is in progress:
- **New DQN models (`poker/dqn.py`, PyTorch):** `<stem>.pt` (weights) plus `<stem>.json` (feature
  version, action heads, layer sizes). `load_network` checks the JSON and fails loudly on a mismatch,
  so **bump `FEATURE_VERSION` in `poker/features.py` whenever the encoding changes**. Training runs put
  them in `runs/<run>/final.pt` and `runs/<run>/checkpoints/`.
- **Old models (`poker/agent.py`, Keras 3 / TensorFlow):** `.weights.h5` files in `models/`
  (`Agent.load_model` also reads legacy Keras 2 `.h5` weights). State representation changes
  invalidate them. TensorFlow is only needed for these, until the old learner moves to `archive/`.

`poker.benchmark` and `poker.env` accept either kind of model path wherever an agent or opponent is expected.

## Additional Documentation

See root directory for research notes and training logs:
- `README.md`: Project overview and current status
- `LAY_OF_THE_LAND.md`: Current assessment and next steps
- `archive/`: Old notes, scripts, grid search results and training logs (predate the Oct 2026 rules/evaluator fixes)
- `logs/`: Training run logs written by `run_training.py` (gitignored)
