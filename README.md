# poker

A research sandbox for training reinforcement-learning agents to play 3-player Texas hold'em. The goal is an agent that is genuinely strong and fun to play against, and to learn along the way. The current agent is a small neural Q-function trained with SARSA-style updates and a curriculum of scripted opponents and self-play.

## Quick start

This project uses [uv](https://docs.astral.sh/uv/). Dependencies (Python 3.12, TensorFlow 2.21 / Keras 3) are installed into a repo-local `.venv`.

```bash
# Install dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Train (logs to logs/, models to models/)
uv run python run_training.py short_description

# Play against trained agents
uv run python interactive_play.py              # opponents' cards hidden
uv run python interactive_play.py --full-info  # see opponents' cards and the AI's action probabilities

# Benchmark a model, or a baseline bot for comparison
uv run python -m poker.benchmark
uv run python -m poker.benchmark --agent maxraise
uv run python -m poker.benchmark --model models/some_run/player_0_latest.weights.h5
```

## Status (Oct 2026)

The game engine was recently fixed in several ways that affect every earlier result:
- The hand evaluator now applies kickers and full tiebreaks. Before, about 16% of 3-way showdowns paid the wrong players.
- Betting rules changed (see below). Before, hands tended to escalate to all-in, and players who were all in could still fold.
- Several training-loop bugs were fixed, notably replay targets that bootstrapped from illegal actions and from terminal states.

As a result, the training results from Dec 2025 – Jan 2026 are not comparable to new runs, and they have been removed from this README (the logs are in `archive/`).

Where things stand under the new rules:
- The Jan 2026 model (`models/player_0_latest.h5`, the default for interactive play) beats the random baselines but is no better than a bot that always bets the maximum.
- A model retrained with the fixed code learned less: it is close to RandomAgent's results against the same opponents. It is in `models/2026-10-02_rules_fixed/`. Our current reading is that the old model's aggression came mostly from the replay-target bug, and that the learner itself needs design changes (see Next steps).

See `LAY_OF_THE_LAND.md` for the full assessment and benchmark tables.

## Game rules

- 3 players, blinds of $1/$2, starting stacks of $5–$35 during training. A game ends when any player runs out of chips.
- Each action puts in $0–$3 (`Action` in `poker/config.py`). An amount equal to what you owe is a call; more than that is a raise.
- Bets are capped so that nobody can go past all in, so there are no side pots.
- At most `MAX_RAISES_PER_STAGE` (4) voluntary bets/raises per betting round. Forced blinds don't count.
- Folding is only legal when facing a bet.
- Once nobody can bet any more, the remaining cards are dealt and the hand goes straight to showdown.
- Split pots are paid in whole chips; odd chips go to the winner closest to the dealer's left.

All rules are enforced in `poker/state.py` (`State.is_legal`, `State.legal_actions`). Agents and the UI get their legal moves from there.

## How it works

- **State** (`Agent.get_private_state`): game stage, hole cards, own stack, own bets, pot, the five public cards (-1 if not yet dealt), opponents' stacks and whether each is still in the hand. Opponents are ordered by seat relative to the player, so the representation doesn't depend on absolute seat.
- **Q-function** (`poker/q_function.py`): an MLP that takes the state plus the action as a single scalar input and outputs one Q-value.
- **Policy**: softmax over the Q-values of legal actions (temperature 1), plus epsilon-greedy exploration early in training.
- **Training** (`poker/play.py`): optional pre-training toward a hand-crafted heuristic, then SARSA updates from an experience replay buffer (Expected SARSA targets over legal actions). Opponents follow a curriculum: scripted random agents first, then a mix of self-play and scripted agents.
- **Scripted opponents**: `RandomAgent` (uniform over legal actions), `SkillfulRandomAgent` (rarely folds strong starting hands, often folds weak ones), `ConsistentRandomAgent` (commits to a random style for each deal), and the card-blind `MaxRaiseAgent` and `CallStationAgent` in `poker/baseline_agents.py`.

## Evaluation

`poker/benchmark.py` reports two numbers against each opponent type:
- **Chips per deal, duplicate**: each shuffled deck is replayed with the agent in every seat, starting from fixed stacks. This cancels most of the card luck. 0 is break-even.
- **Bust-out win rate**: the fraction of full games where the agent finishes with the most chips. 33% is break-even.

Win rates against RandomAgent and SkillfulRandomAgent mostly reward aggression, because those bots fold often. The card-blind MaxRaise and CallStation bots are a better bar: neither can beat the other, so an agent that beats both is using its cards.

## Observations from earlier experiments

These come from runs before the engine and training fixes, so treat them as hypotheses to re-test rather than conclusions:
- A softmax policy seemed to behave better than argmax, which sometimes collapsed into folding most hands.
- Pure self-play appeared prone to collapse. Keeping some scripted opponents in the mix seemed to help.
- Training outcomes varied a lot across random seeds with identical settings.
- Experience replay (batch size 8) looked more stable than one update per step. Larger networks (256×256) were less stable than 64×64 or 128×128.
- Training only against RandomAgent tends to reward "bet until they fold" rather than hand selection.

## Next steps

Candidates, roughly in order (details in `LAY_OF_THE_LAND.md`):
- **Learner design**: one Q output per action instead of a scalar action input; one deal per episode with reward equal to that deal's chip change; normalized inputs; a target network; more updates per episode; reconsider or drop heuristic pre-training.
- **Features**: hand-strength or equity estimates, pot odds, and position.
- **Actions**: consider relative actions (fold / call / raise by k), or pot-fraction bet sizes.
- **Stronger reference opponents**: for example tabular CFR on an abstracted version of the game, which would also allow measuring exploitability.
- **Search**: decision-time search guided by the learned value function (as in Expert Iteration, arXiv:1705.08439), adapted for hidden information.
- **More players**: generalize training and the state representation to any number of players from 2 up to some cap.

## Repository layout

- `poker/`: game engine (`state.py`, `cards.py`, `hands.py`), agents, training (`play.py`), benchmark
- `tests/`: unit tests, including rule tests and an end-to-end training smoke test
- `interactive_play.py`: play against the AI in the terminal
- `run_training.py`, `grid_search.py`, `analyze_grid_search.py`: training and hyperparameter search scripts
- `models/`: saved weights (gitignored)
- `logs/`: training logs (gitignored)
- `archive/`: older notes, scripts, grid search results and training logs from before the Oct 2026 fixes
