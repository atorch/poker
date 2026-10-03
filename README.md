# poker

A research sandbox for training reinforcement-learning agents to play 3-player Texas hold'em. The goal is an agent that is genuinely strong and fun to play against, and to learn along the way. The current agent is a DQN-style learner (PyTorch): a small network with one Q-value per action, trained on single deals against a pool of scripted opponents. The older SARSA learner (Keras) is kept for comparison.

## Quick start

This project uses [uv](https://docs.astral.sh/uv/). Dependencies (Python 3.12, PyTorch for the new learner, TensorFlow 2.21 / Keras 3 for the old one) are installed into a repo-local `.venv`. PyTorch comes from the CPU-only wheel index configured in `pyproject.toml`.

```bash
# Install dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Train the new DQN learner (run directory in runs/; see TICKET_learner_redesign.md)
uv run python -m poker.train_dqn --name mc_callstation --opponents callstation --decisions 400000

# Train the old SARSA learner (logs to logs/, models to models/)
uv run python run_training.py short_description

# Play against trained agents
uv run python interactive_play.py              # opponents' cards hidden (new TD agent by default)
uv run python interactive_play.py --full-info  # see opponents' cards and the AI's action probabilities and Q-values
uv run python interactive_play.py --temperature 0.3            # a less predictable new agent (softmax over Q in BB)
uv run python interactive_play.py --model models/player_0_latest.h5   # the old Jan 2026 model, or --model tag

# Benchmark a model, or a baseline bot for comparison
uv run python -m poker.benchmark
uv run python -m poker.benchmark --agent maxraise
uv run python -m poker.benchmark --agent models/some_run/player_0_latest.weights.h5 --decks 5000 --games 0
uv run python -m poker.benchmark --opponents maxraise tag models/old.weights.h5   # models as opponents

# Look at training runs written with poker.runs.RunLogger
uv run python -m poker.runs list
uv run python -m poker.runs summarize runs/<run>
uv run python -m poker.runs compare runs/<run_a> runs/<run_b>
uv run python -m poker.runs plot runs/<run>      # learning curves as a PNG
```

## Status (Oct 2026)

The game engine was recently fixed in several ways that affect every earlier result:
- The hand evaluator now applies kickers and full tiebreaks. Before, about 16% of 3-way showdowns paid the wrong players.
- It also no longer scores A A 2 3 4 or A A A 2 3 as an ace-high straight (the ace only plays low in the wheel). That bug changed the winners in about 0.3% of 3-way showdowns. Every five-card hand class is now cross-checked against an independent evaluator in the tests.
- Betting rules changed (see below). Before, hands tended to escalate to all-in, and players who were all in could still fold.
- Several training-loop bugs were fixed, notably replay targets that bootstrapped from illegal actions and from terminal states.

As a result, the training results from Dec 2025 – Jan 2026 are not comparable to new runs, and they have been removed from this README (the logs are in `archive/`).

Where things stand (Oct 3, 2026):
- **The new learner** (`poker/train_dqn.py`, see `TICKET_learner_redesign.md`), trained for 1M decisions (about 5 minutes) against all seven scripted bots drawn at random, **beats 6 of them consistently across seeds**. In chips per deal against two copies of each: Random +3.1, Skillful +1.5, Consistent +0.6, CallStation +1.3, `tag` +0.35, `tag-calldown` +0.24. The exception is MaxRaise (about −0.9): in the mix, a raise usually means a strong hand, and the agent can't tell a maniac apart. It is the default for interactive play (`models/2026-10-03_td_pool/final.pt`).
- **Temporal-difference targets were the key.** With Monte-Carlo targets the learner collapses into folding against MaxRaise.
- **The Jan 2026 model** (`models/player_0_latest.h5`) beats the random baselines but is no better than a bot that always bets the maximum. A retrain of the old learner with the fixed code (`models/2026-10-02_rules_fixed/`) is close to RandomAgent.

See `TICKET_learner_redesign.md` (Progress) for the experiments, and `LAY_OF_THE_LAND.md` for the earlier assessment.

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

The new learner (`poker/features.py`, `actions.py`, `env.py`, `dqn.py`, `train_dqn.py`):
- **Features**: 172 numbers from the acting player's point of view.
  - Cards: stage, hole cards and board as 52-dim binary vectors, and the made-hand category (and whether it improves on the board).
  - Preflop equity of the hole cards, from a precomputed table.
  - Money in big blinds: pot, amount owed, pot odds, stacks behind, chips committed.
  - Position, raises this street, and one block per opponent in relative seat order, padded to 6 players.
- **Actions**: relative heads (fold, check/call, raise by 1/2/3) that mean the same thing in every spot. They map onto the game's chip amounts, so the rules are unchanged.
- **Q-network**: an MLP with one output per learned head, in big blinds. Q(fold) is not learned: it is exactly −(chips already committed).
- **Training**:
  - Each episode is one deal, with random per-player stacks, dealer and seat. The reward is the deal's chip change.
  - 256 tables run in parallel with batched network calls, and the policy is ε-greedy.
  - Targets are Expected SARSA (temporal difference, with a target network) by default, or Monte Carlo.
  - Opponents come from a pool or a curriculum of pools.
  - Runs are written to `runs/` (see `poker/runs.py`).

The old learner (`poker/agent.py`, `play.py`): an MLP taking the state plus the action as a single scalar input, a softmax policy, SARSA updates from replay, and a curriculum of scripted agents and self-play.

- **Scripted opponents**: `RandomAgent` (uniform over legal actions), `SkillfulRandomAgent` (rarely folds strong starting hands, often folds weak ones), `ConsistentRandomAgent` (commits to a random style for each deal), and in `poker/baseline_agents.py` the card-blind `MaxRaiseAgent` and `CallStationAgent` plus `TagAgent`, a crude rule-based player that uses its cards and the board (strong hands bet the maximum, medium hands check or call, weak hands check or fold; `tag-calldown` never folds after the flop).

## Evaluation

`poker/benchmark.py` reports, against each opponent type:
- **Chips per deal, duplicate**: each shuffled deck is replayed with the agent in every seat, starting from fixed stacks. This cancels most of the card luck. 0 is break-even. Decks depend only on `--seed` and the opponent, so two agents benchmarked with the same seed play the same cards.
- **Style statistics** from the same deals: VPIP, PFR, postflop aggression frequency (AFq), fold-to-bet (F2B), went-to-showdown (WTSD) and won-at-showdown (W$SD).
- **Bust-out win rate** (optional): the fraction of full games where the agent finishes with the most chips. 33% is break-even.
- The **M1 check** from `TICKET_learner_redesign.md`, and a **behaviour fingerprint** (`poker/diagnostics.py`): how preflop raising and folding correlate with hand equity (from the precomputed table in `poker/data/preflop_equity.json`), and the agent's choices in a few fixed flop spots.

Win rates against RandomAgent and SkillfulRandomAgent mostly reward aggression, because those bots fold often. The card-blind MaxRaise and CallStation bots are a better bar: neither can beat the other, so an agent that beats both is using its cards. TagAgent is a further reference point: a learned agent should beat it head to head.

## Observations

From the new learner (Oct 2026):
- Temporal-difference targets beat Monte Carlo everywhere. Monte-Carlo returns value a call by the agent's own later play, so a policy that starts folding later streets talks itself into folding everything.
- Against a pool of scripted bots, a curriculum (CallStation → MaxRaise → pool) did no better than training on the pool from the start.
- Learning plateaus at about 0.5–1M decisions with the current features: a 3M-decision run was no better.
- The scripted bots form a cycle (MaxRaise beats `tag`, `tag` beats `tag-calldown`, `tag-calldown` beats MaxRaise), so a strategy tuned against one bot can lose to another.

From the old learner, before the engine fixes (hypotheses rather than conclusions): pure self-play appeared prone to collapse, outcomes varied a lot across seeds, and training only against RandomAgent rewarded "bet until they fold".

## Next steps

Agreed direction, generalization first (details in `TICKET_learner_redesign.md`, Progress):
- **Held-out evaluation bots**, never trained against, to measure the generalization gap.
- **Parameterized bot families**, with a controlled random distribution over the mixture weights and each family's parameters: a richer opponent population that is harder to over-exploit.
- **Fix Q overestimation**, which shows up most on later streets against tight opponents: Double-DQN-style targets, a slower target network, n-step returns.
- **Self-play** (phase B): a league of frozen checkpoints plus the bot families, or NFSP.
- **Later, opponent statistics across deals** (a HUD-style VPIP, PFR and aggression per opponent) for MaxRaise-type opponents, checked against held-out bots for over-exploitation.
- **Learner**: a dueling head; postflop equity features (needs the fast evaluator).
- **Stronger reference opponents**: for example tabular CFR on an abstracted version of the game, which would also allow measuring exploitability.
- **Search**: decision-time search guided by the learned value function (as in Expert Iteration, arXiv:1705.08439), adapted for hidden information.
- **More players**: the features already pad to 6 players; training and evaluation would need to generalize.

## Repository layout

- `poker/`: game engine (`state.py`, `cards.py`, `hands.py`), agents, training (`play.py`), evaluation (`benchmark.py`, `diagnostics.py`, `preflop_equity.py`), run directories (`runs.py`)
- `poker/data/`: the precomputed preflop equity table
- `tests/`: unit tests, including rule tests and an end-to-end training smoke test
- `interactive_play.py`: play against the AI in the terminal (any model or scripted bot via `--model`)
- `run_training.py`, `grid_search.py`, `analyze_grid_search.py`: training and hyperparameter search scripts
- `models/`: saved weights (gitignored)
- `runs/`: run directories written by `poker.runs.RunLogger` (gitignored)
- `logs/`: training logs (gitignored)
- `archive/`: older notes, scripts, grid search results and training logs from before the Oct 2026 fixes
