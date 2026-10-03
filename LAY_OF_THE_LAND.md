# Lay of the Land (Oct 2026)

Status check after ~9 months away. Sections 1–5 are the original assessment (written before any changes). **Section 0 tracks what has been done since.**

## 0. Progress log

### Round 1: no-regret fixes (staged with `git add`, not committed)

| Area | Change | Verified by |
|---|---|---|
| Toolchain | Python 3.8 → **3.12** (pinned in `.python-version`), TF 2.8 → **2.21**, Keras 2 → **3**, relocked `uv.lock`. Everything lives in the repo-local `.venv`; uv downloaded the 3.12 interpreter into `~/.local/share/uv/python`. The old env was renamed to `.venv-py38` (gitignored) for rollback. | The old model gives **bit-identical** outputs (max diff 0.0) on 200 fixed inputs, and save/load round-trips |
| Keras 3 port | `.weights.h5` saving (legacy `.h5` still loads), optimizer LR setter, `train_on_batch` with consistent `(n, 1)` targets (mixing shapes crashed training under Keras 3), direct model calls instead of `predict()` (much faster per decision). Removed the transformer option: it was wired to the obsolete 19-feature state layout, so it read the wrong features. | New `tests/test_training_smoke.py` runs a tiny real-TF training job end to end |
| Hand evaluator (R1) | Full tiebreaks: kickers, lower pair, flush cards, full-house pair. Descriptions now name the kickers. | 13 new tests, including your hand and a 3,000-hand cross-check against an independent evaluator (they fail on the old code) |
| Rules (R2–R4) | `State.is_legal` / `State.legal_actions` are now the single source of truth; every agent and the UI use them. Fold is only legal when facing a bet. Raise cap `MAX_RAISES_PER_STAGE = 4` (forced blinds don't count). When nobody can bet, the board is dealt and the hand goes straight to showdown. Split pots are paid in whole chips, odd chip to the first winner left of the dealer. | New `tests/test_rules.py` (11 tests, including a 200-game randomized invariant sweep); 10 of 11 fail on the old engine |
| **New bug found and fixed** | When the *last* player to act folded (check, bet, call, **fold**), the street didn't close: the bettor was asked to act again and could even re-raise. `stage_is_complete` now compares all active players. | Regression test in `test_rules.py` |
| Training bugs (L1, L2, L5, L8) | The replay target is now **Expected SARSA over legal actions** under the softmax policy, with no bootstrapping from terminal states (it was a max over all actions, illegal ones included). Pre-training uses the game's real card encoding. The premium-hand diagnostic reads `hole_cards`. | Unit tests for the replay buffer, legal mask and pre-training encoding |
| UI | Correct labels ("Raise by $1 (put in $3)"), street announcements, net result per player, "stack behind / in pot" display, the final hand is now shown, folded opponents' cards stay hidden outside `--full-info`. | Scripted interactive run |
| Benchmark | `uv run python -m poker.benchmark`: chips per deal with **duplicate dealing** plus bust-out win rate, against Random, Skillful, Consistent, **MaxRaise** and **CallStation** (new `poker/baseline_agents.py`). | MaxRaise against itself scores exactly 0.00 ± 0.00, which is the sanity check for duplicate dealing |
| Hygiene | Old logs, notes, scripts, grid-search results and 150 checkpoints moved to `archive/` (`git mv` for tracked files, plain `mv` for untracked). `run_training.py` writes to a gitignored `logs/`. pytest only collects `tests/`. README gets a status section and corrected TODOs; CLAUDE.md updated. | — |

Tests: 85 → **117 passing** (2 skipped, same as before).

**Not done yet (design decisions, see §4B):** a Q head per action instead of a scalar action input, one deal per episode, input normalization, relative actions, pre-training target scale, more than one replay batch per episode.

### Benchmarks under the new rules (1000 duplicate decks × 3 seats at stacks of 20; 300 bust-out games)

Chips per deal for the row agent against two copies of the column agent; 0 is break-even.

| Agent \ vs 2× | Random | Skillful | Consistent | MaxRaise | CallStation |
|---|---|---|---|---|---|
| Old model (Jan 2026) | +7.09 ± 0.33 | +1.57 ± 0.35 | +0.56 ± 0.40 | **−0.43 ± 0.23** | −0.00 ± 0.09 |
| MaxRaise | +7.90 ± 0.31 | +1.52 ± 0.36 | +0.78 ± 0.42 | 0 | −0.00 ± 0.05 |
| CallStation | +1.33 ± 0.26 | +0.34 ± 0.26 | +0.41 ± 0.28 | −0.02 ± 0.03 | 0 |
| Skillful | +0.21 ± 0.19 | −0.02 ± 0.09 | −0.42 ± 0.18 | −0.56 ± 0.33 | −0.07 ± 0.22 |
| Random | +0.09 ± 0.23 | −0.09 ± 0.20 | −1.72 ± 0.23 | −3.94 ± 0.22 | −0.10 ± 0.18 |

**Retrained with the fixed code** (default `run_training.py` settings: 1500 episodes, about 10 minutes; models in `models/2026-10-02_rules_fixed/`):

| Checkpoint \ vs 2× | Random | Skillful | Consistent | MaxRaise | CallStation |
|---|---|---|---|---|---|
| ep 200 (end of scripted-opponent phase) | +3.28 ± 0.33 | +0.81 ± 0.28 | −0.83 ± 0.37 | −3.32 ± 0.43 | −0.01 ± 0.17 |
| ep 700 (end of mixed phase) | +1.31 ± 0.29 | +0.32 ± 0.25 | −1.51 ± 0.31 | −3.89 ± 0.35 | −0.18 ± 0.19 |
| ep 1500 (final) | +0.38 ± 0.25 | −0.13 ± 0.21 | −1.72 ± 0.27 | −3.91 ± 0.29 | −0.16 ± 0.19 |

The final model scores about the same as RandomAgent (compare the Random row above), and it gets worse during self-play. P(bet | AA) fell from 95% to 67% over the run. **My reading:** most of the old model's apparent skill came from bug L1. A max over all actions, illegal ones included, combined with Q being linear in the action pushed it toward aggression, which happens to work against these opponents. With that bias removed, the learner as designed learns little: one batch of 8 per episode, a scalar action input, multi-deal episodes. **The next gains are in §4B (learner design), not tuning.** The old Jan 2026 model is still the default for `interactive_play.py`.

The rule fixes alone helped the old model, because it can no longer fold when all in. Its bust-out win rate against MaxRaise went from 12% to 31%. It is still **no better than MaxRaise** on any column. **The bar for a "real" agent: clearly positive against both MaxRaise and CallStation.** Neither card-blind bot can manage that against the other, since when nobody folds every hand is decided by the cards.

### Round 2: measurement harness (step 1 of `TICKET_learner_redesign.md`, staged, not committed)

| Area | Change | Verified by |
|---|---|---|
| **Evaluator bug (new)** | `is_straight` stripped the ace and recursed, so it also stripped a second or third ace: **A A 2 3 4 and A A A 2 3 were scored as ace-high straights**. It dated from 2020 and changed the winners in **0.27% of 3-way showdowns**, inflating hands with aces and small cards (e.g. AA 3-way equity 0.743 instead of 0.735). Found because the preflop equity table disagreed with published values. | An exhaustive test over all **7,462** five-card hand classes against the independent reference evaluator, plus regression tests; 0 disagreements in 7,000 equity rollouts (35 + 24 before the fix) |
| `TagAgent` | A rule-based tight-aggressive bot that uses the board, with a call-down variant | Unit tests for hand classes and decisions; legality sweep |
| Preflop equity | `poker/data/preflop_equity.json`: 169 classes × 1 and 2 opponents, 20k rollouts each (SE ≈ 0.003) | Combo-weighted means are 0.4999 and 0.3338 (exactly 1/2 and 1/3 in theory); published heads-up values match |
| Benchmark | Paired decks (a deck RNG seeded per opponent), style stats (VPIP, PFR, AFq, F2B, WTSD, W$SD), model paths as opponents, JSON reports, the M1 check, and the global RNG restored afterwards | `tests/test_benchmark.py` |
| Fingerprint | `poker/diagnostics.py`: preflop rank correlation of P(raise) and P(fold) with equity, top-20% versus bottom-50% spreads, and 4 flop spots | `tests/test_diagnostics.py` |
| Runs | `poker/runs.py`: a `RunLogger` run directory plus `list` / `summarize` / `compare` (seed groups) / `plot` | `tests/test_runs.py` |

Tests: 117 → **184 passing** (2 skipped).

**Calibration against the new panel** (2000 paired decks; full table in the ticket, §7): the scripted bots form a cycle (MaxRaise > `tag` > `tag-calldown` > MaxRaise), and `tag-calldown` is the strongest, meeting 5 of 7 M1 criteria. The Jan 2026 model meets 4 of 7: it beats `tag` (+0.49) but loses to `tag-calldown` (−0.91) and MaxRaise (−0.49). Its preflop rank correlation looks healthy (+0.69), but the spread shows the effect is tiny: P(raise) is only 0.02 higher for the top 20% of hands than for the bottom 50% (`tag`: 0.32).

### Round 3: the new learner (M0 and first M1 results, staged, not committed)

- **PyTorch learner** (`poker/features.py`, `actions.py`, `env.py`, `dqn.py`, `train_dqn.py`):
  - one Q output per relative action head, with Q(fold) computed exactly;
  - single-deal episodes with random per-player stacks;
  - 256 tables batched together;
  - runs written by `RunLogger`;
  - `--seeds` launches one process per seed and prints `compare`.
- **Speed:** a direct 7-card evaluator is about 10× faster than taking the best of 21 combinations. Training runs at about 3,300 decisions/s, so 1M decisions takes about 5 minutes alone, or 13 with 6 runs in parallel.
- **Results:** temporal-difference targets (Expected SARSA plus a target network) beat Monte Carlo everywhere. Monte Carlo collapses into folding against MaxRaise. Trained against all seven scripted bots at random, the TD learner **beats 6 of 7 consistently across seeds**: Random +3.1, Skillful +1.5, Consistent +0.6, CallStation +1.3, `tag` +0.35, `tag-calldown` +0.24. MaxRaise is the exception at about −0.9, which looks like an opponent-identification problem.
- Full table and next options: `TICKET_learner_redesign.md` (Progress).

## TL;DR

- **The plumbing works.** 85 tests pass, training runs, interactive play works, and position invariance looks fixed.
- **The hand evaluator ignores kickers.** In your hand, **Player 2 should have won the whole pot.** The bug gives the wrong set of winners in about **16% of 3-way showdowns** and reports ties about 4× too often.
- **The trained model has learned roughly one thing: "raise a lot."** Its preflop policy is about the same for AA and 72o (raise about 75%, call about 25%). Its Q-values run from 70 to 160, but only 60 chips exist in the game.
- **A one-line "always raise the max" bot beats the trained model on every benchmark,** and a "always check/call" bot beats it head to head. The README's claim of a "generalizable poker strategy" doesn't hold up. Win rate against RandomAgent and SkillfulRandomAgent mostly measures how often those bots fold.
- **The training loop has bugs that would produce this result.** The replay target takes a max over *illegal* actions and has no terminal masking. The action goes into the network as a single number, so Q comes out almost linear in bet size. Pre-training uses the wrong card encoding.
- **The game rules push every hand toward all-in.** There is no cap on raises, and players who are all-in still get asked to act. The model folds about 24% of the time in those all-in spots.

Recommendation: fix correctness first (evaluator, rules, benchmark harness). After that, either rebuild the learner in a simpler, more standard way, or go straight to search (the Expert Iteration idea). Trying to tune the current SARSA setup won't help: its failures are structural, not about hyperparameters.

---

## 1. Your showdown question

Board: 7♠ J♥ T♠ J♠ 7♥. That's two pair (jacks and sevens) on the board for everyone, so the **fifth card (the kicker) decides the hand.**

| Player | Hole | Best five | Result |
|---|---|---|---|
| P2 | 6♦ A♣ | J J 7 7 **A** | **Wins outright** |
| P0 (you) | 3♥ K♦ | J J 7 7 K | 2nd |
| P1 | 2♦ 8♥ | J J 7 7 T (the board plays) | 3rd |

No straight is possible (P1's 8 would need a 9) and no flush (only three spades, and nobody holds one). Under real rules **P2 takes the whole $60 pot**, a $40 net gain. Instead the game split it three ways.

**Root cause:** `poker/hands.py:strength()` returns one number per category with almost no tiebreak:

- High card, flush: only the top card counts.
- Pair, trips, quads: only the rank of the group counts. No kickers.
- Two pair: `200 + higher_pair`. The **lower pair and the kicker are both ignored**, so JJ-33 ties JJ-77. The README marks "two-pair tie breaking" as ✅ DONE, but that's wrong. `test_two_pair_tie_breaking` only checks the higher pair.
- Full house: only the trips count. On a KKK board, 22 ties QQ.

**How often it matters** (Monte Carlo, 4,000 random 3-way showdowns, compared against a correct evaluator): the wrong set of winners in **15.7%** of cases, and ties in **18.9%** versus **4.5%** under real rules. This also hurts training: kickers are worth nothing to the agent, so ace-x hands can't be learned properly.

**Fix:** return a tuple `(category, tiebreak ranks...)`, e.g. `(2, (J, 7, A))`, and compare tuples. It's about 30 lines; my reference version is in the scratchpad. Add tests for kicker cases, including this exact hand.

---

## 2. How the current model actually plays

Everything below uses `models/player_0_latest.h5` with softmax at temperature 1, the same setup as interactive play. Self-play numbers are from 150 episodes. Win rates are from 300 episodes each, so roughly ±5.5 points at 95% confidence. Starting stacks were random, from $5 to $35.

### Self-play (all three seats use the model, stacks of 20)

- **1.2 deals per episode** (median 1). Someone usually busts on the first hand.
- Preflop actions: raise 64%, call 34%, fold 1.7%.
- **Average final pot is about 50 of the 60 chips in play.**
- **669 decisions came up where the player was already all-in** (the only legal actions were fold or check). The model **folded 23.5%** of the time in these spots, giving away its whole stack for nothing.
- Q-values seen ranged from **71 to 159**. With 60 chips total, that scale means nothing.

### The preflop policy barely depends on the hand

Player first to act, facing the big blind:

| Hand | Q(fold) | Q(call 2) | Q(raise 3) | P(raise) |
|---|---|---|---|---|
| AA | 115.8 | 119.6 | 120.8 | 77.5% |
| AKs | 112.6 | 116.3 | 117.6 | 77.9% |
| 72o | 88.8 | 91.8 | 92.8 | 72.7% |
| 32o | 80.8 | 83.6 | 84.6 | 71.7% |

The hand moves the *level* of Q a lot (35 points between AA and 32o, more than half the chips in the game). It barely moves the *gaps between actions*, and the softmax only looks at those gaps. If you feed in-between action values (-1, -0.5, …, 3), Q rises in near-equal steps of about 0.48. **Q is essentially linear in the action number, with the same slope for every hand.** So "bigger bet = better" always holds.

Q(fold) also climbs with pot size: 88 at pot 3, 106 at pot 60, for 72o. A fold's value shouldn't depend on the pot. The network has no working idea of what folding means.

### Benchmarks against simple bots

Win rate for the seat-0 bot against two copies of the opponent (fair share is 33%):

| Seat 0 \ Opponents | Random | Skillful | MaxRaise | CallStation | Trained |
|---|---|---|---|---|---|
| **Trained model** | 88% | 59% | **12%** | 31% | — |
| **MaxRaise** (always bet min(3, max legal)) | **96%** | **76%** | — | 36% | 59% |
| **CallStation** (always check/call) | 87% | 87% | 36% | — | 53% |

The README reports 90% against Random and 63% against Skillful as signs of skill. A bot with no inputs at all beats both numbers. RandomAgent folds often and SkillfulRandomAgent folds weak hands 70% of the time, so pure aggression wins. MaxRaise against CallStation comes out near 33%, because when nobody folds, every hand is decided at showdown (with the buggy evaluator).

---

## 3. Problems found, by area

### Game rules and engine (`poker/state.py`)
| # | Issue | Impact |
|---|---|---|
| R1 | Hand evaluator ignores kickers (§1) | Wrong winners in about 16% of 3-way showdowns |
| R2 | **No cap on raises per round**, with at most $3 added per action | Betting keeps escalating until stacks run out, so nearly every hand goes all-in |
| R3 | Players who are all-in still get asked to act, and **fold is legal for them** | The model folds about 24% of the time here, a huge leak |
| R4 | Fold is legal when checking is free | Never better than checking; the model does it about 2% of the time |
| R5 | Blinds are capped by the *smallest* stack at the table (`min(self.wealth)`) | Minor. One short stack shrinks the blinds for everyone. |
| R6 | Stale TODOs: the README's "big blind should get option" item is already handled (`state.py:182-197`), and so are ties (`state.py:350`; split pots work, the evaluator is what's broken) | Only the comments and README are stale |

### Action design (`poker/config.py`)
- An action means "chips put in on this action." So **`2` is a call when facing $2 and a raise when facing $1.** The network sees one number whose meaning changes with context. The interactive labels get this wrong too: "Raise to $3" really means "put in $3."
- Facing a $3 bet, raising is impossible, because $3 is the most you can add in one action.
- A relative action set would be cleaner: fold, check/call, raise by 1/2/3, or a few pot-fraction sizes. **Any change to the action set means retraining from scratch.**

### Learner (`poker/agent.py`, `poker/play.py`)
| # | Issue | Where |
|---|---|---|
| L1 | **The replay target is `max` over *all* actions, legal or not.** Q is about linear in the action, so the max is nearly always Q(bet 3). That's a self-reinforcing push toward aggression. It's labeled SARSA but is neither SARSA nor proper Q-learning. | `play.py` ~line 850 |
| L2 | **No terminal flag in replay.** Terminal transitions still bootstrap from the next state. | same |
| L3 | With `batch_size>1` the online update is turned off, leaving **one batch of 8 per episode**: about 12k sampled transitions across a 1,500-episode run. That's very little learning signal. | `play.py` ~line 390 |
| L4 | **The action goes in as a single scalar input** instead of one output head per action (standard DQN). This produces the near-linear Q in §2. | `agent.py:get_model_input` |
| L5 | **Pre-training uses the wrong card encoding:** ranks 2–14 and suits 1–4, but the game uses ranks 0–12 and suits 0–3. So "ace" in pre-training (14) never occurs in real play, and "pair of tens or better" is shifted by two ranks. | `agent.py:239-241, 374` |
| L6 | Pre-training targets are absolute wealth (×2 for betting a strong hand). RL targets are future chip changes, with γ=0.9999 and episodes that run until someone busts. The two scales conflict, and with γ≈1 the offset barely decays, which is likely why Q sits around 100. | `agent.py:285-319` |
| L7 | Inputs aren't normalized: ranks 0–12, wealth up to about 100, card slots set to -1 when not yet dealt. | `get_private_state` |
| L8 | The premium-hand fold diagnostic reads `state.private_cards`, which doesn't exist, so it **always reports 0**. | `play.py:326` |
| L9 | Episodes run until someone busts, scored by "most chips at the end." That's very noisy for both credit assignment and evaluation. | design |

### Evaluation
- The opponents used for evaluation fold a lot, so aggression scores well no matter how good the strategy is. MaxRaise and CallStation should be standard baselines.
- No variance reduction. **Duplicate dealing** (replay the same cards with seats rotated) plus chips-per-deal at fixed stacks would make comparisons much sharper than bust-out win rates.
- P(bet | AA) = 100% doesn't tell you much when P(bet | 72o) is around 99% too.

### Docs and hygiene
- The README is long and **contradicts both itself and the code.** It gives three different curriculum descriptions; the code (`RUN_DESCRIPTION` and the `run_sarsa` docstring) gives a fourth. Several "✅ DONE" items are wrong (two-pair tiebreak), and the "Bug fixes" TODOs are stale.
- At the repo root: about 20 `training_output_*.txt` files, 0-byte `*~` editor backups, 7 standalone `test_*.py` scripts outside `tests/`, 3 untracked `debug_*.py` scripts, and about 10 design-note `.md` files, some of which cite line numbers that no longer match.
- 150 checkpoint `.h5` files in `models/` (16 MB; gitignored, which is fine).
- **The toolchain is stale:** Python 3.8 (end of life since Oct 2024), TensorFlow 2.8.4 pinned, numpy below 1.24. It works today, but upgrading means either a TF bump or a switch to PyTorch or JAX.

---

## 4. Options for next steps

### A. Correctness and measurement (cheap, no retraining needed to verify)
1. **Fix the hand evaluator** with tuple rankings, plus kicker tests (including your hand).
2. **Rule fixes:** skip or auto-run-out players who are all-in; make fold illegal (or at least masked for agents) when checking is free; cap raises per round (e.g. 3–4 raises, like limit hold'em).
3. **Benchmark harness:** a script that plays any agent against Random, Skillful, Consistent, MaxRaise, and CallStation with duplicate dealing, reporting chips per deal and bust-out win rate. Record today's numbers (§2) as the baseline to beat.
4. **Interactive UI:** fix action labels, announce when the street changes, show stacks as "stack (committed)," and fix "YOU and Player 1 and Player 2 wins."
5. **README cleanup:** cut it down to what's currently true and move history to a `docs/history/` folder or a single changelog.

### B. Make the value-based learner sound (needs retraining)
Use a standard DQN-style setup:
- One output per action, with illegal actions masked in both the policy *and* the targets.
- Terminal flags, and a target network.
- Store the next action (true SARSA) or use Expected SARSA under the softmax policy.
- **One deal per episode**, with fixed or randomized stacks and reward = chip change for that deal.
- Normalized inputs and one-hot or embedded cards, plus a Monte-Carlo equity feature as a cheap, strong signal.
- Drop pre-training, or redo it using the correct encoding and the right target scale.
- Consider relative actions. If the action set changes anyway, this is the moment.

### C. Search: Expert Iteration (your note on PR #4)
ExIt (Anthony, Tian & Barber 2017, arXiv:1705.08439) alternates between two parts. The *apprentice* is a fast network policy and value function. The *expert* is a tree search guided by the apprentice. The search finds better moves, the apprentice learns to imitate them, and a better apprentice makes the next search stronger. The paper's game was Hex, which has perfect information. **The catch with poker is hidden information.** Some ways to adapt it, roughly from simplest to most principled:

1. **Determinized search (perfect-information Monte Carlo / ISMCTS):** at each decision, sample opponent hole cards (ideally weighted by what their actions suggest), run MCTS or rollouts guided by the value net, and average the results. Easy to build and a big step up from the current agent. Known flaws: "strategy fusion" (it assumes it will learn the hidden cards later) and no reason to bluff or to balance its play.
2. **MCCFR / Deep CFR:** the standard for imperfect-information games. With a raise cap and hand-strength buckets, this game may be small enough for **tabular** MCCFR, which would give a strong reference opponent and a ground truth for measuring exploitability. Pluribus used MCCFR for 6-player play, so the 3-player setting isn't a blocker in practice, even though the theory doesn't guarantee convergence there.
3. **ReBeL / Student of Games:** the true imperfect-information version of ExIt, with search over public belief states plus a value net. It's heavy, and you'd want items 1 and 2 working first.

A sensible path is A → (B or C1) → C2 as the reference opponent. Moving to **2 players (heads-up) first** would make everything easier: Nash equilibrium is well defined and exploitability can actually be computed.

### Possible separate tickets
- `TICKET_hand_evaluator.md`: spec plus a test list. Small enough that it could just be done.
- `TICKET_benchmark_harness.md`: metrics, duplicate dealing, baselines.
- `TICKET_expert_iteration.md`: design for C1 and C2, including how belief sampling would work.

---

## 5. Open questions for you

1. **Goal:** is this mainly a learning project (try RL ideas such as ExIt), or are you aiming for an agent that's actually strong? That decides between B and going straight to C.
2. **Rules:** are you OK changing the game (raise cap, no folding when checking is free, auto-run-out for all-in players, maybe relative actions)? Every option invalidates the existing models.
3. **Player count:** keep 3 players, or simplify to heads-up for a while?
4. **Toolchain:** keep TF 2.8 / Python 3.8, or upgrade now? Switching to PyTorch would be natural if the learner gets rewritten anyway.
5. **Cleanup:** may I archive the old `training_output_*.txt` files, `*~` backups, and stale notes into a subfolder? Nothing would be deleted without your OK.
