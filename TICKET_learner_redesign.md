# Ticket: Redesign the learner

**Status:** step 1 (measurement harness) and M0 (plumbing) done; M1: 6 of 7 criteria met consistently, MaxRaise remaining · **Context:** `LAY_OF_THE_LAND.md` §0, §3 (Learner) and §4B

**Progress.**
- **Step 1, the measuring stick:**
  - `TagAgent` (both variants) in `poker/baseline_agents.py`.
  - The preflop equity table (§3, option a) in `poker/data/preflop_equity.json`, via `poker/preflop_equity.py`.
  - Benchmark upgrades in `poker/benchmark.py`: paired decks, style statistics, model opponents (including `.pt`), JSON reports, the M1 check.
  - The behaviour fingerprint in `poker/diagnostics.py`.
  - Run directories plus `summarize` / `compare` / `list` / `plot` in `poker/runs.py`.
  - Building it exposed an evaluator bug: A A 2 3 4 was scored as a straight (`LAY_OF_THE_LAND.md`, round 2).
- **M0, plumbing (PyTorch):**
  - `poker/actions.py`, `poker/features.py` (172 features, `FEATURE_VERSION = 1`), `poker/env.py`, `poker/dqn.py` and `poker/train_dqn.py`.
  - `State` accepts per-player stacks and counts raises per player.
  - A direct 7-card evaluator in `poker/hands.py`, about 10× faster.
  - Training runs at about 3,300 decisions/s, including gradient steps.
  - Tests cover every §8 item. The learning sanity check is a 30k-decision run that takes about 8 s, and it checks that the agent folds weak hands more than strong ones and that Q(call | AA) > Q(fold | AA). The ticket's original criterion, P(raise | AA) > P(raise | 72o), isn't reliable yet.
- **M1 experiments** (Oct 3; chips per deal against 2× each bot; mean ± sd over 3 seeds; final evals at 1000 paired decks):

  | Training | random | skillful | consistent | maxraise | callstation | tag | tag-calldown | panel | M1 |
  |---|---|---|---|---|---|---|---|---|---|
  | MC, CallStation only, 400k (1 seed) | | | | −2.07 | +0.75 | −0.16 | −0.09 | | |
  | MC, MaxRaise only, 400k | −0.81 | −0.47 | −0.74 | −0.87 ± 0.10 | −0.10 | −0.26 | −0.27 | −0.51 ± 0.05 | 0/7 |
  | **TD**, MaxRaise only, 400k | +1.20 | +0.36 | +0.27 | **+0.25 ± 0.25** | +0.68 | +0.05 | −0.01 | +0.40 ± 0.10 | 4–5/7 |
  | MC, pool, 1M | −0.28 | −0.05 | −0.35 | −0.83 | +0.37 | +0.06 | +0.11 | −0.14 ± 0.04 | 0–2/7 |
  | MC, curriculum, 1M | +0.01 | +0.12 | −0.15 | −0.84 | +0.54 | +0.08 | +0.14 | −0.02 ± 0.15 | 0, 2, 6/7 |
  | **TD, pool, 1M** | +3.11 | +1.47 | +0.64 | −0.91 ± 0.38 | +1.29 ± 0.03 | **+0.35 ± 0.03** | **+0.24 ± 0.07** | **+0.88 ± 0.03** | 5–6/7 |
  | TD, curriculum, 1M | +3.05 | +1.36 | +0.70 | −1.03 ± 0.23 | +1.21 | +0.42 | +0.14 | +0.84 ± 0.13 | 5–6/7 |

  "Pool" means all seven scripted bots, drawn per seat per deal. The curriculum is CallStation 200k → MaxRaise 200k → pool 600k.

  - **Monte Carlo collapses against MaxRaise.** It ends up folding 99% of hands when first in. Its targets value a call by the agent's *own* later play, so once it starts folding later streets, calling now looks worse than folding now, and the effect feeds on itself.
  - **TD (step 2) fixes the collapse and wins every comparison**, so it is now the default (`--targets td`; Monte Carlo stays available).
  - **In the pool, TD reliably beats 6 of the 7 bots**, including both TAG variants. Its style is loose-aggressive: VPIP about 70%, PFR about 40%. It value-bets top pair and folds air facing a bet.
  - **The curriculum doesn't help over the pool with TD**, though it helped Monte Carlo a little.
  - **MaxRaise is the holdout in the pool (≈ −0.9)**, even though TD trained against MaxRaise alone beats it. This is the opponent-identification problem: opponents are re-drawn every deal, so a raise usually signals a strong `tag` or Skillful hand, and the agent pays off a maniac whose raises mean nothing. Options:
    - (a) Weight MaxRaise more heavily in the pool.
    - (b) Keep the same opponents at a table for several deals and add per-opponent statistics from previous deals as features (VPIP, PFR, aggression, like a poker HUD). That needs a `FEATURE_VERSION` bump.
    - (c) Train longer.

- **Next, as agreed on Oct 3: measure and improve generalization before adding opponent-specific features.**
  - **The concern:** M1 numbers are training-set numbers, since we train and evaluate on the same seven bots, and the agent is a best response to that mix. Opponent statistics (HUD features) would help against MaxRaise, but could teach bot-specific over-exploitation that doesn't generalize.
  1. **Held-out evaluation bots**, never trained against: new families, or parameter regions excluded from training. Report them next to the M1 panel as a generalization gap.
  2. **Parameterized bot families** with a random distribution that we control over (a) the mixture weights across families and (b) each family's parameters. For example:
     - `tag(strong/medium thresholds, bluff frequency, fold-postflop probability)`;
     - `aggro(p)`: bets or raises the maximum with probability p, else calls;
     - `station(fold probability)`;
     - `random(fold / call / raise weights)`.
     Sample per seat per deal, or per table for several deals. This gives a continuous, harder-to-exploit population, and the held-out regions come for free.
  3. **Fix Q overestimation** (found Oct 3 by playing the pool agent greedily and comparing the predicted Q of each chosen action with the realized deal reward). The bias is +0.38 BB at the first decision against the pool, and +1.3 to +2.2 BB on the flop, turn and river against 2× `tag`.
     - **Two separate effects:** a general bias from the near-greedy max in the Expected SARSA target, and distribution shift, since 2× `tag` tables are rare in training (1/49 with per-seat sampling).
     - **Fixes to try:** Double-DQN-style targets (choose the next action with the online network, evaluate it with the target network), a slower target network, n-step returns.
     - **Diagnostic:** make this calibration check a standard benchmark report (bias by street, per opponent).
  4. **Self-play (phase B):** a league of frozen checkpoints plus the bot families, then possibly NFSP.
  5. **Only then, opponent statistics across deals (HUD)**, evaluated on held-out bots to catch over-exploitation.

The trainer writes its output through `poker.runs.RunLogger`, and its periodic evaluations through `poker.benchmark.run_benchmark` with a fixed seed, so they are paired across checkpoints and runs.

## Why

With the training bugs fixed, the current learner learns little. Retrained from scratch, it scores about the same as RandomAgent: −3.9 chips/deal against MaxRaise, and it gets *worse* during self-play. The old model looked better only because a bug (a max over illegal actions combined with Q being linear in the action) pushed it toward aggression. The problems are structural:

| Problem | Effect |
|---|---|
| The action is a single scalar network input | Q is close to linear in bet size, and the same slope applies to every hand |
| An action number means different things in different spots (`2` is a call when facing $2 and a raise when facing $1) | The network has to untangle what the action means from context |
| The state has no amount to call, no dealer position and no betting history (`Agent.get_private_state`) | The network often *can't* tell whether "put in $2" is a call or a raise, or where it sits relative to the button |
| Episodes run until someone busts, with γ=0.9999 and chip-change rewards | Long, noisy credit assignment; the value scale drifts with stack size |
| One replay batch of 8 per episode | Roughly 12k sampled transitions in a whole run, so very little learning signal |
| Raw, unnormalized inputs (ranks 0–12, stacks up to about 100, -1 for missing cards) | Harder optimization; hand strength has to be learned from scratch |
| Heuristic pre-training toward targets on a different scale | Starting values fight the real targets |
| Training against self-play while the opponents copy weights each episode | Moving target; no mechanism that converges to sound mixed strategies |

## Goal

An agent that **clearly beats both card-blind baselines** (MaxRaise and CallStation). Neither of those can beat the other, so beating both means the agent is using its cards. Then make it robust in self-play, and make it fun to play against.

## Proposed design

### 1. One deal per episode, reward = chip change
- A training episode is a single deal. The reward is the agent's chip change at the end of the deal, measured in big blinds. There is no discounting (a deal lasts at most 4 streets).
- Stacks are randomized **per deal and per player** (e.g. 5–35 chips, so 2.5–17.5 BB) so the agent learns to play at different stack depths. This needs a small `State` change to accept a list of starting stacks.
- The objective becomes chip EV per hand, the standard target in poker AI. Bust-out ("tournament") win rate stays as a secondary benchmark metric. Its effects (e.g. survival mattering more than chips) are ignored for now.

### 2. One Q output per action, with illegal actions masked
- The network maps the state to a vector of Q-values, one per action, as in standard DQN. Illegal actions are masked to −∞ in both the policy and the targets.
- **Relative action heads, as a representation-only change:** `FOLD`, `CHECK_CALL`, `RAISE_1`, `RAISE_2`, `RAISE_3`, where raise k means "put in the amount owed plus k". These map onto the existing game actions (a chip amount from 0 to 3), so **the game rules don't change**. For example, facing $1, `RAISE_2` puts in $3, and `RAISE_3` (putting in $4) is illegal. Each head means the same thing in every spot.
- **Q(fold) is not learned.** With reward equal to the deal's chip change, folding is worth exactly −(chips already committed this deal), in BB. Compute that directly instead of learning a fold head. This removes noise from the targets, and it rules out the old model's pathology where Q(fold) rose with the pot. The network learns only the check/call and raise heads.
- Possible later addition: a dueling head (separate state value and per-action advantage).

### 3. Features: normalized, with an explicit hand-strength signal
| Group | Encoding |
|---|---|
| Stage | one-hot (4) |
| Hole cards | 52-dim binary, so order doesn't matter |
| Board | 52-dim binary |
| Hand strength | **equity estimate** (see below), plus a one-hot of the current made-hand category |
| Money | pot, amount to call, own stack behind, each opponent's stack behind: all in BB, scaled by a constant |
| Opponents | per-opponent block in relative seat order: present, still in the hand, stack, chips committed this deal and this street, raises this street |
| Position | one-hot of seat offset from the dealer; number of active players |
| Street context | raises so far this street versus the cap |

**N-player readiness:** per-opponent blocks are padded to `MAX_PLAYERS` (say 6) with a "seat present" flag, so supporting 2–6 players later only changes the data, not the architecture. A DeepSets-style shared opponent encoder is a later option.

**Equity feature:** the current evaluator does about 7.5k seven-card evaluations per second (measured Oct 2), which is too slow for Monte-Carlo equity at every decision. Even the made-hand category costs about 130 µs per decision, so cache it per player per street. Options:
- (a) **Preflop:** precompute a lookup table once, covering 169 starting-hand classes × 1–2 opponents, and save it in the repo. This is cheap (minutes) and is a clear win.
- (b) **Postflop:** use a fast evaluator. Either add a dependency such as `phevaluator` (a C extension), or write a lookup-table evaluator in numpy. Then estimate equity with about 100–200 random rollouts.
- (c) Skip postflop equity at first and rely on the made-hand category plus board cards.

Recommendation: do (a) now, start with (c), and move to (b) if postflop play is the bottleneck.

### 4. Learning algorithm, in steps
**Step 1: Monte-Carlo targets** (simplest to get right). Store each (state, action) the learner takes during a deal. When the deal ends, regress Q(s, a) toward the realized deal reward with an **MSE loss**. Deals are short, so the variance is manageable. Monte-Carlo targets need no bootstrapping, no target network and no terminal-flag handling, so there are fewer ways for training to be silently wrong.
- **Why MSE and not Huber:** the Q we want is the *mean* return. Huber with a small δ regresses toward something between the mean and the median. Deal returns are skewed (many small losses, a few big wins), so Huber would bias Q(call/raise) downward against the exact Q(fold) and push the policy toward folding. Rewards are bounded (at most about 17.5 BB at the deepest stacks), so MSE is safe.
- Monte-Carlo returns reflect the policy that generated them, including the learner's own later actions in the deal. Keep the replay buffer small enough that most of it comes from the recent policy. Temporal-difference targets (step 2) don't have this problem.

**Step 2: temporal-difference targets as an improvement** (optional, comparing against step 1): Expected SARSA or n-step returns, with a target network (Polyak averaging). Only worth keeping if it beats Monte Carlo on the benchmark.

**Training regime:**
- **Phase A, fixed opponents:** play only against scripted opponents. This is a well-posed single-agent RL problem, so if the learner can't beat fixed bots, something in the plumbing is wrong. Exploration is ε-greedy decaying to about 0.05, or softmax over Q in BB units. Work up a ladder, moving on only once each rung passes:
  1. **CallStation only.** This is the easiest test that the agent uses its cards: value-bet strong hands, check or fold weak ones. Both rule bots (§7) get +1.43 here.
  2. **MaxRaise only.** This tests calling down against a maniac and folding trash preflop. `tag-calldown` gets +0.75 here, `tag` −0.26.
  3. **The pool:** Random, Skillful, Consistent, MaxRaise, CallStation and the rule bot, sampled per seat per deal.
- **How this differs from the old curriculum.** The old code also started against scripted bots (Random, Consistent and Skillful for 200 episodes) before moving to 60–70% self-play. But that phase got only about 1,600 sampled transitions (200 episodes × one batch of 8), and the old learner never beat the fixed bots: the ep-200 checkpoint was −3.3 chips/deal against MaxRaise. All three of those bots are mostly random. They look only at their hole cards (any ace or TT+), never at the board, and they fold a lot, so aggression was rewarded. Phase A here is a *gate* with a much larger budget. Its pool adds MaxRaise and CallStation, which punish folding and bluffing too much, plus a rule bot that plays the board.
- **Opponent identification.** Against a pool sampled per deal, the learner can only play a best response to the *mixture* unless it can tell from the current deal who it is facing. The best counter-strategies differ: steal relentlessly against Random, call down against MaxRaise, never bluff against CallStation. The per-opponent features (chips committed and raises this street) are what let it work this out. This is also why the M1 thresholds below don't demand maximal exploitation of every bot at once.
- **Phase B, self-play:** **NFSP** (Neural Fictitious Self-Play, Heinrich & Silver 2016). It was designed for exactly this setting, Q-learning in imperfect-information self-play. A best-response Q-network plays against an *average policy* network, which is trained by supervised learning on the best-response network's own past actions. The average policy is the one that converges toward an equilibrium in 2-player games; with 3 players there's no guarantee, but it works well empirically. A cheaper alternative is a league of frozen past checkpoints plus the scripted pool.
- **Phase C (later, separate tickets):** strong reference opponents (tabular CFR on an abstracted game), exploitability estimates, and decision-time search.

### 5. Throughput
Measured Oct 2 on this machine (CPU only):

| | |
|---|---|
| Game engine, scripted players | about 2,800 deals/s, 21k decisions/s (7.5 decisions per deal) |
| Q-network forward pass | 2.0 ms for one state, 9.5 ms for a batch of 1,280: about 270× cheaper per state when batched |
| Hand evaluator | 7.5k seven-card evaluations/s |

- Play **many tables in parallel** (e.g. 256) and batch the Q-network calls across tables. The engine itself is fast enough for 1–5M decisions without changes.
- Train on minibatches of 256 from a replay buffer (see the note on buffer size under step 1), with about 1 gradient step per 1–4 new decisions. **Gradient steps will probably dominate wall time** (e.g. 250k steps for 1M decisions), so the training step should be compiled (`tf.function` / `torch.compile`, or simply a small hand-written loop).
- Target scale for a first run: 1–5M decisions, which should take minutes to tens of minutes.

### 6. Code layout
New modules alongside the old ones. The old `Agent` / `play.py` keep working for comparison until the new learner wins on the benchmark, then they move to `archive/`.
- `poker/features.py`: state → feature vector; versioned and unit-tested.
- `poker/actions.py`: relative action heads ↔ game actions, plus legality masks.
- `poker/env.py`: a vectorized single-deal environment (reset with random stacks and dealer; step until the learner's next decision or the end of the deal).
- `poker/dqn.py`: network, replay buffer, Monte-Carlo and temporal-difference targets, and the agent's `get_action`.
- `poker/train_dqn.py`: training loop, periodic benchmarks, checkpoints.
- Models are saved as `.weights.h5` **plus a JSON config** (feature version, action set, layer sizes), so an incompatible model fails loudly instead of loading into the wrong architecture.

### 7. Evaluation
- **Add a rule-based reference bot** (`TagAgent` in `poker/baseline_agents.py`), both as a benchmark column and as a Phase A opponent. It is the first scripted opponent that reacts to the board. Its rules:
  - Preflop classes: *strong* is 77+, AQ+, AJs, KQs; *medium* is any pair, any ace, two cards T or higher, or suited connectors 56s and up; everything else is *weak*.
  - After the flop: *strong* is two pair or better, or top pair or an overpair; *medium* is any other improvement over the board; *weak* means the board plays.
  - Strong hands put in the maximum, medium hands check or call, weak hands check or fold. A `fold_postflop` flag selects between the two variants below.
- **Calibration** (Oct 2, chips per deal, 2000 paired duplicate decks at stacks of 20, margins ±0.03 to ±0.30, fixed evaluator). These numbers set the M1 bar:

  | Agent \ vs 2× | Random | Skillful | Consistent | MaxRaise | CallStation | tag | tag-calldown | M1 |
  |---|---|---|---|---|---|---|---|---|
  | `tag-calldown` | +0.75 | +0.34 | +0.32 | +0.75 | +1.43 | −0.36 | 0 | 5/7 |
  | `tag` | +0.17 | −0.06 | +0.00 | −0.26 | +1.43 | 0 | +0.36 | 4/7 |
  | MaxRaise | +8.15 | +1.66 | +0.60 | 0 | +0.01 | +0.87 | −0.79 | 4/7 |
  | CallStation | +1.49 | +0.12 | +0.28 | +0.01 | 0 | −1.49 | −1.60 | 3/7 |
  | Old model (Jan 2026) | +7.25 | +1.67 | +0.57 | −0.49 | +0.06 | +0.49 | −0.91 | 4/7 |
  | Retrained (Oct 2) | +0.78 | +0.03 | −1.69 | −3.80 | −0.12 | +0.04 | −0.70 | 2/7 |
  | Random | −0.04 | +0.06 | −2.18 | −3.99 | −0.06 | +0.10 | −0.77 | 1/7 |

  - **No single bot dominates.** MaxRaise beats `tag` (+0.87), `tag` beats `tag-calldown` (+0.36), and `tag-calldown` beats MaxRaise (+0.75). A strategy tuned against one of them can lose to another, so M1 asks for all of them at once.
  - **The strongest scripted bot is `tag-calldown`.** It meets every M1 criterion except the two `tag` columns. (The diagonal is 0 by construction, so no agent can pass against its own copies.)
  - **Exploiting Random is a different skill.** Only near-total aggression gets close to MaxRaise's +8.15 against Random. Any hand selection costs a lot there: a scratch variant that plays like MaxRaise but folds weak hands preflop got only about +2.35. But hand selection is exactly what beats MaxRaise.
- **Benchmark size:** at 1000 decks the margin against MaxRaise is about ±0.36, too wide to settle a +0.5 threshold. Use about 5000 decks (±0.16) for acceptance runs, and fewer for the periodic checks during training.
- Run `poker.benchmark` every N steps and log the results. Opponents can be model paths (`--opponents maxraise models/old.weights.h5`) for head-to-head games (new versus old model, new versus earlier checkpoints).
- Diagnostics: P(raise) by preflop hand class, Q-values by action for fixed reference spots, and action frequencies by street.
- Later: a **best-response probe**, i.e. train a fresh DQN against the frozen agent. How much it wins estimates how exploitable the agent is.

### 8. Tests
- Features: shapes, value ranges, invariance to hole-card order, correct relative seat order, padding for absent seats.
- Actions: mapping between relative heads and game actions, and that masks agree with `State.legal_actions` across randomly generated states.
- Targets: Monte-Carlo returns assigned to every decision in a deal; Q(fold) equals −(chips committed) in BB; masking in temporal-difference targets.
- `TagAgent`: hand classes for a few fixed hands and boards; it never takes an illegal action.
- Environment: chips are conserved, episodes end at the end of the deal, and the per-player starting stacks are honored.
- **Learning sanity check** (fast, a few seconds): after a short run against CallStation, P(raise | AA) > P(raise | 72o) and Q(fold) < Q(call) for AA.
- Keep the end-to-end smoke test, pointed at the new trainer.

## Milestones and acceptance criteria

Chips per deal at stacks of 20, duplicate benchmark (5000 decks for acceptance):

| Milestone | Acceptance |
|---|---|
| **M0: plumbing** | features, actions and environment modules plus tests; vectorized environment; `TagAgent` in the benchmark; Monte-Carlo DQN trains end to end |
| **M1: beats fixed bots** | ≥ +0.5 vs MaxRaise and vs CallStation (the old model gets −0.49 and +0.06); > 0 against 2× `TagAgent` (both variants), with the confidence interval excluding 0; > 0 vs Random, Skillful and Consistent as a sanity check. There is no higher bar against those three: matching MaxRaise's +8.15 against Random conflicts with beating MaxRaise (see §7). |
| **M2: robust in self-play** | M1 thresholds still hold; beats the M1 agent and the Jan 2026 model head to head; no collapse over a long self-play run |
| **M3: fun to play** | the new model is the default for `interactive_play.py`; difficulty/style controls (see open questions) |

## Out of scope for this ticket
Changing the game itself (bet sizes, stack depths, no-limit), CFR or search, and more than 3 players. The design keeps all of these possible later.

## Open questions
1. **Objective:** chip EV per deal (recommended) or bust-out win rate?
2. **Training stack depths:** 5–35 chips (2.5–17.5 BB) is quite shallow. Should training and interactive play use deeper stacks?
3. **Dependencies:** OK to add a fast hand-evaluator package (e.g. `phevaluator`), or keep everything in pure Python and numpy?
4. **Framework:** stay on Keras 3 (works today) or switch to PyTorch for the new learner? Both are fine; PyTorch is more common in RL code you'll read.
5. **What makes it fun?** Strength, readable human-like play, adjustable difficulty (e.g. temperature, or mixing in mistakes), visible reasoning (the `--full-info` probabilities), or something else? This decides what M3 looks like and whether we keep stochastic play.
6. **NFSP vs a checkpoint league** for phase B? NFSP is more principled; a league is simpler. We could start with a league and add NFSP if self-play stalls.
