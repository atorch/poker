# poker

## Quick Start

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable dependency management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --all-extras

# Run the poker agent (uses best config: lr=0.0003, (128,128) network, batch_size=8)
uv run python poker/play.py

# Run tests
uv run pytest tests/

# Run specific test with verbose output
uv run pytest tests/test_simple_stage_transition.py -v -s

# Run grid search for hyperparameter tuning
uv run python grid_search.py --mode medium
```

## Results Summary (Jan 2026)

**✅ SUCCESS: Full Curriculum with Self-Play - Agent Learns Generalizable Poker Strategy**

**Final Performance (1500 episodes, lr=0.0003, (128,128) network, batch_size=8):**
- **vs RandomAgent: 97.6% ± 0.9%** ← Matches best grid search result!
- **vs SkillfulRandomAgent: 85.0% ± 2.2%** ← Strong generalization!
- **Self-play equilibrium: 35.6%** ← Close to Nash (33.3%), no positional bias
- **Performance drop (Random→Skillful): 12.6 pp** ← EXCELLENT (< 20pp threshold)

**Key Achievement: Proved agent learned generalizable poker fundamentals, not just random exploitation**
- SkillfulRandomAgent protects premium hands (only folds AA/AK/pairs 5% of time)
- 12.6pp drop indicates strategic flexibility and proper adaptation
- Agent survived 700 episodes of mixed/self-play without catastrophic forgetting

**Curriculum Progression:**
1. **Random phase (0-800)**: 55% → 87% win rate vs RandomAgent
   - Learned basic exploitation patterns
   - P(bet|AA) reached 100% by episode 100
2. **Mixed phase (800-1200)**: 50% self-play, 25% RandomAgent, 25% SkillfulRandomAgent
   - Maintained 77-87% win rate during training
   - Developed aggressive fast-game strategy (games dropped from 5-10 deals to 1-2 deals)
3. **Self-play phase (1200-1500)**: 70% self-play, 15% RandomAgent, 15% SkillfulRandomAgent
   - Stable 72-76% win rate during training
   - No catastrophic collapse (previous issue with 100% self-play)

**All sanity checks PASS:**
- Premium hands (AA, KK, AK) prefer betting ✓
- Q(check) > Q(fold) when checking is free ✓
- Q-values properly ordered by hand strength ✓
- P(bet|AA) = 100% throughout all 1500 episodes ✓

**🎓 Key Learnings from Training:**
- **Softmax policy essential for poker:** Deterministic argmax led to pathological folding (see previous runs). Softmax enables mixed strategies needed for Nash equilibrium.
- **State space matters:** Added pot size + opponent fold status → agent learned to adjust strategy based on pot odds and number of active opponents.
- **Never go 100% self-play!** Pure self-play causes catastrophic forgetting
  - Agent learns well vs RandomAgent (37.7% win rate in first 100 episodes)
  - Switching to pure self-play → forgets how to exploit weak opponents → collapses into "fold everything" equilibrium
  - **Solution**: Always maintain 10-20% probability of facing RandomAgent/SkillfulRandomAgent as anchor
  - This acts as regularization preventing bad equilibria
- **Curriculum must be gradual:** Discrete jumps (100% random → 50% → 0%) cause instability
  - Better: 100% → 80% → 90% self-play (never reaching 100%)
  - Smooth transitions preserve learned strategies while enabling self-play improvement

**⚠️ Previous Failed Experiment (Argmax Policy):**
- Agent collapsed into pathological behavior: folded pocket aces!
- Self-play episodes became extremely long (300+ deals)
- Fold rate increased to 78% by episode 290
- **Root cause:** Argmax deterministic policy + self-play → everyone learns "folding is safe" → bad equilibrium

## 🎲 CRITICAL: High Training Variance Discovered (Dec 25-30, 2025)

**Problem**: Grid search revealed EXTREME variance in training outcomes with identical hyperparameters.

### Variance Study Results (10 seeds per config)

**WITHOUT Pre-training:**
```
Config: batch_size=8, lr=0.0003, network=(64,64), n_episodes=500
Mean: 47.9% ± 24.5% (std)
Range: 12.3% - 97.3%
Distribution: 7 mediocre seeds (27-60%), 3 good seeds (60-97%)
```

**WITH Pre-training (Q ≈ wealth heuristic):**
```
Config: Same as above + pretrain_wealth_heuristic=True
Mean: 84.3% ± 26.3% (std)
Range: 14.7% - 99.3%
Distribution: 9 excellent seeds (87-99%), 1 catastrophic failure (14.7%)
```

### Key Findings

**✅ Pre-training dramatically improves mean performance:**
- Mean win rate: 47.9% → 84.3% (+36.4 percentage points!)
- P(bet|AA): 72.4% → 93.1% (+20.7 pp)
- 90% of seeds now achieve strong play (87-99% win rate)

**❌ Pre-training does NOT reduce variance:**
- Std dev: 24.5% → 26.3% (slightly worse!)
- Still have catastrophic outliers (1 seed collapsed to 14.7%)
- Variance problem runs deeper than initialization

**💡 Interpretation:**
- Pre-training shifts the distribution upward but doesn't prevent collapse
- Likely training instability during SARSA (not just bad initialization):
  - Catastrophic forgetting during self-play?
  - Learning rate too high for some trajectories?
  - Experience replay buffer poisoning?
  - Curriculum timing issues?
- **Recommendation**: Enable pre-training by default (huge mean improvement), but variance still needs investigation

**Sources of variance (identified):**
1. ~~Network weight initialization (random)~~ ← Partially addressed by pre-training
2. Training dynamics instability (learning rate, curriculum, self-play)
3. Deck shuffling (every deal)
4. Initial dealer/wealth randomization
5. Epsilon-greedy exploration (stochastic)
6. Softmax action sampling
7. Opponent selection during mixed training

**Next steps:**
1. ✅ **DONE**: Implement simple wealth heuristic pre-training
2. ✅ **DONE**: Variance study (10 seeds) with/without pre-training
3. ✅ **DONE**: Enable pre-training by default
4. ✅ **DONE**: Fix turn/river bug (state expanded from 18 to 22 features)
5. ✅ **DONE**: Medium grid search with larger networks for expanded state space
6. ✅ **DONE**: Full curriculum training (random → mixed → self-play) with best config
7. ✅ **DONE**: Add SkillfulRandomAgent evaluation to standard training validation
8. 🔴 **TODO**: Investigate why some seeds still collapse despite pre-training
9. 🔴 **TODO**: Test more sophisticated pre-training (hand strength + pot odds)

## 🎯 Medium Grid Search Results (Jan 2026)

**Context**: After fixing turn/river bug (state 18→22 features), ran medium grid to test if larger networks help.

**Configuration tested:** 24 configs
- Learning rates: [0.0001, 0.0003, 0.001]
- Networks: [(64,64), (128,128), (128,128,128), (256,256)]
- Episodes: [500, 1000]
- Pre-training: ENABLED
- Training: Only vs RandomAgent (no self-play, no SkillfulRandomAgent)
- Evaluation: vs both RandomAgent AND SkillfulRandomAgent (frozen policy)

### Key Findings

**✅ Best configuration found:**
```
lr=0.0003, network=(128,128), batch_size=8, episodes=1000
- Win rate vs RandomAgent:      97.7%
- Win rate vs SkillfulRandomAgent: 88.7%  ← Best generalization!
- P(bet|AA): 100%
```

**🧠 Critical insight: Exploitation vs Generalization**
- **Training only vs RandomAgent** (which folds randomly) can teach two different things:
  1. **Exploitation**: "Wait for random folding, bet when you get lucky"
     - Works great vs RandomAgent (99% win rate possible)
     - FAILS vs smarter opponents that protect premium hands
  2. **Generalization**: "Bet premium hands, fold trash, value bet strategically"
     - Works well vs RandomAgent (88-98% win rate)
     - ALSO works vs SkillfulRandomAgent (80-89% win rate)
- **SkillfulRandomAgent** protects strong hands (only folds AA/AK/pairs 5% of time)
  - Tests if agent learned exploitable patterns or generalizable poker strategy
  - Performance drop of ~14-16pp from RandomAgent → SkillfulRandomAgent is healthy
  - Configs with >30pp drop likely over-fit to random folding

**📊 Network size matters (but smaller is better!):**
- **(64,64)**: 100% success, 93.8% vs Random, 77.4% vs Skillful
- **(128,128)**: 100% success, 91.1% vs Random, 77.5% vs Skillful ← Best balance
- **(128,128,128)**: 83% success, higher variance
- **(256,256)**: Only 67% success - very unstable with catastrophic failures

**⚠️ Variance problem persists:**
- Same config (lr=0.0003, (64,64), 500ep) produced 2.7% to 99% win rate across 10 seeds
- Even "good" configs can fail catastrophically with unlucky random seed
- Larger networks (256,256) amplify instability

**Recommendation**: Use (128,128) network for balance of performance and generalization.

## Next steps to try (Prioritized)

### 🔴 PRIORITY 0: Experience Replay (batch_size > 1) - COMPLETED ✅
**Critical finding from grid search**: Online learning (batch_size=1) is fundamentally unstable even in pure random phase.

**Grid Search Results (Dec 2025) - Phase 2 Completed (batch_size sweep):**
- ✅ Tested batch_size: [8, 32, 64] with experience replay
- ✅ Fixed config: lr=0.0003, network=(64,64), epsilon_decay=200, temp=1.0, 300 episodes random phase
- ❌ Did not test self-play (stayed in random phase only)

**Key findings - Experience replay improves stability but issues remain:**
| Batch | Ep 100 | Ep 200 | Ep 300 | Robustness | P(bet\|AA) | Sanity |
|-------|--------|--------|--------|------------|----------|---------|
| 8     | 30.3%  | 28.3%  | **40.3%** ✅ | 5.2% ✅ | 72.6% | ❌ |
| 32    | 25.3%  | 37.7%  | 32.0%  | **5.0%** ✅ | 61.0% | ❌ |
| 64    | 24.7%  | **8.7%** ⬇️ | 31.0%  | 9.4% ❌ | 66.9% | ❌ |

**Critical insights:**
1. **✅ Experience replay WORKS**: Robustness improved from 30% std (batch_size=1) to ~5% std (batch_size=8,32)
2. **⚠️ Larger ≠ Better**: batch_size=64 showed catastrophic collapse at ep 200 (8.7% win rate!)
   - Suggests larger batches may cause underfitting or stale gradients in this setting
3. **🏆 Best performer**: batch_size=8 achieved 40% win rate with 5.2% std
   - Sweet spot between stability and learning speed
4. **❌ All failed sanity checks**: P(bet|AA) ranged 61-73%, below 85% threshold
   - Issue is deeper than just batch_size - may need better reward shaping or longer training
5. **Stability vs Performance tradeoff**:
   - batch_size=32: Most stable (5.0% std) but lower final win rate (32%)
   - batch_size=8: Slightly less stable but best final performance (40.3%)

**Previous Grid Search - Phase 1 (learning_rate sweep with batch_size=1):**
| LR | Ep 100 | Ep 200 | Ep 300 | Robustness | P(bet\|AA) |
|----|--------|--------|--------|------------|----------|
| 0.0001 | 24% | 10% ⬇️ | 3% ⬇️ | 8.7% std | 42% ❌ |
| 0.0003 | 40% | 18% ⬇️ | **90%** ⬆️ | **30.1% std** ❌ | 97% |
| 0.001 | 34% | 78% ⬆️ | 50% ⬇️ | 18.2% std | 65% |

**Next Steps (PRIORITY 0 continued):**
1. **Investigate sanity check failures**: Why does P(bet|AA) stay < 85% even with stable training?
   - Possible causes: insufficient exploration, poor reward signal, need longer training
   - Try: Extended training (500-1000 episodes), reward shaping for premium hands
2. **Test batch_size=16**: May be sweet spot between 8 (high performance) and 32 (high stability)
3. **Add batch_size=1 to comparison**: Confirm that experience replay is improvement over online learning
4. **Investigate batch_size=64 collapse**: Why did it catastrophically fail at ep 200?
   - Check if replay buffer size matters (currently defaults based on batch_size)
   - May need to tune buffer_size independently from batch_size

**Success criteria for moving to self-play:**
- Frozen win rate > 40% consistently across multiple runs ✅ (achieved with batch_size=8)
- Robustness std < 10% ✅ (achieved with batch_size=8,32)
- P(bet|AA) > 85% at all checkpoints ❌ (best: 72.6%)
- Sanity checks pass consistently ❌

### 🔴 PRIORITY 1: Fix Catastrophic Forgetting During Self-Play (Previously PRIORITY 0)
**Problem**: Agent learns well vs RandomAgent (38% win rate) but collapses during self-play (folds premium hands)

**Root causes identified:**
1. **Epsilon-greedy + SARSA + self-play = toxic combination**
   - Epsilon-greedy injects random actions into Q-learning targets
   - During self-play, this creates unstable multi-agent feedback loops
   - Q-values drift negative → conservative play → collapse
2. **Too much self-play too quickly** - abrupt transitions (100% random → 80% self-play)
3. **Learning rate too high** - rapidly overwrites learned strategies

**Fixes implemented (testing in progress):**
- **✅ DONE: Remove epsilon-greedy during self-play**:
  - Phase 1 (random): Keep epsilon decay for exploration
  - Phase 2 & 3 (mixed/self-play): Epsilon = 0, rely ONLY on softmax
  - Prevents epsilon from corrupting SARSA Q-targets
- **✅ DONE: Smoother curriculum with higher anchor percentage**:
  - Phase 1: 100% RandomAgent (500 episodes)
  - Phase 2: 50% self-play, 50% Random/Skillful (100 episodes)
  - Phase 3: 70% self-play, 30% Random/Skillful (never 100%!)
  - Much gentler transition prevents sudden collapse
- **✅ DONE: Lower learning rates**:
  - Phase 1: LR=0.001 (fast initial learning)
  - Phase 2: LR=0.0002 (5x lower, preserve strategies)
  - Phase 3: LR=0.0001 (10x lower, very conservative)
- **✅ DONE: SkillfulRandomAgent**: Provides explicit signal that premium hands have value

**Note**: Grid search (new PRIORITY 0) should be done first to find stable hyperparameters for random phase before attempting self-play again

### ✅ COMPLETED: Softmax Policy + Expanded State Space
- **✅ DONE: Reduce initial wealth**: Now randomized $15-$35, episodes finish quickly
- **✅ DONE: Curriculum learning**: Random → Mixed → Self-play successfully implemented
- **✅ DONE: Fix Q-value anomaly**: Fixed by adding opponent wealth to state + randomizing initial wealth
- **✅ DONE: Softmax policy implementation**: Replaced argmax with softmax sampling, enables mixed strategies
- **✅ DONE: Retrain with softmax**: 400 episodes completed, all sanity checks pass, 68% win rate
- **✅ DONE: Expand state space**: Added pot size + opponent fold status (now 19 inputs)
- **✅ DONE: Extended random phase**: Now 500 episodes vs random (was 100) for solid foundation
- **✅ DONE: Gradual mixing ratios**: 100% random → 80% self-play → 90% self-play (never 100%)
- **✅ DONE: Learning rate decay schedule**: 0.001 → 0.0005 → 0.0002 by curriculum phase

### 🟡 PRIORITY 2: True Self-Play & Nash Equilibrium Convergence
- **Current limitation**: Self-play win rate is 65-68% instead of 33% (Nash equilibrium)
  - **Root cause**: Not true self-play! Learning agent updates Q-function during episode, opponents frozen
  - Opponents copy weights at episode start, then learning agent improves during episode
  - Learning agent fights "slightly outdated versions of itself" → persistent edge
- **Fix option A - Frozen evaluation**: Add separate eval episodes with no updates/exploration
  - Measure win rate with frozen policies to check Nash convergence
  - Current 65-68% is "training performance", not "equilibrium performance"
- **Fix option B - Multi-agent RL**: All 3 agents update simultaneously
  - More complex but would enable true Nash convergence
  - Would require tracking separate Q-functions for each position

### 🟡 PRIORITY 3: Investigate Win Rate Decline (May be addressed by grid search)
- **Observation**: Win rate vs random drops from ~78% (early) to ~67% (later episodes)
  - Could indicate more conservative/less exploitable play (good!)
  - Could indicate underfitting or need for architecture changes
  - Profile Q-values over time: are they converging or diverging?
  - Try different learning rates (currently using default Adam)
  - Try different network architectures (current: 4 layers, 64 units each)
  - Train for more episodes (1000+) to see if it's just variance
- **Temperature tuning**: Currently using temperature=1.0, experiment with decay schedule
  - Start high (exploration) → decay to lower values (exploitation)
  - May help convergence
- **✅ DONE: Simplified exploration**: Removed redundant epsilon-greedy during self-play
  - Random phase: Uses epsilon-greedy for exploration (decays ~100% to ~2%)
  - Mixed/self-play: Uses ONLY softmax temperature (epsilon=0)
  - Prevents epsilon from corrupting SARSA Q-targets in multi-agent learning
  - Temperature stays constant at 1.0 (poker Nash equilibria require mixed strategies)

### State representation improvements
- **✅ DONE: Track opponent wealth**: Now included in state representation
- **✅ DONE: Fix public cards encoding** - Agent can now see turn & river cards!
  - **Fixed**: Now encodes all 5 public cards (10 features: 5 cards × 2 per card)
  - **Impact**: State dimension changed from 18 to 22 features for 3 players
  - **Note**: Invalidates all previous grid search results (different state space)
  - **Added tests**: `test_state_representation_size`, `test_own_wealth_index`, `test_public_cards_all_stages`
  - **Verified**: own_wealth index remains at 5 (pre-training unaffected)
- **Encode pot odds**: Add (pot_size / cost_to_call) to private state - crucial for rational betting
- **Track opponent aggression**: Count opponent bets/raises per stage as features
- **Better hand strength encoding**: Instead of raw rank/suit, include pre-computed hand strength percentile
- **Relative wealth ratios**: Try (own_wealth / total_wealth) instead of absolute values
- **Position encoding**: Add button position feature (SB/BB/dealer disadvantage)

### Advanced training techniques
- **✅ PRIORITY 0: Experience replay**: Store (s,a,r,s') transitions and sample mini-batches instead of pure online SARSA
  - Grid search revealed batch_size=1 is fundamentally unstable (see PRIORITY 0 above)
  - This is now the top priority after discovering online learning instability
- **🌟 Pre-training with heuristic Q-values** (HIGH PRIORITY - could solve variance problem!)
  - **Motivation**: Random initialization causes massive variance (30% vs 99% with same config!)
  - **Idea**: Bootstrap Q-function with simple poker heuristics BEFORE RL training
  - **Common-sense heuristics to pre-train:**
    1. Higher wealth is monotonically better: Q(wealth=$30) > Q(wealth=$20)
    2. Premium hands are strong: Q(AA, bet) >> Q(AA, fold)
    3. Trash hands are weak: Q(7-2, fold) > Q(7-2, bet $3)
    4. Pot odds matter: Q(call with flush draw, pot=$10) > Q(call with flush draw, pot=$2)
    5. Position matters: Q(button, raise) > Q(early position, raise) for marginal hands
  - **Implementation options:**
    - **Option A (Supervised pre-training)**: Create synthetic (state, action, Q-value) tuples with handcrafted targets, train network to fit them (10-100 epochs) before SARSA
    - **Option B (Rule-based expert)**: Implement SkillfulHeuristicAgent that plays based on hand strength + pot odds, run 50-100 episodes, use its trajectories as initial Q-targets
    - **Option C (Smart initialization)**: Initialize final layer weights/biases to output Q∝(hand_strength + wealth + pot_odds) instead of random
    - **Option D (Imitation learning)**: Run behavioral cloning on expert demonstrations first, then fine-tune with SARSA
  - **Benefits:**
    - ✅ All runs start from same reasonable baseline → **massively reduces variance**
    - ✅ Faster convergence (fewer episodes wasted learning "wealth is good")
    - ✅ Less likely to fall into pathological local minima (e.g., "fold everything")
    - ✅ More sample efficient - use domain knowledge instead of random exploration
  - **Challenges:**
    - How to design heuristics without overfitting to human-like play?
    - Might bias away from Nash equilibrium (if heuristics are exploitable)
    - Adds complexity to training pipeline
    - Need to validate that pre-trained model actually helps (ablation study)
  - **Experiment design:**
    1. Pre-train on heuristics for N epochs (tune N)
    2. Evaluate pre-trained Q-function (should beat pure random, fail vs SkillfulRandom)
    3. Fine-tune with SARSA vs RandomAgent (should converge faster than random init)
    4. Compare variance: Run 10 seeds with pre-training vs 10 without
    5. Success metric: Variance reduction >50% AND faster convergence
  - **Related work**: This is similar to curriculum learning, imitation learning, and reward shaping in RL
- **Add target network**: Freeze Q-target periodically to reduce instability from chasing moving target
- **Tune learning rate**: Currently using default Keras optimizer - try lower LR or decay schedule
  - **Phase-specific learning rates**: Decrease LR during pure self-play phase to stabilize convergence (self-play is more sensitive to overwriting previously learned strategies)
- **Reduce exploration more gradually**: Current schedule decays fast (exp(-ep/200)), try slower decay or epsilon-greedy with higher floor
- **Self-play evolution**: Instead of copying weights, maintain a pool of past agents and sample opponents
- **Variance reduction**: Track episode returns and use baseline/advantage estimation
- **✅ DONE: Extended mixed-play curriculum**: Now 500 random → 100 mixed (80% self-play) → remainder at 90% self-play
- **✅ DONE: Gradual mixing ratios**: Implemented 100% → 80% → 90% (never 100%) to prevent catastrophic forgetting
- **✅ DONE: SkillfulRandomAgent for catastrophic forgetting prevention**: Implemented agent that randomizes most actions BUT has hand-strength-aware folding
  - Never folds strong hands (any pair of face cards, anything with 1+ aces, etc.)
  - Otherwise plays randomly (uniform distribution over legal actions)
  - **Key insight**: Always maintain small probability (5-10%) of facing SkillfulRandomAgent, even during pure self-play
  - Prevents learning agent from "forgetting" how to exploit weak opponents (common issue in pure self-play)
  - Acts as regularization: ensures learned policy generalizes beyond self-play equilibrium

### Reward shaping experiments
- **Terminal reward for winning**: Add small bonus for surviving (currently only intermediate wealth changes counted)
- **Immediate hand strength feedback**: Give small intermediate rewards for improving hand on flop/turn/river
- **Penalize dominated play**: Negative reward for obviously bad moves (folding AA pre-flop, calling with 7-2)

### Bug fixes & correctness
- **Fix pre-flop stage completion**: Big blind should get option to raise (TODO on line 175 in state.py)
- **Handle ties correctly**: Multiple winners should split pot (TODO on line 295 in state.py)
- **Low-wealth edge cases**: Add pytest for when player can't match minimum bet
- **✅ DONE: Two-pair tie breaking**: Implemented proper tie-breaking logic for two-pair hands

### Architecture & model changes
- **Transformer architecture**: Replace current MLP with transformer to better capture sequential betting patterns and attention over cards
- **Bigger network**: Current model might be too small - try more layers/units
- **Dueling DQN**: Separate value and advantage streams
- **Recurrent model**: LSTM to capture betting patterns across the hand history

### Evaluation & analysis
- **✅ DONE: Action distribution tracking**: Now logging fold/check/call/raise percentages
- **✅ DONE: Sanity checks**: Premium hands don't fold, agent prefers check over fold, Q-values ordered correctly
- **✅ DONE: Compare to random**: Successfully beats random play at 80% win rate
- **Frozen 3-way self-play equilibrium test**: Critical sanity check for Nash convergence
  - Take learned weights, freeze policy (no updates, no exploration)
  - All 3 players use identical frozen policy
  - Play 1000+ games and measure win rates
  - **Expected**: Each player wins ~33.3% (with statistical variance)
  - **Purpose**: Verify (1) no positional bias bugs, (2) true equilibrium convergence
  - If one player consistently wins more, indicates bug in game logic or position handling
- **Head-to-head tournaments**: Pit different checkpoint models against each other
- **Exploitability metric**: Measure how far from Nash equilibrium (hard, but approximations exist)
- **Action distribution by context**: Plot fold/check/call/raise frequencies by hand strength and pot size
- **Compare to rule-based heuristics**: Test against tight-aggressive, loose-passive strategies
- **Visualize Q-values**: Heatmap of Q(s,a) across hand strength and pot size

### Code quality
- **✅ DONE: Test coverage for game dynamics**: Tests for betting, folding, stage transitions, ties, low wealth
- **Better pytest coverage for training**: Add tests for reward calculation and SARSA updates
- **Move sanity checks to pytest**: Convert the learned strategy tests into proper unit tests
- **Config file**: Pull hyperparameters (learning rate, exploration, max_deals) into YAML/JSON
- **Logging**: Use proper logging instead of prints, track metrics to tensorboard/wandb
- **Type hints**: Add typing throughout for better IDE support
- **Suppress model summary spam**: get_model() prints summary every time - only print once

### Performance optimization (Low priority - training is fast enough)
- **Profile the self-play loop**: Identify bottlenecks in run_one_episode and state.update
- **Vectorize state encoding**: Batch the get_private_state calls if possible
- **Model prediction batching**: Instead of predict() per action, batch predictions