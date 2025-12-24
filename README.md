# poker

## Quick Start with uv (Recommended)

```bash
# Install dependencies and run
uv run python poker/play.py

# Run tests
uv run pytest tests/

# Run specific test with verbose output
uv run pytest tests/test_simple_stage_transition.py -v -s
```

## Alternative: Docker Setup

```bash
sudo docker build ~/poker --tag=poker_docker
sudo docker run -it -v ~/poker:/home/poker poker_docker bash
pip install -e .
python poker/play.py
```

## Results Summary (Dec 2025)

**❌ FAILURE: Pure Self-Play Causes Catastrophic Forgetting (Latest Run)**
- Started strong: 37.7% win rate after 100 episodes vs RandomAgent (beats 33.3% fair share)
- **Catastrophic collapse during self-play transition:**
  - Episode 100-200 (mixed): Win rate crashed to 26%
  - Episode 200-400 (pure self-play): 33.5% win rate, but learned pathological strategy
- **Final sanity checks all FAIL:**
  - Premium hands (AA, KK, AK) now prefer FOLDING ❌
  - Q(fold) > Q(check) even when checking is free ❌
  - All Q-values negative (-0.48 to -0.90), agent learned to fold everything
- **Root cause**: Pure 100% self-play → "everyone folds" bad equilibrium
  - Agent completely forgot how to exploit RandomAgent
  - Self-play feedback loop: if opponents fold more → folding becomes "safe" → everyone folds
- Action distribution: 31% fold at end (but folding premium hands!)

**Previous SUCCESS: Softmax Policy + Expanded State Space**
- **68% overall win rate** across 400 episodes (fair share: 33%)
- Curriculum learning stages all completed successfully:
  - Random phase (0-100): 67-78% win rate vs random play
  - Mixed phase (100-200): 66-69% win rate
  - Self-play (200-400): 65-68% win rate
- **All sanity checks pass:**
  - Premium hands (AA, KK, AK) prefer betting over folding ✓
  - Q(check) > Q(fold) when checking is free ✓
  - Q-values properly ordered by hand strength ✓
- Reasonable action distribution: 31% fold, 62% positive bets
- Fast convergence: avg 18 deals/episode, median 15 deals

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

## Next steps to try (Prioritized)

### 🔴 PRIORITY 0: Experience Replay (batch_size > 1) - NEXT
**Critical finding from grid search**: Online learning (batch_size=1) is fundamentally unstable even in pure random phase.

**Grid Search Results (Dec 2025) - Phase 1 Completed:**
- ✅ Tested learning rates: [0.0001, 0.0003, 0.001]
- ✅ Fixed config: network=(64,64), epsilon_decay=200, temp=1.0, 300 episodes random phase
- ❌ Only tested batch_size=1 (online learning)
- ❌ Did not test self-play (stayed in random phase only)

**Key findings - Extreme instability with batch_size=1:**
| LR | Ep 100 | Ep 200 | Ep 300 | Robustness | P(bet\|AA) |
|----|--------|--------|--------|------------|----------|
| 0.0001 | 24% | 10% ⬇️ | 3% ⬇️ | 8.7% std | 42% ❌ |
| 0.0003 | 40% | 18% ⬇️ | **90%** ⬆️ | **30.1% std** ❌ | 97% |
| 0.001 | 34% | 78% ⬆️ | 50% ⬇️ | 18.2% std | 65% |

**Critical insights:**
1. **Wild instability**: "Best" config (lr=0.0003, 90% final) has worst robustness (30.1% std)
   - Win rate swung 40% → 18% → 90% across checkpoints
   - The 90% result was likely variance, not stable learning
2. **All learning rates problematic with batch_size=1**:
   - Too low (0.0001): Converges to bad policy (fold everything)
   - "Medium" (0.0003): Extremely unstable, lucky checkpoints
   - Too high (0.001): Oscillates wildly, fails sanity checks
3. **Online SARSA fundamentals**: batch_size=1 means every transition immediately updates Q-function
   - Moving Q-targets + high variance poker rewards → instability
   - Agent "chases its own tail" with rapidly changing value estimates

**Next Step (PRIORITY 0):**
Implement **experience replay buffer** and test batch_size > 1 to stabilize learning:
1. Add replay buffer to agent.py (store last N transitions)
2. Modify training loop to:
   - Store (s, a, r, s') transitions in buffer
   - Sample random mini-batches for updates
   - Update Q-function on batches instead of single transitions
3. Grid search batch_size: [1, 8, 32, 128] with best LR from phase 1 (0.0003)
4. Measure if larger batches reduce robustness_std_win_rate

**Success criteria:**
- Frozen win rate progresses smoothly (no wild swings)
- Robustness std < 10% (vs 30% currently)
- Consistent results across multiple runs with same hyperparameters
- P(bet|AA) stays > 85% at all checkpoints

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
- **Encode pot odds**: Add (pot_size / cost_to_call) to private state - crucial for rational betting
- **Track opponent aggression**: Count opponent bets/raises per stage as features
- **Better hand strength encoding**: Instead of raw rank/suit, include pre-computed hand strength percentile
- **Relative wealth ratios**: Try (own_wealth / total_wealth) instead of absolute values
- **Position encoding**: Add button position feature (SB/BB/dealer disadvantage)

### Advanced training techniques
- **✅ PRIORITY 0: Experience replay**: Store (s,a,r,s') transitions and sample mini-batches instead of pure online SARSA
  - Grid search revealed batch_size=1 is fundamentally unstable (see PRIORITY 0 above)
  - This is now the top priority after discovering online learning instability
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