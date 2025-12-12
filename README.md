# poker

```bash
sudo docker build ~/poker --tag=poker_docker
sudo docker run -it -v ~/poker:/home/poker poker_docker bash
pip install -e .
python poker/play.py
```

## Results Summary (Dec 2025)

**✅ SUCCESS: Softmax Policy + Expanded State Space (Latest Run)**
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
- **Curriculum learning works:** Random → Mixed → Self-play prevents collapse into bad equilibria.
- **Interesting observation:** Win rate vs random play declined from ~78% (early episodes) to ~67% (later episodes). This may indicate:
  - More conservative/optimal play emerging (less exploitable)
  - Exploration decay reducing "lucky" aggressive plays
  - Natural variance (small sample sizes)
  - Could warrant investigation: model architecture, learning rate, or longer training

**⚠️ Previous Failed Experiment (Argmax Policy):**
- Agent collapsed into pathological behavior: folded pocket aces!
- Self-play episodes became extremely long (300+ deals)
- Fold rate increased to 78% by episode 290
- **Root cause:** Argmax deterministic policy + self-play → everyone learns "folding is safe" → bad equilibrium

## Next steps to try (Prioritized)

### ✅ COMPLETED: Softmax Policy + Expanded State Space
- **✅ DONE: Reduce initial wealth**: Now randomized $15-$35, episodes finish quickly
- **✅ DONE: Curriculum learning**: Random → Mixed → Self-play successfully implemented
- **✅ DONE: Fix Q-value anomaly**: Fixed by adding opponent wealth to state + randomizing initial wealth
- **✅ DONE: Softmax policy implementation**: Replaced argmax with softmax sampling, enables mixed strategies
- **✅ DONE: Retrain with softmax**: 400 episodes completed, all sanity checks pass, 68% win rate
- **✅ DONE: Expand state space**: Added pot size + opponent fold status (now 19 inputs)

### 🟡 PRIORITY 1: True Self-Play & Nash Equilibrium Convergence
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

### 🟡 PRIORITY 2: Investigate Win Rate Decline
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
- **Simplify exploration**: Currently using BOTH epsilon-greedy AND softmax (redundant!)
  - Remove epsilon-greedy, use only softmax temperature for exploration
  - Cleaner code, single exploration mechanism
  - **Important**: Temperature should NOT decay to zero! Poker Nash equilibria require mixed strategies
  - Q-function must be learned with the same temperature that will be used at "convergence"
  - Possible schedules:
    - Constant temperature=1.0 (simplest, may be sufficient)
    - Decay with floor: temperature = max(1.0, 2.0 * exp(-episode/200)) → decays from 2.0 to 1.0
    - Start high for exploration, but maintain nonzero temperature for stochastic play

### State representation improvements
- **✅ DONE: Track opponent wealth**: Now included in state representation
- **Encode pot odds**: Add (pot_size / cost_to_call) to private state - crucial for rational betting
- **Track opponent aggression**: Count opponent bets/raises per stage as features
- **Better hand strength encoding**: Instead of raw rank/suit, include pre-computed hand strength percentile
- **Relative wealth ratios**: Try (own_wealth / total_wealth) instead of absolute values
- **Position encoding**: Add button position feature (SB/BB/dealer disadvantage)

### Advanced training techniques
- **Try experience replay**: Store (s,a,r,s') transitions and sample mini-batches instead of pure online SARSA
- **Add target network**: Freeze Q-target periodically to reduce instability from chasing moving target
- **Tune learning rate**: Currently using default Keras optimizer - try lower LR or decay schedule
- **Reduce exploration more gradually**: Current schedule decays fast (exp(-ep/200)), try slower decay or epsilon-greedy with higher floor
- **Self-play evolution**: Instead of copying weights, maintain a pool of past agents and sample opponents
- **Variance reduction**: Track episode returns and use baseline/advantage estimation

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