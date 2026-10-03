# River Curriculum Learning - Future Enhancement

**Status:** Idea for future implementation
**Date:** 2026-01-04
**Context:** After implementing improved pre-training, considering additional curriculum strategies

---

## Core Idea

Start some training episodes from **random river situations** (with 5 public cards visible and bets already made) rather than always starting from pre-flop.

This creates scenarios with immediate consequences and faster credit assignment.

## Motivation

### Current Challenge: Delayed Credit Assignment

In normal poker hands:
1. Pre-flop bet → reward = $0.00 (no wealth change)
2. Flop bet → reward = $0.00
3. Turn bet → reward = $0.00
4. River bet → reward = $0.00
5. Showdown/fold → reward = (win/loss)

**Problem:** Early actions only learn value through continuation value propagation with SARSA.
- Requires many episodes to propagate correctly
- Can lead to poor intermediate Q-values
- May contribute to "always fold" pathology

### River Curriculum Solution

Start from river with bets already made:
1. River bet → immediate reward (hand resolves in 1-2 actions)
2. Clear cause-effect: bet with strong hand → win pot immediately
3. Pot-committed learning: "Already invested $X, pot is $Y, hand is strong → must bet"

## Key Benefits

### 1. Faster Credit Assignment
- Hand resolves in 1-2 actions from river start
- Immediate reward signal (no multi-round propagation needed)
- Agent learns "strong hand + large pot = bet" directly

### 2. Pot Odds Learning
- State shows: `total_bet_by_self=$10, pot=$25, hand=strong`
- Clear expected value calculation
- Can't fold for "free" anymore (sunk cost already invested)
- Forces learning: "Don't fold when pot odds favor calling"

### 3. Hand Strength Calibration
- With 5 public cards visible, hand strength is deterministic
- Easier to evaluate: pair < two pair < straight < flush < full house
- No uncertainty about future cards
- Builds foundation for earlier street decisions

### 4. Immediate Negative Reward for Bad Folds
- If you've already bet $10 and pot is $25 with strong hand:
  - Folding → immediate -$10 reward (lose investment)
  - Betting → likely +$25 reward (win pot)
- Clear signal that folding with investment is bad

## Implementation Approach

### State Construction

```python
def create_random_river_state(n_players=3, initial_wealth=20):
    """
    Create a valid game state starting at river with random setup.

    Returns a State object with:
    - game_stage = GameStage.RIVER
    - 5 random public cards dealt
    - Random hole cards for each player
    - Random but realistic bet history
    - Wealth reduced by bets made
    """
    state = State(n_players=n_players, initial_wealth=initial_wealth)

    # Deal 5 public cards (flop + turn + river)
    state.public_cards = state.deal_k_cards(5)
    state.game_stage = GameStage.RIVER

    # Generate random bet history
    for stage in [GameStage.PRE_FLOP, GameStage.FLOP, GameStage.TURN, GameStage.RIVER]:
        for player in range(n_players):
            if stage < GameStage.RIVER:
                # Past rounds: random bets between 0 and 3
                bet = np.random.choice([0, 1, 2, 3])
                state.bets_by_stage[stage][player] = [bet] if bet > 0 else []
            elif stage == GameStage.RIVER:
                # River: might have some bets already, or none
                if np.random.random() < 0.5:
                    bet = np.random.choice([1, 2, 3])
                    state.bets_by_stage[stage][player] = [bet]

    # Adjust wealth to reflect bets made
    # (In real game, wealth changes at hand resolution, but for training
    #  we need to show pot is already committed)
    # Actually, this is tricky - see challenges below

    return state
```

### Challenges to Solve

1. **Wealth consistency:**
   - Real game: wealth doesn't change until resolution
   - Training: need to show "you've already committed $X"
   - **Solution:** Use `total_bet_by_self` in state (already tracked!)
   - Agent sees: wealth=$15, total_bet_by_self=$5, pot=$15
   - Can infer: "I've invested $5, can win $15"

2. **Bet history realism:**
   - Can't have bets that exceed wealth
   - Total bets must be consistent across players (pot size)
   - Betting rounds must follow poker rules (raises, calls)
   - **Solution:** Constraints in random generation, or use recorded game states

3. **State validity:**
   - Must ensure `bets_by_stage` is legal
   - Must ensure no player is all-in incorrectly
   - Must ensure current_player rotation is correct
   - **Solution:** Careful state construction + validation tests

## Curriculum Design Options

### Option 1: River-First Curriculum (Recommended)

Gradually decrease river starts, increase normal starts:

```
Phase 0 (200 eps): 80% river, 20% normal - Learn hand strength fast
Phase 1 (300 eps): 50% river, 50% normal - Transition period
Phase 2 (500 eps): 20% river, 80% normal - Mostly normal play
Phase 3 (500 eps): 0% river, 100% normal - Full game mastery
```

**Advantages:**
- Fast initial learning (hand strength, pot odds)
- Smooth transition to full game
- Early success → better Q-value initialization for full hands

**Disadvantages:**
- Distribution shift between phases
- Need to ensure generalization

### Option 2: Stage-Progression Curriculum

Start from late stages, work backward to pre-flop:

```
Phase 0 (150 eps): 100% river starts (5 cards visible)
Phase 1 (150 eps): 100% turn starts (4 cards visible)
Phase 2 (200 eps): 100% flop starts (3 cards visible)
Phase 3 (500 eps): 100% pre-flop starts (0 cards visible)
Phase 4 (500 eps): Normal training with opponent mix
```

**Advantages:**
- Natural progression from simple (full info) to complex (hidden info)
- Each stage builds on previous learning
- Mimics "learn to walk before you run"

**Disadvantages:**
- More implementation work (need turn starts, flop starts)
- Longer total training time
- More phases to tune

### Option 3: Persistent River Mixing

Always include some river starts throughout training:

```
Throughout all episodes: 10-20% start at river, 80-90% normal
- Maintains hand strength evaluation
- Provides continuous quick-reward signal
- Prevents forgetting pot odds
```

**Advantages:**
- Simpler implementation (one training loop)
- Continuous reinforcement of hand strength
- Helps prevent catastrophic forgetting

**Disadvantages:**
- May slow down pre-flop learning
- Distribution mismatch might confuse agent
- Less clear curriculum progression

### Option 4: River Post-Training Fine-Tuning

Train normally, then add river episodes at the end:

```
Phase 1-3: Normal training (1500 eps)
Phase 4 (200 eps): 50% river starts - Fine-tune hand strength evaluation
```

**Advantages:**
- Minimal disruption to current training
- Can fix hand strength issues after main training
- Easy to test as add-on

**Disadvantages:**
- Doesn't help with early learning
- May not fix deep Q-value issues
- Feels like a patch rather than a solution

## When to Consider This

**Implement river curriculum if:**
- ✗ Improved pre-training doesn't fix "always fold" pathology
- ✗ Agent still doesn't learn hand strength after many episodes
- ✗ Q(bet|AA) remains lower than Q(fold|AA) even late in training
- ✗ Agent folds good hands with large pots already committed

**Skip river curriculum if:**
- ✓ Improved pre-training successfully teaches hand strength
- ✓ Agent learns to bet strong hands and fold weak hands
- ✓ Win rates improve to >40% vs RandomAgent
- ✓ Training is already working well

## Implementation Checklist

If we decide to implement:

- [ ] Create `create_random_river_state()` function
- [ ] Add validation tests for river state construction
- [ ] Ensure bet history is realistic and consistent
- [ ] Test that rewards work correctly from river starts
- [ ] Add curriculum parameter to `run_one_episode()`
- [ ] Add `river_start_probability` to training config
- [ ] Test Phase 0 with 80% river starts
- [ ] Monitor if hand strength learning improves
- [ ] Compare to baseline (no river curriculum)
- [ ] Document results and iterate

## Alternative: Use Recorded Game States

Instead of random generation, could use real game states from previous training:

```python
# During normal training, occasionally save interesting states
if state.game_stage == GameStage.RIVER and pot_size > 10:
    save_state_snapshot(state, filename="river_states.pkl")

# Later, load and use for curriculum
river_states = load_state_snapshots("river_states.pkl")
state = random.choice(river_states)
```

**Advantages:**
- Guaranteed realistic states
- No validation issues
- Captures actual game distribution

**Disadvantages:**
- Need to train first to collect states
- Limited diversity (only seen situations)
- Storage overhead

## Related Ideas

### 1. Heads-Up Simplification
- Start with 2-player games (simpler)
- Then move to 3-player games
- Easier strategy space initially

### 2. Fixed-Opponent Curriculum
- Start training vs SkillfulRandomAgent only
- Learn to exploit predictable opponents first
- Then add ConsistentRandomAgent and self-play

### 3. Incremental Betting Actions
- Start with just [fold, call, bet $3]
- Add [bet $2] later
- Add [bet $1] last
- Simpler action space initially

## Decision Point

**Current Status (2026-01-04):**
- ✅ Improved pre-training implemented
- ✅ New training run started with v5 (improved pre-training)
- ⏳ Waiting for training results

**Next Steps:**
1. Wait for current training run to complete
2. Analyze results:
   - Does agent learn to bet with strong hands?
   - Is P(bet|AA) > 80%?
   - Does win rate exceed 40% vs RandomAgent?
3. **If YES:** Document river curriculum as "future enhancement" (keep this file)
4. **If NO:** Prioritize river curriculum implementation

---

## Notes

- River curriculum is a form of **curriculum learning** (Bengio et al., 2009)
- Similar to starting AlphaGo from endgame positions
- Common in game AI to start with "easier" subproblems
- May help with sparse reward problem in poker

## References

- Curriculum learning principles
- OpenAI's "learning to walk" progression
- DeepMind's progressive training techniques
- Poker AI literature on hand strength evaluation
