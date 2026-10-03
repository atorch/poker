# Reward Structure Analysis

## User's Question

Are there immediate negative rewards for betting that make learning difficult? Should we rework the reward function to only assign rewards when hands resolve?

## Answer: No, the current reward structure is correct

### Current Reward Calculation

From `poker/play.py` lines 332-356:

```python
wealth_before_action = state.wealth[learning_player]
state.update(action)
# ... other players act ...
wealth_after_action = state.wealth[learning_player]
reward = wealth_after_action - wealth_before_action
```

### Key Finding: Wealth Only Changes at Hand Resolution

Testing confirms (see `test_reward_structure.py`):

```
--- Player 0 bets $3 ---
Wealth before: $20.00
Wealth after: $20.00
Reward: $0.00  ← NO IMMEDIATE NEGATIVE REWARD!
**Betting $3 caused wealth change of $0.00**
```

**How it works:**
1. When you bet $3, the bet is tracked in `bets_by_stage[stage][player]`
2. Your `wealth` array value does **not** change yet
3. Reward = $0.00 (no wealth change during the hand)
4. Only when the hand resolves (showdown or fold) does `redistribute_wealth_and_reinitialize()` get called
5. At resolution: winners gain pot, losers lose their bets, wealth changes, reward is assigned

### State Already Tracks What We Need

From `poker/agent.py` lines 509-537, the state representation already includes:

- `own_wealth` - current wealth
- `total_bet_by_self` - how much you've committed this hand
- `pot_size` - total pot size
- All public cards, opponent info, etc.

So the agent can already compute:
- How much it's invested
- How much it could win
- The risk/reward ratio

### Why This Design is Good

**Poker-appropriate reward structure:**
- Mirrors real poker: You don't "lose" money when you bet, you invest it in the pot
- Reward comes from winning/losing the hand, not from the action itself
- Matches the actual game dynamics

**SARSA propagation should work:**
- Actions during hand get reward ≈ 0
- Final action before resolution gets full reward (positive or negative)
- With gamma=0.9999, Q-values should propagate backward through the hand
- Early betting actions learn their value through continuation value

## The Real Problem: Not Reward Structure, but Learning Dynamics

The "always fold" pathology isn't caused by immediate negative rewards (which don't exist).

**Actual issues:**
1. **Poor initialization** - Random Q-values don't understand poker basics
   - **FIXED** by improved pre-training (wealth awareness, hand strength, pot odds)

2. **Credit assignment across multiple rounds** - Standard RL challenge
   - Pre-flop bet gets reward = 0
   - Flop bet gets reward = 0
   - Turn bet gets reward = 0
   - River bet gets reward = (win/loss)
   - SARSA must propagate value backward
   - Requires many episodes to learn correctly

3. **Selection bias in opponents** - ConsistentRandomAgent addresses this
   - **FIXED** by preventing cumulative fold exploitation

## Conclusion

**Do NOT rework the reward function.** The current structure is correct for poker.

The user's proposed change (only assign rewards at hand resolution) is **already what we do**!
- Wealth doesn't change during betting
- Rewards are assigned at resolution
- State already tracks total_bet_by_self

**Focus instead on:**
- ✅ Improved pre-training (already implemented)
- ✅ Better opponent mix (ConsistentRandomAgent curriculum)
- Possibly: Increase training episodes to allow better credit assignment
- Possibly: Adjust learning rate schedule

## Test Verification

Run `test_reward_structure.py` to verify:
- Betting causes reward = $0.00 (no immediate penalty)
- Only hand resolution causes wealth/reward changes
- Multi-round betting accumulates bets but doesn't change wealth until resolution
