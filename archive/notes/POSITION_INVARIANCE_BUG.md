# Position Invariance Bug Analysis

## 🐛 Root Cause: Player 0 Wins 70% Instead of 33%

When all 3 players use identical frozen policies, player 0 wins ~70% instead of the expected ~33%. This indicates a **position-dependent state representation bug**.

## The Bug

In `agent.py:230-243`, opponent data is ordered by **absolute player index**:

```python
opponent_wealths = [
    game_state.wealth[i]
    for i in range(len(game_state.wealth))
    if i != self.player_index  # ❌ BUG: Absolute index order!
]
```

### What This Means

**Player 0's state vector:**
```
[..., wealth[1], wealth[2], active[1], active[2]]
      ^^^^^^^^    ^^^^^^^^    ^^^^^^^^^  ^^^^^^^^^
      opponent 1  opponent 2  opp1 active opp2 active
```

**Player 1's state vector:**
```
[..., wealth[0], wealth[2], active[0], active[2]]
      ^^^^^^^^    ^^^^^^^^    ^^^^^^^^^  ^^^^^^^^^
      opponent 0  opponent 2  opp0 active opp2 active
```

**Player 2's state vector:**
```
[..., wealth[0], wealth[1], active[0], active[1]]
      ^^^^^^^^    ^^^^^^^^    ^^^^^^^^^  ^^^^^^^^^
      opponent 0  opponent 1  opp0 active opp1 active
```

### Why This Breaks Self-Play

The neural network was **trained exclusively as player 0**, so it learned:
- "Feature at index 13 (first opponent wealth) = player 1's wealth"
- "Feature at index 14 (second opponent wealth) = player 2's wealth"
- Patterns like: "When first opponent has low wealth, bluff more"

When player 1 uses the **same weights**:
- Feature at index 13 now = player 0's wealth (not player 1!)
- Feature at index 14 now = player 2's wealth
- The learned pattern no longer makes sense

**Result:** Player 1 and 2 make worse decisions because their state representations don't match what the model was trained on.

## The Fix

Replace absolute indexing with **relative position ordering**:

```python
# Option 1: Order by relative position from current player
opponent_wealths = [
    game_state.wealth[(self.player_index + offset) % self.n_players]
    for offset in range(1, self.n_players)
]

# Option 2: Order by relative position from dealer (poker convention)
opponent_wealths = []
for offset in range(1, self.n_players):
    opp_idx = (self.dealer + offset) % self.n_players
    if opp_idx != self.player_index:
        opponent_wealths.append(game_state.wealth[opp_idx])
```

With relative ordering, **all players see equivalent state vectors** for equivalent situations.

## Secondary Issue: Dealer Always Starts at 0

In `run_frozen_episode`, we create State without specifying `initial_dealer`, so it defaults to 0:

```python
state = State(n_players=len(players), initial_wealth=initial_wealth, verbose=False)
```

**Fix:**
```python
initial_dealer = np.random.randint(0, len(players))
state = State(n_players=len(players), initial_wealth=initial_wealth,
              initial_dealer=initial_dealer, verbose=False)
```

This ensures no systematic first-deal advantage.

## How to Verify the Fix

Run the frozen self-play test after fixing:

```bash
python poker/play.py
```

**Expected after fix:**
```
[Validation 2] Frozen self-play equilibrium test...
  Player 0 win rate: 33.5% ± 3.0% (95% CI)
  ✓ PASS: Win rate consistent with fair share
```

**Before fix (current):**
```
[Validation 2] Frozen self-play equilibrium test...
  Player 0 win rate: 70.0% ± 3.0% (95% CI)  # ❌ WAY too high!
  ⚠️  WARNING: Significant deviation from fair share!
```

## Test Coverage

Run `test_position_invariance.py` to verify:
- Dealer rotation works correctly ✓
- Deck is shuffled each deal ✓
- Opponent ordering bug is documented (test currently passes, documenting buggy behavior)
- After fix: Update test to verify position invariance

```bash
pytest tests/test_position_invariance.py -v
```
