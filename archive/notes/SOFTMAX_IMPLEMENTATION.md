# Softmax Policy Implementation

## What We Implemented

### 1. Softmax with Temperature Function
**Location:** `poker/agent.py` lines 9-41

**Key Features:**
- Computes probability distribution over Q-values using softmax
- Handles illegal actions (Q = -inf) by assigning them probability 0
- Temperature parameter controls exploration:
  - `temperature → 0`: Greedy (argmax)
  - `temperature = 1.0`: Standard softmax
  - `temperature → ∞`: Uniform random
- Numerically stable implementation (subtracts max for exp calculation)

### 2. Agent Changes
**Location:** `poker/agent.py` lines 45, 49, 309-313

**Changes:**
- Added `temperature` parameter to `__init__` (default: 1.0)
- Replaced `np.argmax(q_values)` with `np.random.choice(actions, p=softmax_probs)`
- Legal action filtering preserved (illegal actions → -inf → probability 0)

### 3. Comprehensive Test Coverage
**Location:** `tests/test_agent.py` lines 41-119

**6 New Tests:**
1. `test_softmax_basic`: Validates probability distribution properties
2. `test_softmax_with_illegal_actions`: Ensures illegal actions get prob 0
3. `test_softmax_temperature_low`: Verifies low temp → greedy behavior
4. `test_softmax_temperature_high`: Verifies high temp → uniform behavior
5. `test_softmax_all_illegal_raises_error`: Error handling
6. `test_softmax_equal_q_values`: Uniform distribution for equal Q-values

**Test Results:** All 19 tests pass ✓

## Why This Matters for Poker

### Nash Equilibrium Requires Mixed Strategies

In poker, optimal play is NOT deterministic. Example with Queen-Jack:
- **Argmax (OLD):** If Q(raise)=10.1 and Q(call)=10.0, ALWAYS raise
- **Softmax (NEW):** Raise ~52%, Call ~48% (probabilistic mix)

This prevents exploitation - opponents can't predict your actions.

### What We Observed with Argmax

From 400-episode training run:
- Episodes 0-100: Beat random agents at 80% (good!)
- Episodes 200-400: Self-play collapsed
  - Agent started folding pocket aces
  - Fold rate increased to 77.9%
  - Episodes dragged to 300+ deals
  - Win rate stayed at 73% instead of converging to 33%

**Root cause:** Argmax + self-play = "folding is safe" feedback loop

## Next Steps

1. **Delete old model:** `rm models/player_0_latest.h5`
2. **Retrain from scratch:** Run with softmax policy
3. **Monitor metrics:**
   - Action distribution (fold/check/call/raise %)
   - Win rate convergence in self-play
   - Episode length stability
   - Q-value sanity checks

4. **Expected outcomes:**
   - More balanced action distribution
   - Win rate → 33% in self-play (Nash equilibrium)
   - Agent learns to play premium hands
   - Mixed strategies emerge (same hand → different actions)

## Temperature Tuning Guide

**Start with temperature=1.0** (standard softmax), then adjust:

- **If agent too random:** Lower temperature to 0.5
  - More greedy, exploits Q-value differences more
  - Useful if losing to random opponents

- **If agent too deterministic:** Raise temperature to 2.0
  - More exploration, more uniform over actions
  - Useful if stuck in local minima

- **Monitor training stability:**
  - Episodes getting longer? May need more folding (lower temp)
  - Episodes too short? May need more calling (higher temp)
  - Action distribution gives good diagnostics
