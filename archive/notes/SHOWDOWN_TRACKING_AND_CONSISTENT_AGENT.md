# Showdown Tracking Bug Fix & ConsistentRandomAgent Implementation

## Summary

This document describes two major improvements made on 2026-01-04:

1. **Fixed critical showdown tracking bug** that reported 100% showdowns (impossible)
2. **Implemented ConsistentRandomAgent** to prevent cumulative fold exploitation

---

## Part 1: Showdown Tracking Bug

### The Problem

Training output showed impossible statistics:
```
Showdown frequency:
  Total deals: 29903
  Showdowns (2+ players): 29903 (100.0%)
  Wins by fold (1 player): 0 (0.0%)
```

This was suspicious because:
- Agent was folding constantly (Q(fold) highest for all hands)
- Random agents also fold with some probability
- **100% showdowns is mathematically impossible**

### Root Cause

Classic "check after reset" bug in `poker/play.py`:

```python
# BUG: Check has_folded AFTER it's been reset
if state.n_deals > episode_stats['previous_n_deals']:
    active_players = sum(1 for folded in state.has_folded if not folded)
    # ^^ has_folded was ALREADY reset to [False, False, False]!
```

**Sequence of events:**
1. Deal completes → `redistribute_wealth_and_reinitialize()` called
2. → calls `initialize_pre_flop()` which:
   - Increments `n_deals`
   - Resets `has_folded = [False, False, False]`
3. Back in play.py, we detect `n_deals` increased
4. We check `has_folded` - but it's already reset!
5. Result: Always shows 3 active players → always counts as showdown

### The Fix

Save `has_folded` BEFORE `state.update()`:

```python
# Save has_folded BEFORE state.update() (which may reset it)
episode_stats['previous_has_folded'] = list(state.has_folded)

state.update(action)

# Later, when deal changes:
if state.n_deals > episode_stats['previous_n_deals'] and episode_stats['previous_has_folded'] is not None:
    # Use PREVIOUS has_folded (before reset)
    active_players = sum(1 for folded in episode_stats['previous_has_folded'] if not folded)
    if active_players >= 2:
        episode_stats['showdowns'] += 1
    else:
        episode_stats['wins_by_fold'] += 1
```

### Tests Added

`tests/test_showdown_tracking.py` with 5 tests:
- `test_showdown_tracking_logic_with_showdown`: 2+ players active → showdown
- `test_showdown_tracking_logic_with_fold`: 1 player active → win by fold
- `test_bug_has_folded_reset_after_deal`: Regression test for the bug
- `test_state_has_folded_not_reset_before_deal_end`: State validity
- `test_showdown_definition`: Core logic verification

---

## Part 2: ConsistentRandomAgent Implementation

### The Problem

Agents were learning exploitation strategies instead of generalizable poker:

**v2 ("always bet" pathology):**
- SkillfulRandomAgent folded weak hands 20% per action
- Cumulative probability: P(fold after n actions) = 1 - 0.8^n
  - 3 rounds: 48.8%
  - 5 rounds: 67.2%
- **Agent learned:** "Maximize betting rounds" not "Evaluate hand strength"
- Result: Always bet $3 with any hand, Q-values exploded to 400k+

**v3 ("always fold" pathology - opposite extreme):**
- SkillfulRandomAgent folded weak hands 70% immediately
- Only strong hands stayed in
- **Agent learned:** "Opponents who stay in have strong hands → Always fold"
- Result: Q(fold) highest even for pocket aces, win rate < 33.3%

**Root cause:** Selection bias
- Agent only learns from hands that go to showdown
- With 70% immediate fold for weak hands, showdowns biased toward strong hands
- Agent infers: "If opponent hasn't folded, I'm beaten"

### The Solution: ConsistentRandomAgent

**Key insight:** Instead of i.i.d. decisions per action, **commit to a strategy per deal**.

**Strategy distribution (committed once per deal):**
- **Aggressive (30%):** Plays strong, sometimes continues with weak hands
- **Passive (40%):** Folds weak hands immediately, bets strong hands
- **Bluffing (30%):** Always bets, even with terrible hands

**Why this prevents exploitation:**
1. **No cumulative fold probability:** Fold decision made once upfront
2. **Can't exploit multiple betting rounds:** Strategy already committed
3. **Forces hand evaluation learning:** Agent sees both bluffs AND strong hands at showdown
4. **Diverse training signal:** Sometimes faces aggressive play, sometimes passive, sometimes bluffs

### Implementation

`poker/consistent_random_agent.py` with:
- Strategy commitment per deal (tracked via `current_deal_number`)
- Customizable strategy distribution (`strategy_probs` parameter)
- Hand strength evaluation (matches SkillfulRandomAgent definition)
- Three distinct strategy modes with different behaviors

### Tests Added

`tests/test_consistent_random_agent.py` with 9 comprehensive tests:
- `test_strong_hand_definition`: Verify hand classification
- `test_strategy_commitment_per_deal`: **Core anti-exploitation property**
- `test_strategy_changes_between_deals`: Diverse opponent behavior
- `test_passive_strategy_folds_weak_hands`: Fold weak hands immediately
- `test_aggressive_strategy_rarely_folds_strong_hands`: Protect premium hands
- `test_bluffing_strategy_never_folds`: Maniac player type
- `test_only_selects_legal_actions`: Legality verification
- `test_custom_strategy_distribution`: Tunable difficulty
- `test_prevents_cumulative_fold_exploitation`: Integration test

---

## Changes to Existing Code

### Modified Files

1. **poker/play.py:**
   - Fixed showdown tracking (lines 253-256, 276-286, 332-335)
   - Added `previous_has_folded` tracking

2. **poker/skillful_random_agent.py:**
   - Updated fold probabilities: strong 5%→2%, weak 20%→70%
   - Updated docstring to explain anti-exploitation design

3. **run_training.py & poker/play.py:**
   - Updated run description to "v3_skillful_anti_exploit_gamma9999"

### New Files

1. **poker/consistent_random_agent.py:** Full implementation
2. **tests/test_showdown_tracking.py:** 5 tests
3. **tests/test_consistent_random_agent.py:** 9 tests

---

## Next Steps

### Immediate

1. **Revert SkillfulRandomAgent changes** - The 70% immediate fold created "always fold" pathology
   - Suggest: 40% immediate fold for weak hands (balanced)
   - Or use ConsistentRandomAgent instead

2. **Re-run training** with corrected opponent behavior

3. **Verify showdown tracking** now reports realistic percentages (not 100%/0%)

### Future Curriculum Design

Consider using ConsistentRandomAgent in training curriculum:
- **Phase 1 (200 eps):** RandomAgent only
- **Phase 2 (500 eps):** 50% ConsistentRandomAgent, 50% self-play
- **Phase 3 (800 eps):** 40% ConsistentRandomAgent, 60% self-play

**Advantages:**
- No cumulative fold exploitation
- Diverse showdown scenarios (bluffs + strong hands)
- Forces hand evaluation learning
- More realistic opponent behavior

---

## Testing

All tests pass:
```bash
uv run pytest tests/test_showdown_tracking.py -v       # 5 passed
uv run pytest tests/test_consistent_random_agent.py -v # 9 passed
uv run pytest tests/ -v                                 # 76 passed, 2 skipped
```
