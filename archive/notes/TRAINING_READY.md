# Full Curriculum Training - Ready to Launch

## Configuration Summary

**Best configuration from medium grid search:**
- **Learning rate**: 0.0003
- **Network**: (128, 128) - two hidden layers with 128 units each
- **Batch size**: 8 (experience replay)
- **Temperature**: 1.0 (softmax exploration)
- **Epsilon decay**: 200
- **Pre-training**: ENABLED (wealth heuristic)

**Performance (trained only vs RandomAgent):**
- vs RandomAgent: 97.7% win rate
- vs SkillfulRandomAgent: 88.7% win rate ← Best generalization in grid!

## Curriculum Stages (1500 episodes total)

### Phase 1: Pure Random (episodes 0-800)
- **Opponents**: 100% RandomAgent
- **Goal**: Exploit pure randomness, establish baseline
- **Learning rate**: 0.0003
- **Epsilon**: Decays from ~100% → ~2%

### Phase 2: Mixed Opponents (episodes 800-1200)
- **Opponents**: 50% self-play, 50% RandomAgent/SkillfulRandomAgent
- **Goal**: Learn generalized strategy, not just exploit random folding
- **Learning rate**: 0.00006 (0.2× reduced)
- **Epsilon**: 0.0 (softmax-only)

### Phase 3: Self-Play with Anchoring (episodes 1200-1500)
- **Opponents**: 70% self-play, 30% RandomAgent/SkillfulRandomAgent
- **Goal**: Develop advanced strategies while preventing collapse
- **Learning rate**: 0.00003 (0.1× reduced)
- **Epsilon**: 0.0 (softmax-only)

## Key Concepts Documented

### Exploitation vs Generalization (now in README and code)
Training only vs RandomAgent can teach:
1. **Exploitation**: Wait for random folding → fails vs smart opponents
2. **Generalization**: Bet premium, fold trash → works vs both

SkillfulRandomAgent tests if agent learned generalizable poker strategy by:
- Protecting strong hands (only folds AA/AK/pairs 5% of time)
- Playing weak hands randomly
- Performance drop of ~14-16pp is healthy (indicates generalization)

## How to Run

Open a new terminal and run:

```bash
cd /home/adrian/poker
uv run python run_training.py
```

This will:
- ✅ Display output in terminal (real-time)
- ✅ Save to `training_output_2026_01_02_full_curriculum_128x128_batch8.txt`
- ✅ Run full 1500-episode curriculum
- ✅ Test against both RandomAgent and SkillfulRandomAgent at checkpoints

## Files Changed (staged, not committed)

- ✅ `poker/play.py`: Updated defaults to best config, expanded docstring
- ✅ `README.md`: Added medium grid results + exploitation vs generalization section
- ✅ `run_training.py`: New wrapper for output logging
- ✅ All tests pass (57 passed, 2 skipped)

## Expected Runtime

~30-60 minutes (depending on hardware)
- 1500 episodes
- Checkpoints every 100 episodes
- Evaluation at checkpoints (300 episodes each vs Random + Skillful)

## What to Watch For

1. **Phase 1 (0-800)**: Should reach >80% vs RandomAgent
2. **Phase 2 (800-1200)**: Watch for stable transition, no collapse
3. **Phase 3 (1200+)**: Final win rate >85% vs Random, >75% vs Skillful
4. **P(bet|AA)**: Should stay >95% throughout (sanity check)
5. **Variance**: This is a single seed - may succeed or fail (30% failure rate known)
