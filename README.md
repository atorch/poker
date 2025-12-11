# poker

```bash
sudo docker build ~/poker --tag=poker_docker
sudo docker run -it -v ~/poker:/home/poker poker_docker bash
pip install -e .
python poker/play.py
```

## Next steps to try

### Training stability & convergence
- **Reduce initial wealth**: Lower from 100 to 20-30 so players go broke faster → episodes end faster → faster learning
- **Curriculum learning - beat random first**: Start against agents that play completely at random, verify we can learn to beat them before self-play
- **Delayed self-play**: Only start self-play after we've learned to beat random agents - don't try to learn everything at once
- **Reduce exploration more gradually**: Current schedule decays fast (exp(-ep/200)), try slower decay or epsilon-greedy with higher floor
- **Fix Q-value anomaly**: Worse hands (7-2) show *lower* Q-values than premium hands (AA), which is backwards - investigate sign error or state encoding issue
- **Try experience replay**: Store (s,a,r,s') transitions and sample mini-batches instead of pure online SARSA
- **Add target network**: Freeze Q-target periodically to reduce instability from chasing moving target
- **Tune learning rate**: Currently using default Keras optimizer - try lower LR or decay schedule

### State representation improvements
- **Encode pot odds**: Add (pot_size / cost_to_call) to private state - crucial for rational betting
- **Track opponent aggression**: Count opponent bets/raises per stage as features
- **Better hand strength encoding**: Instead of raw rank/suit, include pre-computed hand strength percentile
- **Relative wealth**: Currently track absolute wealth, try ratio (own_wealth / total_wealth)
- **Position encoding**: Add button position feature (SB/BB/dealer disadvantage)

### Reward shaping & training
- **Immediate hand strength feedback**: Give small intermediate rewards for improving hand on flop/turn/river
- **Penalize dominated play**: Negative reward for obviously bad moves (folding AA pre-flop, calling with 7-2)
- **Curriculum learning**: Start with heads-up (2 players), then add 3rd player once converged
- **Self-play evolution**: Instead of copying weights, maintain a pool of past agents and sample opponents
- **Variance reduction**: Track episode returns and use baseline/advantage estimation

### Bug fixes & correctness
- **Fix pre-flop stage completion**: Big blind should get option to raise (TODO on line 175 in state.py)
- **Handle ties correctly**: Multiple winners should split pot (TODO on line 295 in state.py)
- **Low-wealth edge cases**: Add pytest for when player can't match minimum bet
- **Fix blind reward timing**: Reward calculation gets confusing when learning player is forced to post blind (TODO on line 70 in play.py)
- **Two-pair tie breaking**: Verify new tie-breaking logic for two-pair hands

### Architecture & model changes
- **Transformer architecture**: Replace current MLP with transformer to better capture sequential betting patterns and attention over cards
- **Bigger network**: Current model might be too small - try more layers/units
- **Dueling DQN**: Separate value and advantage streams
- **Softmax policy instead of argmax**: Output action probabilities and sample from them (e.g., raise 80% / call 20%) - this is part of optimal poker strategy, not just exploration. Poker Nash equilibria are mixed strategies!
- **Recurrent model**: LSTM to capture betting patterns across the hand history

### Evaluation & analysis
- **Head-to-head tournaments**: Pit different checkpoint models against each other
- **Exploitability metric**: Measure how far from Nash equilibrium (hard, but approximations exist)
- **Action distribution analysis**: Plot fold/check/bet frequencies by position and hand strength
- **Compare to heuristics**: Test against rule-based opponents (tight-aggressive, loose-passive, random)
- **Visualize Q-values**: Heatmap of Q(s,a) across hand strength and pot size

### Code quality
- **Better pytest coverage**: Comprehensive tests for game dynamics (folding, betting, stage transitions) and reward calculation to catch bugs
- **Move sanity checks to pytest**: Convert the learned strategy tests into proper unit tests
- **Separate eval from training**: Don't run sanity checks during training loop, only at end
- **Config file**: Pull hyperparameters (learning rate, exploration, max_deals) into YAML/JSON
- **Logging**: Use proper logging instead of prints, track metrics to tensorboard/wandb
- **Type hints**: Add typing throughout for better IDE support

### Performance optimization
- **Profile the self-play loop**: Identify bottlenecks in run_one_episode and state.update
- **Vectorize state encoding**: Batch the get_private_state calls if possible
- **Model prediction batching**: Instead of predict() per action, batch predictions
- **Low-hanging fruit**: Quick wins for speed without premature optimization (e.g., avoid redundant calculations, cache expensive operations)