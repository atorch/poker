from datetime import datetime
import numpy as np
import math
import os
import random
import tensorflow as tf

from poker.state import State, GameStage
from poker.agent import Agent, ReplayBuffer
from poker.random_agent import RandomAgent
from poker.skillful_random_agent import SkillfulRandomAgent
from poker.consistent_random_agent import ConsistentRandomAgent
from poker.config import MIN_INITIAL_WEALTH, MAX_INITIAL_WEALTH, TYPICAL_INITIAL_WEALTH, DEFAULT_ACTIONS
from poker.cards import Rank

# Description of this training run for logging/versioning
RUN_DESCRIPTION = "v4 ConsistentRandomAgent Curriculum: Prevents cumulative fold exploitation via strategy commitment per deal. Phase 1: 50% Random, 40% Consistent, 10% Skillful. Phase 2: 60% self-play, 30% Consistent, 10% Skillful. Phase 3: 70% self-play, 30% Consistent (no Random/Skillful). gamma=0.9999 + showdown tracking"


def is_premium_hand(private_cards):
    """
    Check if private cards constitute a premium/strong starting hand.

    Premium hands include:
    - Pocket pairs of face cards or aces (JJ, QQ, KK, AA)
    - Two face cards (any combination of J, Q, K, A)

    Args:
        private_cards: List of 2 Card objects

    Returns:
        bool: True if hand is premium
    """
    if len(private_cards) != 2:
        return False

    ranks = [card.rank for card in private_cards]
    rank_values = [r.value for r in ranks]

    # Face cards are Jack (9), Queen (10), King (11), Ace (12)
    face_card_threshold = Rank.JACK.value

    # Both cards are face cards or aces
    if all(r >= face_card_threshold for r in rank_values):
        return True

    return False


def get_pocket_aces_bet_probability(agent):
    """
    Compute the probability that the agent will bet (not fold) with pocket aces.

    This is a key diagnostic: a well-trained poker agent should almost always
    bet with pocket aces pre-flop. If this probability is low, the Q-function
    has learned something pathological.

    Args:
        agent: The agent to test

    Returns:
        float: Probability of taking a non-fold action with pocket aces (0.0-1.0)
    """
    from poker.cards import Card, Suit
    from poker.agent import softmax_with_temperature

    # Create a test state with pocket aces
    game_state = State(n_players=agent.n_players, initial_wealth=TYPICAL_INITIAL_WEALTH)
    game_state.hole_cards[agent.player_index] = [
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.ACE, Suit.SPADES)
    ]

    # Get Q-values for all actions
    private_state = agent.get_private_state(game_state)
    model_input = agent.get_model_input(private_state, agent.actions)
    q_values = agent.model.predict(model_input, verbose=0)[:, 0]

    # Apply legality constraints (same as in agent.get_action)
    minimum_legal_bet = game_state.minimum_legal_bet()
    maximum_legal_bet = game_state.maximum_legal_bet()

    for index, action in enumerate(agent.actions):
        if action >= 0 and ((action < minimum_legal_bet) or (action > maximum_legal_bet)):
            q_values[index] = -np.inf

    # Compute softmax probabilities
    action_probs = softmax_with_temperature(q_values, agent.temperature)

    # Probability of betting = 1 - P(fold)
    fold_idx = agent.actions.index(-1)
    prob_bet = 1.0 - action_probs[fold_idx]

    return prob_bet


def run_frozen_episode(players, max_deals=50000, min_initial_wealth=MIN_INITIAL_WEALTH, max_initial_wealth=MAX_INITIAL_WEALTH):
    """
    Run a single episode with frozen policies (no learning, no exploration).
    Used for evaluation/validation after training is complete.

    Returns:
        (winner_index, eliminated_index): Indices of winner and eliminated player
                                          (or (-1, -1) if max_deals reached)
    """
    initial_wealth = np.random.randint(min_initial_wealth, max_initial_wealth + 1)
    # Randomize initial dealer to avoid systematic first-deal bias
    initial_dealer = np.random.randint(0, len(players))
    state = State(n_players=len(players), initial_wealth=initial_wealth,
                  initial_dealer=initial_dealer, verbose=False)

    deals = 0
    while not state.terminal:
        deals += 1
        if deals >= max_deals:
            # No winner - episode timed out
            return -1, -1

        current_player = state.current_player

        # Get action with no exploration (proba_random_action=0)
        action = players[current_player].get_action(state, proba_random_action=0.0)
        state.update(action)

    # Find the winner (player with the most wealth) and eliminated player (wealth <= 0)
    winner = int(np.argmax(state.wealth))
    eliminated = int(np.argmin(state.wealth))
    return winner, eliminated


def compute_win_rate_with_ci(wins, n_games, confidence=0.95):
    """
    Compute win rate and confidence interval using normal approximation.

    Args:
        wins: Number of wins
        n_games: Total number of games
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        (win_rate, ci_margin): Win rate and margin of error
    """
    if n_games == 0:
        return 0.0, 0.0

    win_rate = wins / n_games

    # Use normal approximation to binomial (valid for large n)
    # Standard error = sqrt(p * (1-p) / n)
    std_err = math.sqrt(win_rate * (1 - win_rate) / n_games)

    # Z-score for confidence level (1.96 for 95%, 2.576 for 99%)
    z_score = 1.96 if confidence == 0.95 else 2.576
    ci_margin = z_score * std_err

    return win_rate, ci_margin


def evaluate_frozen_agent(agent, opponents, n_episodes=1000, max_deals=50000, progress_interval=0):
    """
    Evaluate a frozen agent (no learning, no exploration) against opponents.

    Args:
        agent: The agent to evaluate (must be at player_index=0)
        opponents: List of opponent agents
        n_episodes: Number of games to play
        max_deals: Maximum deals per episode
        progress_interval: Print progress every N episodes (0 = no progress output)

    Returns:
        dict with keys: wins, survivals, eliminations, win_rate, survival_rate,
                        elimination_rate, ci_margins (dict), timeouts
    """
    players = [agent] + opponents
    wins = 0
    survivals = 0
    eliminations = 0
    timeouts = 0

    for i in range(n_episodes):
        winner, eliminated = run_frozen_episode(players, max_deals=max_deals)

        # Track win (highest wealth)
        if winner == agent.player_index:
            wins += 1

        # Track survival (wealth > 0) and elimination (wealth went to 0)
        if winner == -1:
            timeouts += 1
        elif eliminated == agent.player_index:
            eliminations += 1
        else:
            survivals += 1  # Agent didn't get eliminated, so survived

        # Print progress at specified intervals
        if progress_interval > 0 and (i + 1) % progress_interval == 0:
            current_win_rate = wins / (i + 1)
            print(f"    Progress: {i + 1}/{n_episodes} episodes | Win rate so far: {100 * current_win_rate:.1f}%")

    win_rate, win_ci = compute_win_rate_with_ci(wins, n_episodes)
    survival_rate, survival_ci = compute_win_rate_with_ci(survivals, n_episodes)
    elimination_rate, elim_ci = compute_win_rate_with_ci(eliminations, n_episodes)

    return {
        'wins': wins,
        'survivals': survivals,
        'eliminations': eliminations,
        'win_rate': win_rate,
        'survival_rate': survival_rate,
        'elimination_rate': elimination_rate,
        'ci_margins': {
            'win': win_ci,
            'survival': survival_ci,
            'elimination': elim_ci,
        },
        'timeouts': timeouts,
    }


def run_one_episode(episode, players, max_deals=50000, min_initial_wealth=MIN_INITIAL_WEALTH, max_initial_wealth=MAX_INITIAL_WEALTH, curriculum_random_episodes=100, curriculum_mixed_episodes=100, epsilon_decay_rate=200, replay_buffer=None, gamma=0.99):

    # This is (roughly) Sutton and Barto Figure 6.9
    # page 130, TODO compare to page 131
    # page 244

    # Epsilon-greedy exploration: Use ONLY during random phase
    # During mixed/self-play, rely solely on softmax for exploration
    # This prevents epsilon from corrupting SARSA Q-targets during self-play
    if episode < curriculum_random_episodes:
        # Random phase: Decay epsilon from ~100% to ~2%
        proba_random_action = 0.02 + 0.98 * np.exp(-episode / epsilon_decay_rate)
    else:
        # Mixed/self-play phases: NO epsilon-greedy, only softmax exploration
        proba_random_action = 0.0

    # Randomize initial wealth to give better state coverage across wealth levels
    # Smaller values → faster episodes → faster learning
    initial_wealth = np.random.randint(min_initial_wealth, max_initial_wealth + 1)

    state = State(n_players=len(players), initial_wealth=initial_wealth, verbose=False)

    # Track diagnostics
    episode_stats = {
        'deals': 0,
        'actions': [],  # List of ALL actions taken (learning agent + opponents)
        'learning_agent_actions': [],  # List of ONLY learning agent's actions
        'action_types': {'fold': 0, 'check': 0, 'call': 0, 'raise': 0},  # Detailed action breakdown (all players)
        'hit_max_deals': False,
        'learning_agent_survived': False,  # wealth > 0 at end
        'learning_agent_won': False,  # had highest wealth at end
        'learning_agent_eliminated': False,  # wealth went to 0
        'premium_hands_seen': 0,  # Times agent was dealt premium hand
        'premium_hands_folded': 0,  # Times agent folded premium hand
        'deals_with_premium_checked': set(),  # Track which deals we've already counted premium hands for
        'showdowns': 0,  # Deals ending with 2+ players showing cards
        'wins_by_fold': 0,  # Deals ending with everyone else folding
        'previous_n_deals': 0,  # Track when deals increment to count showdowns
        'previous_has_folded': None,  # Track has_folded before state.update() to detect showdowns correctly
    }

    # The learning agent is always at players[0] with player_index=0
    learning_player = 0

    private_state = players[learning_player].get_private_state(state)
    action = players[learning_player].get_action(state, proba_random_action)

    cumulative_reward = 0

    # Note: at the beginning of every episode, we push updates to the q function
    #  from the learning player to all other Agent players (not RandomAgent)
    for player_index in range(state.n_players):
        if player_index != learning_player and hasattr(players[player_index], 'model'):
            players[player_index].model.set_weights(
                players[learning_player].model.get_weights()
            )

    while not state.terminal:

        # Detect when a deal completes and track showdown vs win-by-fold
        # IMPORTANT: Check BEFORE state.update() so we see has_folded before reset
        if state.n_deals > episode_stats['previous_n_deals'] and episode_stats['previous_has_folded'] is not None:
            # A deal just completed - check if it was a showdown using PREVIOUS has_folded
            # (before initialize_pre_flop reset it)
            active_players = sum(1 for folded in episode_stats['previous_has_folded'] if not folded)
            if active_players >= 2:
                episode_stats['showdowns'] += 1
            else:
                episode_stats['wins_by_fold'] += 1
            episode_stats['previous_n_deals'] = state.n_deals

        # Track deals and check max limit
        episode_stats['deals'] = state.n_deals
        if state.n_deals >= max_deals:
            episode_stats['hit_max_deals'] = True
            state.terminal = True
            break

        # Track action for diagnostics (all actions)
        episode_stats['actions'].append(action)
        # Track learning agent's actions separately
        episode_stats['learning_agent_actions'].append(action)

        # Categorize action type (fold/check/call/raise)
        min_bet = state.minimum_legal_bet()
        is_fold = action < 0
        if is_fold:
            episode_stats['action_types']['fold'] += 1
        elif min_bet == 0 and action == 0:
            episode_stats['action_types']['check'] += 1
        elif action == min_bet:
            episode_stats['action_types']['call'] += 1
        else:  # action > min_bet
            episode_stats['action_types']['raise'] += 1

        # Track premium hand folding (only on PRE_FLOP, at first action of new deal)
        # We track this BEFORE the action is applied so we can count folds
        current_deal = state.n_deals
        if (state.game_stage == GameStage.PRE_FLOP and
            state.current_player == learning_player and
            current_deal not in episode_stats['deals_with_premium_checked']):

            # Mark this deal as checked to prevent double-counting
            episode_stats['deals_with_premium_checked'].add(current_deal)

            # Check if agent has premium hand
            if hasattr(state, 'private_cards') and learning_player < len(state.private_cards):
                player_cards = state.private_cards[learning_player]
                if is_premium_hand(player_cards):
                    episode_stats['premium_hands_seen'] += 1
                    # Note: We only count as folded if the FIRST action is a fold
                    # (subsequent actions after raises don't count)
                    if is_fold:
                        episode_stats['premium_hands_folded'] += 1

        wealth_before_action = state.wealth[learning_player]

        # Save has_folded BEFORE state.update() (which may call initialize_pre_flop and reset it)
        episode_stats['previous_has_folded'] = list(state.has_folded)

        state.update(action)

        # Note: we need to make the other players act so that we can get back to the learning player
        #  Even though the other (non-learning) players are acting, it
        #  is possible that the learning player is forced
        #  to act (bet the small blind or big blind) in this block
        while state.current_player != learning_player and not state.terminal:

            action = players[state.current_player].get_action(
                state, proba_random_action
            )
            episode_stats['actions'].append(action)
            state.update(action)

        # Note: we calculate the player's reward _after_ the other players act
        # TODO This can create odd rewards when the other players fold and
        #  the learning player is either small or big blind in the next round
        wealth_after_action = state.wealth[learning_player]
        reward = wealth_after_action - wealth_before_action
        cumulative_reward += reward
        # print(f" *** Reward for player {learning_player} is ${reward} ***")

        # TODO Put this in pytest
        assert state.wealth[learning_player] == initial_wealth + cumulative_reward

        next_private_state = players[learning_player].get_private_state(state)
        next_action = players[learning_player].get_action(state, proba_random_action)

        if state.terminal:

            # Note: we've reached the end of the episode. There is no continuation value,
            #  and, after updating q, we are done
            print(f"Reached a terminal state after {state.n_deals} deals: player wealths are {state.wealth}")

            # Terminal reward is just the immediate wealth change, NOT including a bonus for winning.
            # This avoids double-counting: we already give intermediate rewards for wealth changes,
            # so adding a "winner bonus" would count the same chips twice.
            # The Q-function should still learn that high-wealth states are valuable through
            # bootstrapping: states with more wealth → survive longer → more future intermediate rewards.
            # (In the future, we might add a small reward for being the last survivor, but for now
            # we keep it simple and let the agent learn wealth value implicitly.)
            updated_guess_for_q = reward

        else:

            continuation_value = players[learning_player].predicted_q(
                next_private_state, next_action
            )
            # SARSA Bellman equation with discount factor gamma
            # gamma < 1.0 prevents Q-value divergence by exponentially decaying future rewards
            # We use gamma=0.99 (light discounting) since poker episodes are finite and fast
            updated_guess_for_q = reward + gamma * continuation_value

        # Update Q-function: either immediately (online) or store in replay buffer (batch)
        if replay_buffer is None:
            # Online learning (batch_size=1): immediate update
            players[learning_player].update_q(private_state, action, updated_guess_for_q)
        else:
            # Experience replay (batch_size>1): store transition for later batch update
            # Store (state, action, reward, next_state) for SARSA-style updates
            replay_buffer.add(private_state, action, reward, next_private_state)

        action = next_action
        private_state = next_private_state

    # Track final outcome for learning agent
    final_wealth = state.wealth[learning_player]

    # Survival: agent has wealth > 0 at end
    if final_wealth > 0:
        episode_stats['learning_agent_survived'] = True

    # Win: agent has highest wealth (or tied for highest)
    max_wealth = max(state.wealth)
    if final_wealth == max_wealth:
        episode_stats['learning_agent_won'] = True

    # Elimination: agent's wealth went to 0
    if final_wealth <= 0:
        episode_stats['learning_agent_eliminated'] = True

    return episode_stats


def test_game_fairness_with_random_agents(n_players=3, n_episodes=1000, max_deals=25_000):
    """
    PRE-TRAINING TEST: Verify game logic has no positional bias.

    All players are RandomAgents, so each should win ~1/n_players of the time.
    If player 0 wins significantly more/less, there's a bug in the game logic.

    This runs before training to catch game logic bugs early.
    """
    print("\n" + "=" * 70)
    print("PRE-TRAINING FAIRNESS TEST: RandomAgent vs RandomAgent vs RandomAgent")
    print("=" * 70)
    print(f"Testing game logic with {n_players} RandomAgents playing {n_episodes} episodes...")
    print(f"Expected: Each player wins ~{100/n_players:.1f}% (fair share)")
    print()

    # Create all RandomAgents with identical behavior
    random_agents = [RandomAgent(player_index=i) for i in range(n_players)]

    # Track wins and eliminations for each player position
    wins_by_position = [0] * n_players
    eliminations_by_position = [0] * n_players
    timeouts = 0

    for episode in range(n_episodes):
        winner, eliminated = run_frozen_episode(random_agents, max_deals=max_deals)

        if winner == -1:
            timeouts += 1
        else:
            wins_by_position[winner] += 1
            eliminations_by_position[eliminated] += 1

        # Print progress
        if (episode + 1) % 200 == 0:
            print(f"  Progress: {episode + 1}/{n_episodes} episodes")
            for i in range(n_players):
                win_rate = wins_by_position[i] / (episode + 1)
                print(f"    Player {i}: {100 * win_rate:.1f}% ({wins_by_position[i]} wins)")

    # Final results
    print(f"\n  Final Results ({n_episodes} episodes):")
    fair_share = 1.0 / n_players

    all_fair = True
    for i in range(n_players):
        win_rate, ci_margin = compute_win_rate_with_ci(wins_by_position[i], n_episodes)
        deviation = abs(win_rate - fair_share)

        print(f"    Player {i}: {100 * win_rate:.1f}% ± {100 * ci_margin:.1f}% ({wins_by_position[i]} wins)")

        # Check if deviation is statistically significant (more than 2 standard errors)
        if deviation > 2 * ci_margin:
            all_fair = False
            print(f"      ⚠️  WARNING: Significant deviation from fair share!")

    if timeouts > 0:
        print(f"    Timeouts: {timeouts}")

    # Also check elimination fairness (each player should be eliminated ~equally)
    print(f"\n  Elimination distribution:")
    games_with_winner = sum(wins_by_position)
    for i in range(n_players):
        elim_rate = eliminations_by_position[i] / games_with_winner if games_with_winner > 0 else 0
        print(f"    Player {i}: {100 * elim_rate:.1f}% ({eliminations_by_position[i]} eliminations)")

    print()
    if all_fair:
        print("  ✓ PASS: All players have fair win rates - no positional bias detected")
    else:
        print("  ❌ FAIL: Positional bias detected in game logic!")
        print("  This must be fixed before training a learning agent.")
        print("  Check: dealer rotation, blind posting, action ordering, state representation")

    print("=" * 70)
    return all_fair


def run_sarsa(
    n_players,
    n_episodes=1500,
    model_path="models/player_0_latest.h5",
    save_interval=10,
    max_deals=5_000,
    curriculum_random_episodes=200,  # REDUCED from 800 to prevent over-fitting to RandomAgent
    curriculum_mixed_episodes=500,   # INCREASED from 400 to get more SkillfulRandomAgent exposure
    learning_rate=0.0003,
    hidden_layers=(128, 128),
    epsilon_decay_rate=200,
    temperature=1.0,
    batch_size=8,
    replay_buffer_size=10000,
    updates_per_episode=1,
    skip_fairness_test=False,
    skip_equilibrium_test=False,
    random_seed=None,
    pretrain_wealth_heuristic=True,  # Enable by default - improves mean performance dramatically
    load_existing_model=False,  # Default: fresh training runs to avoid confounding from previous weights
    gamma=0.9999,  # Discount factor for Bellman equation (stability vs long-term planning trade-off)
):
    """
    Train a poker agent using SARSA with curriculum learning.

    Defaults use best config from medium grid search (lr=0.0003, (128,128) network, batch_size=8).

    IMPROVED CURRICULUM (Jan 2, 2026): Much less RandomAgent exposure to prevent over-fitting

    Curriculum stages (default: 1500 episodes total):
    1. Learn basic exploitation (episodes 0-200) - REDUCED from 800
       - All opponents play completely random (RandomAgent)
       - Learn basic profitable patterns: fold trash, bet premium hands
       - Goal: Exploit pure randomness, establish baseline strategy
       - Learning rate: 0.0003 (initial learning rate from grid search)
       - Epsilon-greedy: Decays from ~100% to ~2% for exploration
       - SHORT phase to avoid learning "always bet $3 = profit"

    2. Learn hand selection (episodes 200-700) - INCREASED from 400
       - 70% SkillfulRandomAgent, 30% self-play
       - SkillfulRandomAgent protects premium hands (only folds AA/AK/pairs 5% of time)
       - Goal: Learn hand selection and positional play, not just "bet until they fold"
       - Learning rate: 0.00006 (0.2x initial, to preserve learned strategies)
       - Epsilon-greedy: DISABLED (0.0) - softmax-only exploration
       - Prevents epsilon from corrupting SARSA Q-targets during multi-agent learning

    3. Self-play with strong anchoring (episodes 700+)
       - 60% self-play, 40% SkillfulRandomAgent (never below 30% Skillful!)
       - Goal: Develop advanced strategies through self-play while preventing collapse
       - Learning rate: 0.00003 (0.1x initial, very conservative for stability)
       - Epsilon-greedy: DISABLED (0.0) - softmax-only exploration
       - HIGH anchor percentage prevents catastrophic forgetting and over-fitting
       - SkillfulRandomAgent provides explicit signal that premium hands have value

    Pre-training (pretrain_wealth_heuristic=True):
       - Initializes Q-function to approximate wealth before RL training
       - Dramatically improves mean performance (47.9% → 84.3% in variance study)
       - Does NOT reduce variance but provides better starting point

    Discount Factor (gamma):
       - gamma=0.99: Light discounting for stability while preserving long-term planning
       - Prevents Q-value divergence due to unbounded Bellman iteration
       - Justification for high gamma (close to 1.0):
         * Episodes have finite duration (someone goes broke quickly)
         * No "infinite horizon" - terminal states bound Q-values naturally
         * Poker decisions happen "fast" - future rewards are immediately relevant
       - Lower gamma (0.9-0.95) would be more stable but sacrifice strategic depth

    Experience Replay (batch_size > 1):
       - batch_size=1: Online learning (immediate updates after each transition)
       - batch_size>1: Experience replay (store transitions, sample mini-batches)
       - replay_buffer_size: Max transitions to store (default: 10000)
       - updates_per_episode: Number of batch updates after each episode (default: 1)
       - Best from grid search: batch_size=8
    """

    # Set random seed for reproducibility if provided
    if random_seed is not None:
        print(f"\n🎲 Setting random seed to {random_seed} for reproducibility")
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        random.seed(random_seed)

    # PRE-TRAINING TEST: Verify game logic is fair before training (optional - skip during grid search)
    if not skip_fairness_test:
        print("\nRunning pre-training fairness test...")
        game_is_fair = test_game_fairness_with_random_agents(
            n_players=n_players,
            n_episodes=1000,  # Fast test: 1000 episodes
            max_deals=max_deals
        )

        if not game_is_fair:
            print("\n⚠️  Game fairness test failed! Fix game logic before training.")
            print("Aborting training.\n")
            return
    else:
        print("\n(Skipping fairness test - assuming game logic is correct)")


    # Create learning agent (player 0) and opponent agents
    learning_agent = Agent(
        player_index=0,
        n_players=n_players,
        learning_rate=learning_rate,
        hidden_layers=hidden_layers,
        temperature=temperature
    )

    # Create opponent agents (reuse these across episodes to avoid recreating models)
    opponent_agents = [
        Agent(
            player_index=i,
            n_players=n_players,
            learning_rate=learning_rate,
            hidden_layers=hidden_layers,
            temperature=temperature
        )
        for i in range(1, n_players)
    ]

    # Load existing model only if explicitly requested
    # Default is fresh training to avoid confounding from previous runs
    # (e.g., inheriting pathological Q-values from over-fitted models)
    model_loaded = False
    if load_existing_model:
        model_loaded = learning_agent.load_model(model_path)
    else:
        print(f"Skipping model load (load_existing_model=False) - starting fresh training")

    # Pre-train with wealth heuristic if enabled and no model was loaded
    if pretrain_wealth_heuristic and not model_loaded:
        learning_agent.pretrain_on_wealth_heuristic(
            n_samples=1000,
            n_epochs=10,
            batch_size=32,
            verbose=1
        )

    # Track cumulative statistics
    all_stats = {
        'deals_per_episode': [],
        'actions_per_episode': [],
        'max_deals_hits': 0,
        'positive_bet_percentages': [],  # All players
        'fold_percentages': [],  # All players
        'learning_agent_positive_bet_percentages': [],  # Learning agent only
        'learning_agent_fold_percentages': [],  # Learning agent only
        'action_types': {'fold': 0, 'check': 0, 'call': 0, 'raise': 0},  # Total counts across all episodes
        'survival_count': 0,  # Episodes where wealth > 0 at end
        'win_count': 0,  # Episodes where agent had highest wealth
        'elimination_count': 0,  # Episodes where wealth went to 0
        'premium_hands_seen': 0,  # Total premium hands dealt
        'premium_hands_folded': 0,  # Total premium hands folded
        'total_showdowns': 0,  # Total deals ending in showdown (2+ players)
        'total_wins_by_fold': 0,  # Total deals ending with only 1 player remaining
        # Track metrics by curriculum phase
        'by_phase': {
            'diverse': {'episodes': 0, 'survivals': 0, 'wins': 0, 'eliminations': 0,
                       'premium_seen': 0, 'premium_folded': 0},
            'mixed': {'episodes': 0, 'survivals': 0, 'wins': 0, 'eliminations': 0,
                     'premium_seen': 0, 'premium_folded': 0},
            'self-play': {'episodes': 0, 'survivals': 0, 'wins': 0, 'eliminations': 0,
                         'premium_seen': 0, 'premium_folded': 0},
        }
    }

    print(f"\nStarting SARSA training with {n_episodes} episodes")
    print(f"Max deals per episode: {max_deals}")
    print(f"Saving models to: {model_path}")
    print(f"\nConsistentRandomAgent Curriculum (prevents cumulative fold exploitation):")
    print(f"  Episodes 0-{curriculum_random_episodes}: 50% RandomAgent, 40% ConsistentRandomAgent, 10% SkillfulRandomAgent (LR={learning_rate}, epsilon decays)")
    print(f"  Episodes {curriculum_random_episodes}-{curriculum_random_episodes + curriculum_mixed_episodes}: 60% self-play, 30% ConsistentRandomAgent, 10% SkillfulRandomAgent (LR={learning_rate * 0.2}, epsilon=0)")
    if n_episodes > curriculum_random_episodes + curriculum_mixed_episodes:
        print(f"  Episodes {curriculum_random_episodes + curriculum_mixed_episodes}-{n_episodes}: 70% self-play, 30% ConsistentRandomAgent (no Random/Skillful) (LR={learning_rate * 0.1}, epsilon=0)")
    else:
        print(f"  (No phase 3 - only {n_episodes} episodes total)")
    print("=" * 70)
    print(f"\nDESCRIPTION: {RUN_DESCRIPTION}")
    print("=" * 70)

    # Create experience replay buffer (used only if batch_size > 1)
    replay_buffer = None
    if batch_size > 1:
        replay_buffer = ReplayBuffer(capacity=replay_buffer_size)
        print(f"\nExperience Replay: ENABLED")
        print(f"  Batch size: {batch_size}")
        print(f"  Buffer capacity: {replay_buffer_size}")
        print(f"  Updates per episode: {updates_per_episode}")
    else:
        print(f"\nExperience Replay: DISABLED (batch_size=1, online learning)")
    print("=" * 70)

    for episode in range(n_episodes):

        # Curriculum learning: determine opponent types based on episode number
        # Phase 1: Diverse opponent mix (episodes 0-200)
        # 50% RandomAgent, 40% ConsistentRandomAgent, 10% SkillfulRandomAgent
        if episode < curriculum_random_episodes:
            if episode == 0:
                learning_agent.set_learning_rate(learning_rate)
                print(f"\n[Episode {episode}] Phase 1: Diverse opponent mix (50% Random, 40% Consistent, 10% Skillful)")

            opponents = []
            for i in range(1, n_players):
                rand_val = np.random.rand()
                if rand_val < 0.50:
                    # 50% RandomAgent
                    opponents.append(RandomAgent(player_index=i, actions=learning_agent.actions))
                elif rand_val < 0.90:
                    # 40% ConsistentRandomAgent (prevents cumulative fold exploitation)
                    opponents.append(ConsistentRandomAgent(player_index=i, actions=learning_agent.actions))
                else:
                    # 10% SkillfulRandomAgent (minimal exposure to avoid selection bias)
                    opponents.append(SkillfulRandomAgent(player_index=i, actions=learning_agent.actions))
            players = [learning_agent] + opponents
            stage = "diverse"

        elif episode < curriculum_random_episodes + curriculum_mixed_episodes:
            # Phase 2: Transition to self-play with Consistent anchoring (LR = 0.2x initial)
            # 60% self-play, 30% ConsistentRandomAgent, 10% SkillfulRandomAgent
            if episode == curriculum_random_episodes:
                phase2_lr = learning_rate * 0.2
                learning_agent.set_learning_rate(phase2_lr)
                print(f"\n[Episode {episode}] Phase 2: Self-play transition (60% self-play, 30% Consistent, 10% Skillful) - reducing learning rate to {phase2_lr}, epsilon to 0.0")

            opponents = []
            for i in range(1, n_players):
                rand_val = np.random.rand()
                if rand_val < 0.60:
                    # 60% self-play
                    opponent_agents[i-1].model.set_weights(learning_agent.model.get_weights())
                    opponents.append(opponent_agents[i-1])
                elif rand_val < 0.90:
                    # 30% ConsistentRandomAgent (maintains diverse training signal)
                    opponents.append(ConsistentRandomAgent(player_index=i, actions=learning_agent.actions))
                else:
                    # 10% SkillfulRandomAgent (minimal)
                    opponents.append(SkillfulRandomAgent(player_index=i, actions=learning_agent.actions))
            players = [learning_agent] + opponents
            stage = "mixed"

        else:
            # Phase 3: Heavy self-play with Consistent anchoring (LR = 0.1x initial)
            # 70% self-play, 30% ConsistentRandomAgent (no more Random/Skillful)
            if episode == curriculum_random_episodes + curriculum_mixed_episodes:
                phase3_lr = learning_rate * 0.1
                learning_agent.set_learning_rate(phase3_lr)
                print(f"\n[Episode {episode}] Phase 3: Heavy self-play (70% self-play, 30% Consistent) - reducing learning rate to {phase3_lr}")

            opponents = []
            for i in range(1, n_players):
                rand_val = np.random.rand()
                if rand_val < 0.70:
                    # 70% self-play
                    opponent_agents[i-1].model.set_weights(learning_agent.model.get_weights())
                    opponents.append(opponent_agents[i-1])
                else:
                    # 30% ConsistentRandomAgent (prevents catastrophic forgetting, no exploitation learning)
                    opponents.append(ConsistentRandomAgent(player_index=i, actions=learning_agent.actions))
            players = [learning_agent] + opponents
            stage = "self-play"

        episode_stats = run_one_episode(episode, players, max_deals=max_deals,
                                         curriculum_random_episodes=curriculum_random_episodes,
                                         curriculum_mixed_episodes=curriculum_mixed_episodes,
                                         epsilon_decay_rate=epsilon_decay_rate,
                                         replay_buffer=replay_buffer,
                                         gamma=gamma)

        # Track statistics
        all_stats['deals_per_episode'].append(episode_stats['deals'])
        all_stats['actions_per_episode'].append(len(episode_stats['actions']))
        if episode_stats['hit_max_deals']:
            all_stats['max_deals_hits'] += 1

        # Track outcomes (overall and by phase)
        all_stats['by_phase'][stage]['episodes'] += 1

        if episode_stats.get('learning_agent_survived', False):
            all_stats['survival_count'] += 1
            all_stats['by_phase'][stage]['survivals'] += 1

        if episode_stats.get('learning_agent_won', False):
            all_stats['win_count'] += 1
            all_stats['by_phase'][stage]['wins'] += 1

        if episode_stats.get('learning_agent_eliminated', False):
            all_stats['elimination_count'] += 1
            all_stats['by_phase'][stage]['eliminations'] += 1

        # Track premium hand stats
        all_stats['premium_hands_seen'] += episode_stats.get('premium_hands_seen', 0)
        all_stats['premium_hands_folded'] += episode_stats.get('premium_hands_folded', 0)
        all_stats['by_phase'][stage]['premium_seen'] += episode_stats.get('premium_hands_seen', 0)
        all_stats['by_phase'][stage]['premium_folded'] += episode_stats.get('premium_hands_folded', 0)

        # Track showdown frequency
        all_stats['total_showdowns'] += episode_stats.get('showdowns', 0)
        all_stats['total_wins_by_fold'] += episode_stats.get('wins_by_fold', 0)

        # Aggregate action type counts
        for action_type in ['fold', 'check', 'call', 'raise']:
            all_stats['action_types'][action_type] += episode_stats['action_types'][action_type]

        # Calculate action percentages for ALL players
        actions = episode_stats['actions']
        if actions:
            positive_bets = sum(1 for a in actions if a > 0)
            folds = sum(1 for a in actions if a < 0)
            pct_positive = 100 * positive_bets / len(actions)
            pct_fold = 100 * folds / len(actions)
        else:
            pct_positive = 0
            pct_fold = 0
        all_stats['positive_bet_percentages'].append(pct_positive)
        all_stats['fold_percentages'].append(pct_fold)

        # Calculate action percentages for LEARNING AGENT ONLY
        agent_actions = episode_stats['learning_agent_actions']
        if agent_actions:
            agent_positive_bets = sum(1 for a in agent_actions if a > 0)
            agent_folds = sum(1 for a in agent_actions if a < 0)
            agent_pct_positive = 100 * agent_positive_bets / len(agent_actions)
            agent_pct_fold = 100 * agent_folds / len(agent_actions)
        else:
            agent_pct_positive = 0
            agent_pct_fold = 0
        all_stats['learning_agent_positive_bet_percentages'].append(agent_pct_positive)
        all_stats['learning_agent_fold_percentages'].append(agent_pct_fold)

        # Experience replay: Sample and train on batches (only if batch_size > 1)
        if replay_buffer is not None and len(replay_buffer) >= batch_size:
            for _ in range(updates_per_episode):
                # Sample mini-batch from replay buffer
                states, actions, rewards, next_states = replay_buffer.sample(batch_size)

                # Compute SARSA targets for each transition in batch
                target_q_values = np.zeros(batch_size)
                for i in range(batch_size):
                    # Get next action's Q-value (SARSA uses the actual next action taken)
                    # Note: In replay buffer, we need to recompute next_action using current policy
                    # For now, use a simplified approach: estimate max Q for next state
                    model_input = learning_agent.get_model_input(next_states[i], learning_agent.actions)
                    next_q_values = learning_agent.model.predict(model_input, verbose=0)[:, 0]

                    # SARSA target: reward + gamma * Q(next_state, next_action)
                    # Use max for simplicity (this is closer to Q-learning)
                    # TODO: Store next_action in buffer for true SARSA
                    max_next_q = np.max(next_q_values)
                    target_q_values[i] = rewards[i] + gamma * max_next_q

                # Update Q-function with mini-batch
                learning_agent.update_q_batch(states, actions, target_q_values)

        # Print progress every 10 episodes with diagnostics
        if episode % 10 == 0 and episode > 0:
            recent_deals = all_stats['deals_per_episode'][-10:] if episode > 0 else [episode_stats['deals']]
            recent_agent_fold_pct = all_stats['learning_agent_fold_percentages'][-10:] if episode > 0 else [agent_pct_fold]
            recent_agent_bet_pct = all_stats['learning_agent_positive_bet_percentages'][-10:] if episode > 0 else [agent_pct_positive]
            avg_deals = np.mean(recent_deals)
            avg_agent_fold_pct = np.mean(recent_agent_fold_pct)
            avg_agent_bet_pct = np.mean(recent_agent_bet_pct)
            win_rate = 100 * all_stats['win_count'] / (episode + 1)

            # Diagnostic: probability of betting with pocket aces
            prob_bet_aces = get_pocket_aces_bet_probability(learning_agent)

            # NEW: Monitor Q-value magnitudes to catch pathological over-fitting early
            # Sample Q-values from random state to check for explosion
            test_state = State(n_players=n_players, initial_wealth=TYPICAL_INITIAL_WEALTH)
            test_private_state = learning_agent.get_private_state(test_state)
            test_input = learning_agent.get_model_input(test_private_state, learning_agent.actions)
            test_q_values = learning_agent.model.predict(test_input, verbose=0)[:, 0]
            max_q = np.max(np.abs(test_q_values))

            # Warning: if max |Q| > 10k, model is likely over-fitting
            q_warning = " ⚠️ Q-EXPLOSION!" if max_q > 10000 else ""

            print(f"Episode {episode:4d} ({stage:10s}) | Wins: {win_rate:5.1f}% | Deals: {avg_deals:5.1f} | Agent Fold: {avg_agent_fold_pct:5.1f}% | Agent Bet>0: {avg_agent_bet_pct:5.1f}% | P(bet|AA): {100*prob_bet_aces:5.1f}% | MaxQ: {max_q:7.0f}{q_warning}")

        # Save model periodically
        if episode > 0 and episode % save_interval == 0:
            # Save latest
            learning_agent.save_model(model_path)

            # Save episode-numbered checkpoint in same directory as model_path
            # This allows grid_search to load and evaluate each checkpoint
            model_dir = os.path.dirname(model_path)
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            checkpoint_path = os.path.join(model_dir, f"{model_name}_ep{episode}.h5")
            learning_agent.save_model(checkpoint_path)

        # MID-TRAINING VALIDATION: Every 100 episodes, test frozen agent vs RandomAgent
        # This helps us track when/how the policy degrades over time
        if episode > 0 and episode % 100 == 0:
            print("\n" + "=" * 70)
            if episode == curriculum_random_episodes:
                print(f"MID-TRAINING CHECKPOINT: After Random Phase (Episode {episode})")
            else:
                print(f"MID-TRAINING CHECKPOINT: Episode {episode}")
            print("=" * 70)
            print("Testing frozen agent vs RandomAgent (no learning, no exploration)...")

            random_test_opponents = [
                RandomAgent(player_index=i, actions=learning_agent.actions)
                for i in range(1, n_players)
            ]
            results = evaluate_frozen_agent(
                learning_agent, random_test_opponents, n_episodes=300, max_deals=max_deals
            )

            print(f"\nFrozen evaluation results at episode {episode}:")
            print(f"  Win rate:        {100 * results['win_rate']:.1f}% ± {100 * results['ci_margins']['win']:.1f}% (95% CI)")
            print(f"  Survival rate:   {100 * results['survival_rate']:.1f}% ± {100 * results['ci_margins']['survival']:.1f}%")
            print(f"  Elimination rate: {100 * results['elimination_rate']:.1f}% ± {100 * results['ci_margins']['elimination']:.1f}%")
            print(f"  Games played: 300, Timeouts: {results['timeouts']}")
            print(f"  Expected: >33.3% win rate (beating random opponents)")

            if results['win_rate'] > 1.0 / n_players:
                print(f"  ✓ PASS: Agent beats random play")
            else:
                print(f"  ⚠️  WARNING: Agent does NOT beat random play yet")
                if episode == curriculum_random_episodes:
                    print(f"  Continuing to mixed phase anyway...")

            # Test vs ConsistentRandomAgent
            print(f"\nTesting frozen agent vs ConsistentRandomAgent (no learning, no exploration)...")
            consistent_test_opponents = [
                ConsistentRandomAgent(player_index=i, actions=learning_agent.actions)
                for i in range(1, n_players)
            ]
            results_consistent = evaluate_frozen_agent(
                learning_agent, consistent_test_opponents, n_episodes=300, max_deals=max_deals
            )

            print(f"\nFrozen evaluation vs ConsistentRandomAgent at episode {episode}:")
            print(f"  Win rate:        {100 * results_consistent['win_rate']:.1f}% ± {100 * results_consistent['ci_margins']['win']:.1f}% (95% CI)")
            print(f"  Survival rate:   {100 * results_consistent['survival_rate']:.1f}% ± {100 * results_consistent['ci_margins']['survival']:.1f}%")
            print(f"  Elimination rate: {100 * results_consistent['elimination_rate']:.1f}% ± {100 * results_consistent['ci_margins']['elimination']:.1f}%")
            print(f"  Games played: 300, Timeouts: {results_consistent['timeouts']}")
            print(f"  Expected: >33.3% win rate")

            if results_consistent['win_rate'] > 1.0 / n_players:
                print(f"  ✓ PASS: Agent beats ConsistentRandomAgent")
            else:
                print(f"  ❌ FAIL: Agent does NOT beat ConsistentRandomAgent")

            # Run sanity checks on Q-function
            print(f"\nSanity check at episode {episode}:")
            learning_agent.sanity_check_learned_strategy()
            print("=" * 70 + "\n")

    # Save final model
    print("\n" + "=" * 70)
    print("Training complete! Saving final model...")
    learning_agent.save_model(model_path)

    # Save final timestamped checkpoint
    timestamp = datetime.now().strftime("%Y-%m-%d")
    final_checkpoint_path = f"models/player_0_{timestamp}_ep{n_episodes}_final.h5"
    learning_agent.save_model(final_checkpoint_path)

    # Print final statistics
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Total episodes: {n_episodes}")
    print(f"\nOverall performance (all {n_episodes} episodes):")
    print(f"  Win rate:        {100 * all_stats['win_count'] / n_episodes:.1f}% (had highest wealth)")
    print(f"  Survival rate:   {100 * all_stats['survival_count'] / n_episodes:.1f}% (wealth > 0 at end)")
    print(f"  Elimination rate: {100 * all_stats['elimination_count'] / n_episodes:.1f}% (wealth went to 0)")
    print(f"  Fair share (3 players): 33.3%")
    print(f"\nTimes hit max deals limit: {all_stats['max_deals_hits']} ({100*all_stats['max_deals_hits']/n_episodes:.1f}%)")
    print(f"Average deals per episode: {np.mean(all_stats['deals_per_episode']):.1f}")
    print(f"Median deals per episode: {np.median(all_stats['deals_per_episode']):.1f}")
    print(f"Max deals in any episode: {np.max(all_stats['deals_per_episode'])}")

    # Showdown frequency tracking
    total_deals = all_stats['total_showdowns'] + all_stats['total_wins_by_fold']
    if total_deals > 0:
        showdown_pct = 100 * all_stats['total_showdowns'] / total_deals
        print(f"\nShowdown frequency:")
        print(f"  Total deals: {total_deals}")
        print(f"  Showdowns (2+ players): {all_stats['total_showdowns']} ({showdown_pct:.1f}%)")
        print(f"  Wins by fold (1 player): {all_stats['total_wins_by_fold']} ({100 - showdown_pct:.1f}%)")
        if showdown_pct < 20:
            print(f"  ⚠️  WARNING: Low showdown rate - agent may not be learning hand strength properly")
        elif showdown_pct < 40:
            print(f"  ✓ Moderate showdown rate - some hand evaluation learning")
        else:
            print(f"  ✓ Good showdown rate - agent sees many hand comparisons")

    print(f"\n[All Players]")
    print(f"  Average % positive bets: {np.mean(all_stats['positive_bet_percentages']):.1f}%")
    print(f"  Average % folds: {np.mean(all_stats['fold_percentages']):.1f}%")

    print(f"\n[Learning Agent Only]")
    print(f"  Average % positive bets: {np.mean(all_stats['learning_agent_positive_bet_percentages']):.1f}%")
    print(f"  Average % folds: {np.mean(all_stats['learning_agent_fold_percentages']):.1f}%")

    # Detailed action breakdown
    total_actions = sum(all_stats['action_types'].values())
    if total_actions > 0:
        print(f"\nDetailed action breakdown (total actions: {total_actions}):")
        print(f"  Fold:  {all_stats['action_types']['fold']:6d} ({100 * all_stats['action_types']['fold'] / total_actions:5.1f}%)")
        print(f"  Check: {all_stats['action_types']['check']:6d} ({100 * all_stats['action_types']['check'] / total_actions:5.1f}%)")
        print(f"  Call:  {all_stats['action_types']['call']:6d} ({100 * all_stats['action_types']['call'] / total_actions:5.1f}%)")
        print(f"  Raise: {all_stats['action_types']['raise']:6d} ({100 * all_stats['action_types']['raise'] / total_actions:5.1f}%)")

    # Premium hand fold rate summary
    if all_stats['premium_hands_seen'] > 0:
        overall_fold_rate = 100 * all_stats['premium_hands_folded'] / all_stats['premium_hands_seen']
        print(f"\nPremium hand behavior (AA, KK, QQ, JJ, AK, AQ, AJ, KQ, KJ, QJ):")
        print(f"  Premium hands dealt: {all_stats['premium_hands_seen']}")
        print(f"  Premium hands folded: {all_stats['premium_hands_folded']} ({overall_fold_rate:.1f}%)")
        print(f"  Expected: Agent should rarely fold premium hands (<10%)")
        if overall_fold_rate > 50:
            print(f"  ❌ WARNING: Agent folds premium hands >50% of the time!")
        elif overall_fold_rate > 25:
            print(f"  ⚠️  CAUTION: Agent folds premium hands >{overall_fold_rate:.0f}% of the time")
        else:
            print(f"  ✓ Agent plays premium hands aggressively")

    # Breakdown by curriculum stage
    print(f"\nPerformance by curriculum phase:")
    for phase_name in ['diverse', 'mixed', 'self-play']:
        phase_data = all_stats['by_phase'][phase_name]
        n_eps = phase_data['episodes']
        if n_eps > 0:
            win_rate = 100 * phase_data['wins'] / n_eps
            survival_rate = 100 * phase_data['survivals'] / n_eps
            elim_rate = 100 * phase_data['eliminations'] / n_eps
            print(f"\n  {phase_name.capitalize()} phase ({n_eps} episodes):")
            print(f"    Win rate:        {win_rate:5.1f}% (had highest wealth)")
            print(f"    Survival rate:   {survival_rate:5.1f}% (wealth > 0 at end)")
            print(f"    Elimination rate: {elim_rate:5.1f}% (wealth went to 0)")

            # Report premium hand fold rate for this phase
            if phase_data['premium_seen'] > 0:
                phase_fold_rate = 100 * phase_data['premium_folded'] / phase_data['premium_seen']
                print(f"    Premium hands:   {phase_data['premium_seen']} dealt, {phase_data['premium_folded']} folded ({phase_fold_rate:.1f}%)")

            if phase_name == 'random':
                print(f"    Expected vs random: >33.3% win rate")
                if win_rate > 33.3:
                    print(f"    ✓ Agent learned to beat random play")
                else:
                    print(f"    ❌ Agent did NOT beat random play during training")

    # Sanity check the learned strategy
    learning_agent.sanity_check_learned_strategy()

    learning_agent.describe_learned_q_function()

    # POST-TRAINING VALIDATION
    print("\n" + "=" * 70)
    print("POST-TRAINING VALIDATION")
    print("=" * 70)

    # Validation 1: Frozen agent vs RandomAgent
    print(f"\n[Validation 1] Testing frozen agent vs {n_players-1} RandomAgent opponent(s) ({n_players}-player game)...")
    print(f"  Running 1000 episodes with no learning and no exploration...")
    random_opponents = [
        RandomAgent(player_index=i, actions=learning_agent.actions)
        for i in range(1, n_players)
    ]
    results = evaluate_frozen_agent(
        learning_agent, random_opponents, n_episodes=1000, max_deals=max_deals
    )
    print(f"\n  Results:")
    print(f"    Win rate:        {100 * results['win_rate']:.1f}% ± {100 * results['ci_margins']['win']:.1f}% (95% CI)")
    print(f"    Survival rate:   {100 * results['survival_rate']:.1f}% ± {100 * results['ci_margins']['survival']:.1f}%")
    print(f"    Elimination rate: {100 * results['elimination_rate']:.1f}% ± {100 * results['ci_margins']['elimination']:.1f}%")
    print(f"    Games: {results['wins']} wins, {results['survivals']} survivals, {results['eliminations']} eliminations (out of 1000)")
    if results['timeouts'] > 0:
        print(f"    Timeouts: {results['timeouts']}")
    fair_share_random = 1.0 / n_players
    print(f"    Expected: >{100*fair_share_random:.1f}% (beating random opponents in {n_players}-player game)")
    if results['win_rate'] > fair_share_random:
        print(f"    ✓ PASS: Agent beats random play")
    else:
        print(f"    ❌ FAIL: Agent does not consistently beat random play")

    # Validation 2: Frozen agent vs SkillfulRandomAgent (generalization test)
    print(f"\n[Validation 2] Testing frozen agent vs {n_players-1} SkillfulRandomAgent opponent(s) ({n_players}-player game)...")
    print(f"  Running 1000 episodes with no learning and no exploration...")
    print(f"  SkillfulRandomAgent protects premium hands (only folds AA/AK/pairs 5% of time)")
    skillful_opponents = [
        SkillfulRandomAgent(player_index=i, actions=learning_agent.actions)
        for i in range(1, n_players)
    ]
    results_skillful = evaluate_frozen_agent(
        learning_agent, skillful_opponents, n_episodes=1000, max_deals=max_deals
    )
    print(f"\n  Results:")
    print(f"    Win rate:        {100 * results_skillful['win_rate']:.1f}% ± {100 * results_skillful['ci_margins']['win']:.1f}% (95% CI)")
    print(f"    Survival rate:   {100 * results_skillful['survival_rate']:.1f}% ± {100 * results_skillful['ci_margins']['survival']:.1f}%")
    print(f"    Elimination rate: {100 * results_skillful['elimination_rate']:.1f}% ± {100 * results_skillful['ci_margins']['elimination']:.1f}%")
    print(f"    Games: {results_skillful['wins']} wins, {results_skillful['survivals']} survivals, {results_skillful['eliminations']} eliminations (out of 1000)")
    if results_skillful['timeouts'] > 0:
        print(f"    Timeouts: {results_skillful['timeouts']}")

    # Compare to RandomAgent performance
    drop_random_to_skillful = (results['win_rate'] - results_skillful['win_rate']) * 100
    print(f"\n  Generalization analysis:")
    print(f"    RandomAgent win rate:      {100 * results['win_rate']:.1f}%")
    print(f"    SkillfulRandomAgent win rate: {100 * results_skillful['win_rate']:.1f}%")
    print(f"    Performance drop:          {drop_random_to_skillful:.1f} pp")

    if results_skillful['win_rate'] > fair_share_random:
        if drop_random_to_skillful < 20:
            print(f"    ✓ EXCELLENT: Agent beats SkillfulRandomAgent with good generalization (drop < 20pp)")
        elif drop_random_to_skillful < 30:
            print(f"    ✓ GOOD: Agent beats SkillfulRandomAgent (drop < 30pp)")
        else:
            print(f"    ⚠️  WARNING: Large performance drop (>30pp) suggests over-fitting to random folding")
    else:
        print(f"    ❌ FAIL: Agent does not beat SkillfulRandomAgent (learned exploitative, not generalizable strategy)")

    print(f"\n[Validation 3] Testing frozen agent vs {n_players-1} ConsistentRandomAgent opponent(s) ({n_players}-player game)...")
    print(f"  Running 1000 episodes with no learning and no exploration...")
    print(f"  ConsistentRandomAgent commits to strategy per deal (prevents cumulative fold exploitation)")
    consistent_opponents = [
        ConsistentRandomAgent(player_index=i, actions=learning_agent.actions)
        for i in range(1, n_players)
    ]
    results_consistent = evaluate_frozen_agent(
        learning_agent, consistent_opponents, n_episodes=1000, max_deals=max_deals
    )
    print(f"\n  Results:")
    print(f"    Win rate:        {100 * results_consistent['win_rate']:.1f}% ± {100 * results_consistent['ci_margins']['win']:.1f}% (95% CI)")
    print(f"    Survival rate:   {100 * results_consistent['survival_rate']:.1f}% ± {100 * results_consistent['ci_margins']['survival']:.1f}%")
    print(f"    Elimination rate: {100 * results_consistent['elimination_rate']:.1f}% ± {100 * results_consistent['ci_margins']['elimination']:.1f}%")
    print(f"    Games: {results_consistent['wins']} wins, {results_consistent['survivals']} survivals, {results_consistent['eliminations']} eliminations (out of 1000)")
    if results_consistent['timeouts'] > 0:
        print(f"    Timeouts: {results_consistent['timeouts']}")

    if results_consistent['win_rate'] > fair_share_random:
        print(f"    ✓ PASS: Agent beats ConsistentRandomAgent ({100 * results_consistent['win_rate']:.1f}% > {100*fair_share_random:.1f}%)")
    else:
        print(f"    ❌ FAIL: Agent does NOT beat ConsistentRandomAgent (should be >{100*fair_share_random:.1f}%)")

    # Validation 4: Frozen self-play equilibrium test (optional - skip during grid search)
    if not skip_equilibrium_test:
        print(f"\n[Validation 4] Frozen self-play equilibrium test...")
        print(f"  Testing for positional bias and Nash convergence...")
        print(f"  All {n_players} players use identical frozen policy (learning agent's weights)")
        print(f"  Running 1000 episodes...")

        # Create clones of the learning agent for other positions
        # Must use same architecture (hidden_layers) to match weight shapes
        frozen_clones = []
        for i in range(1, n_players):
            clone = Agent(
                player_index=i,
                n_players=n_players,
                temperature=learning_agent.temperature,
                learning_rate=learning_agent.learning_rate,
                hidden_layers=learning_agent.hidden_layers
            )
            clone.model.set_weights(learning_agent.model.get_weights())
            frozen_clones.append(clone)

        results = evaluate_frozen_agent(
            learning_agent, frozen_clones, n_episodes=1000, max_deals=max_deals, progress_interval=100
        )

        fair_share = 1.0 / n_players
        print(f"\n  Results:")
        print(f"    Player 0 win rate: {100 * results['win_rate']:.1f}% ± {100 * results['ci_margins']['win']:.1f}% (95% CI)")
        print(f"    Wins: {results['wins']}/1000")
        if results['timeouts'] > 0:
            print(f"    Timeouts: {results['timeouts']}")
        print(f"    Expected (Nash equilibrium): {100 * fair_share:.1f}% ± {100 * results['ci_margins']['win']:.1f}%")

        # Check if deviation is statistically significant (more than 2 standard errors)
        deviation = abs(results['win_rate'] - fair_share)
        if deviation > 2 * results['ci_margins']['win']:
            print(f"    ⚠️  WARNING: Significant deviation from fair share!")
            print(f"    This may indicate:")
            print(f"      - Positional bias bug in game logic")
            print(f"      - Agent hasn't converged to Nash equilibrium")
            print(f"      - Training artifact from player 0 always being learning agent")
        else:
            print(f"    ✓ PASS: Win rate consistent with fair share (no obvious positional bias)")
    else:
        print(f"\n(Skipping equilibrium test - grid search mode)")

    print("\n" + "=" * 70)
    print(f"DESCRIPTION: {RUN_DESCRIPTION}")
    print("=" * 70)


def main(n_players=3):

    run_sarsa(n_players)


if __name__ == "__main__":
    main()
