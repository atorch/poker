from datetime import datetime
import numpy as np
import math
import os

from poker.state import State, GameStage
from poker.agent import Agent
from poker.random_agent import RandomAgent
from poker.skillful_random_agent import SkillfulRandomAgent
from poker.config import MIN_INITIAL_WEALTH, MAX_INITIAL_WEALTH, TYPICAL_INITIAL_WEALTH
from poker.cards import Rank

# Description of this training run for logging/versioning
RUN_DESCRIPTION = "NO epsilon during self-play + smoother curriculum (100%→50%→70% self-play) + lower LR (0.001→0.0002→0.0001) + stronger anchoring (50%/30% random)"


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


def run_one_episode(episode, players, max_deals=50000, min_initial_wealth=MIN_INITIAL_WEALTH, max_initial_wealth=MAX_INITIAL_WEALTH, curriculum_random_episodes=100, curriculum_mixed_episodes=100, epsilon_decay_rate=200):

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
        'actions': [],  # List of all actions taken
        'action_types': {'fold': 0, 'check': 0, 'call': 0, 'raise': 0},  # Detailed action breakdown
        'hit_max_deals': False,
        'learning_agent_survived': False,  # wealth > 0 at end
        'learning_agent_won': False,  # had highest wealth at end
        'learning_agent_eliminated': False,  # wealth went to 0
        'premium_hands_seen': 0,  # Times agent was dealt premium hand
        'premium_hands_folded': 0,  # Times agent folded premium hand
        'deals_with_premium_checked': set(),  # Track which deals we've already counted premium hands for
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

        # Track deals and check max limit
        episode_stats['deals'] = state.n_deals
        if state.n_deals >= max_deals:
            episode_stats['hit_max_deals'] = True
            state.terminal = True
            break

        # Track action for diagnostics
        episode_stats['actions'].append(action)

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
            updated_guess_for_q = reward + continuation_value

        players[learning_player].update_q(private_state, action, updated_guess_for_q)

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
    actions = [-1, 0, 1, 2, 3]
    random_agents = [RandomAgent(player_index=i, actions=actions) for i in range(n_players)]

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
    n_episodes=800,
    model_path="models/player_0_latest.h5",
    save_interval=10,
    max_deals=5_000,
    curriculum_random_episodes=500,
    curriculum_mixed_episodes=100,
    learning_rate=0.0003,
    hidden_layers=(64, 64),
    epsilon_decay_rate=200,
    temperature=1.0,
):
    """
    Train a poker agent using SARSA with curriculum learning.

    Curriculum stages (default: 800 episodes total):
    1. Beat random agents (episodes 0-500)
       - All opponents play completely random
       - Learn basic profitable patterns: fold trash, bet premium hands
       - Learning rate: 0.0003 (initial learning rate from grid search)
       - Epsilon-greedy: Decays from ~100% to ~2% for exploration
       - Longer phase ensures solid foundation before self-play

    2. Mixed opponents (episodes 500-600)
       - 50% self-play, 50% RandomAgent/SkillfulRandomAgent (smoother transition)
       - Learning rate: 0.00006 (0.2x initial, to preserve learned strategies)
       - Epsilon-greedy: DISABLED (0.0) - softmax-only exploration
       - Prevents epsilon from corrupting SARSA Q-targets during multi-agent learning

    3. Self-play with anchoring (episodes 600+)
       - 70% self-play, 30% RandomAgent/SkillfulRandomAgent (never 100%!)
       - Learning rate: 0.00003 (0.1x initial, very conservative for stability)
       - Epsilon-greedy: DISABLED (0.0) - softmax-only exploration
       - High anchor percentage prevents catastrophic forgetting
       - SkillfulRandomAgent provides explicit signal that premium hands have value
    """

    # PRE-TRAINING TEST: Verify game logic is fair before training
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

    # Try to load existing model for the learning player
    learning_agent.load_model(model_path)

    # Track cumulative statistics
    all_stats = {
        'deals_per_episode': [],
        'actions_per_episode': [],
        'max_deals_hits': 0,
        'positive_bet_percentages': [],
        'fold_percentages': [],
        'action_types': {'fold': 0, 'check': 0, 'call': 0, 'raise': 0},  # Total counts across all episodes
        'survival_count': 0,  # Episodes where wealth > 0 at end
        'win_count': 0,  # Episodes where agent had highest wealth
        'elimination_count': 0,  # Episodes where wealth went to 0
        'premium_hands_seen': 0,  # Total premium hands dealt
        'premium_hands_folded': 0,  # Total premium hands folded
        # Track metrics by curriculum phase
        'by_phase': {
            'random': {'episodes': 0, 'survivals': 0, 'wins': 0, 'eliminations': 0,
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
    print(f"\nCurriculum learning schedule:")
    print(f"  Episodes 0-{curriculum_random_episodes}: 100% RandomAgent (LR={learning_rate}, epsilon decays)")
    print(f"  Episodes {curriculum_random_episodes}-{curriculum_random_episodes + curriculum_mixed_episodes}: 50% self-play, 50% Random/Skillful (LR={learning_rate * 0.2}, epsilon=0)")
    if n_episodes > curriculum_random_episodes + curriculum_mixed_episodes:
        print(f"  Episodes {curriculum_random_episodes + curriculum_mixed_episodes}-{n_episodes}: 70% self-play, 30% Random/Skillful (LR={learning_rate * 0.1}, epsilon=0)")
    else:
        print(f"  (No phase 3 - only {n_episodes} episodes total)")
    print("=" * 70)
    print(f"\nDESCRIPTION: {RUN_DESCRIPTION}")
    print("=" * 70)

    for episode in range(n_episodes):

        # Curriculum learning: determine opponent types based on episode number
        if episode < curriculum_random_episodes:
            # Phase 1: All random opponents (use initial learning rate)
            if episode == 0:
                learning_agent.set_learning_rate(learning_rate)
            players = [learning_agent] + [
                RandomAgent(player_index=i, actions=learning_agent.actions)
                for i in range(1, n_players)
            ]
            stage = "random"
        elif episode < curriculum_random_episodes + curriculum_mixed_episodes:
            # Phase 2: Mixed opponents - 50% self-play, 50% random/skillful (LR = 0.2x initial)
            # Smoother transition: More random agents to prevent sudden shift
            if episode == curriculum_random_episodes:
                phase2_lr = learning_rate * 0.2
                learning_agent.set_learning_rate(phase2_lr)
                print(f"\n[Episode {episode}] Entering mixed phase - reducing learning rate to {phase2_lr}, epsilon to 0.0")

            opponents = []
            for i in range(1, n_players):
                rand_val = np.random.rand()
                if rand_val < 0.50:
                    # 50% self-play
                    opponent_agents[i-1].model.set_weights(learning_agent.model.get_weights())
                    opponents.append(opponent_agents[i-1])
                elif rand_val < 0.75:
                    # 25% RandomAgent
                    opponents.append(RandomAgent(player_index=i, actions=learning_agent.actions))
                else:
                    # 25% SkillfulRandomAgent
                    opponents.append(SkillfulRandomAgent(player_index=i, actions=learning_agent.actions))
            players = [learning_agent] + opponents
            stage = "mixed"
        else:
            # Phase 3: Self-play with anchoring - 70% self-play, 30% random/skillful (LR = 0.1x initial)
            # NEVER 100% self-play - prevents catastrophic forgetting!
            # Higher anchor percentage for smoother learning
            if episode == curriculum_random_episodes + curriculum_mixed_episodes:
                phase3_lr = learning_rate * 0.1
                learning_agent.set_learning_rate(phase3_lr)
                print(f"\n[Episode {episode}] Entering self-play phase - reducing learning rate to {phase3_lr}")

            opponents = []
            for i in range(1, n_players):
                rand_val = np.random.rand()
                if rand_val < 0.70:
                    # 70% self-play
                    opponent_agents[i-1].model.set_weights(learning_agent.model.get_weights())
                    opponents.append(opponent_agents[i-1])
                elif rand_val < 0.85:
                    # 15% RandomAgent
                    opponents.append(RandomAgent(player_index=i, actions=learning_agent.actions))
                else:
                    # 15% SkillfulRandomAgent
                    opponents.append(SkillfulRandomAgent(player_index=i, actions=learning_agent.actions))
            players = [learning_agent] + opponents
            stage = "self-play"

        episode_stats = run_one_episode(episode, players, max_deals=max_deals,
                                         curriculum_random_episodes=curriculum_random_episodes,
                                         curriculum_mixed_episodes=curriculum_mixed_episodes,
                                         epsilon_decay_rate=epsilon_decay_rate)

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

        # Aggregate action type counts
        for action_type in ['fold', 'check', 'call', 'raise']:
            all_stats['action_types'][action_type] += episode_stats['action_types'][action_type]

        # Calculate action percentages
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

        # Print progress every 10 episodes with diagnostics
        if episode % 10 == 0 and episode > 0:
            recent_deals = all_stats['deals_per_episode'][-10:] if episode > 0 else [episode_stats['deals']]
            recent_positive_pct = all_stats['positive_bet_percentages'][-10:] if episode > 0 else [pct_positive]
            recent_fold_pct = all_stats['fold_percentages'][-10:] if episode > 0 else [pct_fold]
            avg_deals = np.mean(recent_deals)
            avg_positive_pct = np.mean(recent_positive_pct)
            avg_fold_pct = np.mean(recent_fold_pct)
            win_rate = 100 * all_stats['win_count'] / (episode + 1)

            # Diagnostic: probability of betting with pocket aces
            prob_bet_aces = get_pocket_aces_bet_probability(learning_agent)

            print(f"Episode {episode:4d} ({stage:10s}) | Wins: {win_rate:5.1f}% | Deals: {avg_deals:5.1f} | Fold: {avg_fold_pct:5.1f}% | Bet>0: {avg_positive_pct:5.1f}% | P(bet|AA): {100*prob_bet_aces:5.1f}%")

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
    print(f"Average % positive bets: {np.mean(all_stats['positive_bet_percentages']):.1f}%")
    print(f"Average % folds: {np.mean(all_stats['fold_percentages']):.1f}%")

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
    for phase_name in ['random', 'mixed', 'self-play']:
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

    # Validation 2: Frozen self-play equilibrium test
    print(f"\n[Validation 2] Frozen self-play equilibrium test...")
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

    print("\n" + "=" * 70)
    print(f"DESCRIPTION: {RUN_DESCRIPTION}")
    print("=" * 70)


def main(n_players=3):

    run_sarsa(n_players)


if __name__ == "__main__":
    main()
