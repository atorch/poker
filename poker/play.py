from datetime import datetime
import numpy as np

from poker.state import State
from poker.agent import Agent
from poker.random_agent import RandomAgent

# Description of this training run for logging/versioning
RUN_DESCRIPTION = "softmax for mixed strategies; added pot size and opponent fold status to state space"


def run_one_episode(episode, players, max_deals=50000, min_initial_wealth=15, max_initial_wealth=35):

    # This is (roughly) Sutton and Barto Figure 6.9
    # page 130, TODO compare to page 131
    # page 244

    # Note: the probability of random (exploratory) actions decreases over time
    proba_random_action = 0.02 + 0.98 * np.exp(-episode / 200)

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
        'learning_agent_won': False,
    }

    learning_player = state.current_player

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
        if action < 0:
            episode_stats['action_types']['fold'] += 1
        elif min_bet == 0 and action == 0:
            episode_stats['action_types']['check'] += 1
        elif action == min_bet:
            episode_stats['action_types']['call'] += 1
        else:  # action > min_bet
            episode_stats['action_types']['raise'] += 1

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

    # Track if learning agent won (has wealth > 0 at termination)
    if state.wealth[learning_player] > 0:
        episode_stats['learning_agent_won'] = True

    return episode_stats


def run_sarsa(
    n_players,
    n_episodes=400,
    model_path="models/player_0_latest.h5",
    save_interval=10,
    max_deals=25_000,
    curriculum_random_episodes=100,
    curriculum_mixed_episodes=100,
):
    """
    Train a poker agent using SARSA with curriculum learning.

    Curriculum stages:
    1. Beat random agents (episodes 0 to curriculum_random_episodes)
       - All opponents play completely random
       - Learn basic profitable patterns: fold trash, bet premium hands

    2. Mixed opponents (episodes curriculum_random_episodes to curriculum_random_episodes + curriculum_mixed_episodes)
       - 50% random opponents, 50% self-play
       - Prevents overfitting to random exploitation

    3. Self-play (remaining episodes)
       - All opponents use learned policy
       - Converge toward Nash equilibrium strategies
    """

    # Create learning agent (player 0) and opponent agents
    learning_agent = Agent(player_index=0, n_players=n_players)

    # Create opponent agents (reuse these across episodes to avoid recreating models)
    opponent_agents = [Agent(player_index=i, n_players=n_players) for i in range(1, n_players)]

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
        'win_count': 0,  # Track wins (wealth > 0 at terminal)
    }

    print(f"\nStarting SARSA training with {n_episodes} episodes")
    print(f"Max deals per episode: {max_deals}")
    print(f"Saving models to: {model_path}")
    print(f"\nCurriculum learning schedule:")
    print(f"  Episodes 0-{curriculum_random_episodes}: vs Random agents")
    print(f"  Episodes {curriculum_random_episodes}-{curriculum_random_episodes + curriculum_mixed_episodes}: vs Mixed (50% random, 50% self-play)")
    print(f"  Episodes {curriculum_random_episodes + curriculum_mixed_episodes}-{n_episodes}: vs Self-play")
    print("=" * 70)
    print(f"\nDESCRIPTION: {RUN_DESCRIPTION}")
    print("=" * 70)

    for episode in range(n_episodes):

        # Curriculum learning: determine opponent types based on episode number
        if episode < curriculum_random_episodes:
            # Phase 1: All random opponents
            players = [learning_agent] + [
                RandomAgent(player_index=i, actions=learning_agent.actions)
                for i in range(1, n_players)
            ]
            stage = "random"
        elif episode < curriculum_random_episodes + curriculum_mixed_episodes:
            # Phase 2: Mixed opponents (50% random, 50% self-play)
            opponents = []
            for i in range(1, n_players):
                if np.random.rand() < 0.5:
                    opponents.append(RandomAgent(player_index=i, actions=learning_agent.actions))
                else:
                    # Reuse pre-created opponent agent and copy weights
                    opponent_agents[i-1].model.set_weights(learning_agent.model.get_weights())
                    opponents.append(opponent_agents[i-1])
            players = [learning_agent] + opponents
            stage = "mixed"
        else:
            # Phase 3: Self-play - reuse opponent agents and copy weights
            for i in range(n_players - 1):
                opponent_agents[i].model.set_weights(learning_agent.model.get_weights())
            players = [learning_agent] + opponent_agents
            stage = "self-play"

        episode_stats = run_one_episode(episode, players, max_deals=max_deals)

        # Track statistics
        all_stats['deals_per_episode'].append(episode_stats['deals'])
        all_stats['actions_per_episode'].append(len(episode_stats['actions']))
        if episode_stats['hit_max_deals']:
            all_stats['max_deals_hits'] += 1

        # Track if learning agent won (survived)
        if episode_stats.get('learning_agent_won', False):
            all_stats['win_count'] += 1

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
        if episode % 10 == 0:
            recent_deals = all_stats['deals_per_episode'][-10:] if episode > 0 else [episode_stats['deals']]
            recent_positive_pct = all_stats['positive_bet_percentages'][-10:] if episode > 0 else [pct_positive]
            recent_fold_pct = all_stats['fold_percentages'][-10:] if episode > 0 else [pct_fold]
            avg_deals = np.mean(recent_deals)
            avg_positive_pct = np.mean(recent_positive_pct)
            avg_fold_pct = np.mean(recent_fold_pct)
            win_rate = 100 * all_stats['win_count'] / (episode + 1)

            print(f"Episode {episode:4d} ({stage:10s}) | Wins: {win_rate:5.1f}% | Deals: {avg_deals:5.1f} | Fold: {avg_fold_pct:5.1f}% | Bet>0: {avg_positive_pct:5.1f}%")

        # Save model periodically
        if episode > 0 and episode % save_interval == 0:
            # Save latest
            learning_agent.save_model(model_path)

            # Save timestamped checkpoint
            timestamp = datetime.now().strftime("%Y-%m-%d")
            checkpoint_path = f"models/player_0_{timestamp}_ep{episode}.h5"
            learning_agent.save_model(checkpoint_path)

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
    print(f"Overall win rate: {100 * all_stats['win_count'] / n_episodes:.1f}% (fair share: {100/n_players:.1f}%)")
    print(f"Times hit max deals limit: {all_stats['max_deals_hits']} ({100*all_stats['max_deals_hits']/n_episodes:.1f}%)")
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

    # Breakdown by curriculum stage
    if curriculum_random_episodes > 0:
        random_wins = sum(1 for i in range(min(curriculum_random_episodes, len(all_stats['deals_per_episode'])))
                         if i < len(all_stats['deals_per_episode']))  # Placeholder, will track properly
        print(f"\nCurriculum breakdown:")
        print(f"  Random phase (0-{curriculum_random_episodes}): episodes completed")
        print(f"  Mixed phase ({curriculum_random_episodes}-{curriculum_random_episodes + curriculum_mixed_episodes}): episodes completed")
        print(f"  Self-play phase ({curriculum_random_episodes + curriculum_mixed_episodes}+): episodes completed")

    # Sanity check the learned strategy
    learning_agent.sanity_check_learned_strategy()

    learning_agent.describe_learned_q_function()

    print("\n" + "=" * 70)
    print(f"DESCRIPTION: {RUN_DESCRIPTION}")
    print("=" * 70)


def main(n_players=3):

    run_sarsa(n_players)


if __name__ == "__main__":
    main()
