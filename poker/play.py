from datetime import datetime
import numpy as np

from poker.state import State
from poker.agent import Agent


def run_one_episode(episode, players, initial_wealth=100, max_deals=50000):

    # This is (roughly) Sutton and Barto Figure 6.9
    # page 130, TODO compare to page 131
    # page 244

    # Note: the probability of random (exploratory) actions decreases over time
    proba_random_action = 0.02 + 0.98 * np.exp(-episode / 200)

    state = State(n_players=len(players), initial_wealth=initial_wealth, verbose=False)

    # Track diagnostics
    episode_stats = {
        'deals': 0,
        'actions': [],  # List of all actions taken
        'hit_max_deals': False
    }

    learning_player = state.current_player

    private_state = players[learning_player].get_private_state(state)
    action = players[learning_player].get_action(state, proba_random_action)

    cumulative_reward = 0

    # Note: at the beginning of every episode, we push updates to the q function
    #  from the learning player to all other players
    for player_index in range(state.n_players):
        if player_index != learning_player:
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
            updated_guess_for_q = reward

        else:

            continuation_value = players[learning_player].predicted_q(
                next_private_state, next_action
            )
            updated_guess_for_q = reward + continuation_value

        players[learning_player].update_q(private_state, action, updated_guess_for_q)

        action = next_action
        private_state = next_private_state

    return episode_stats


def run_sarsa(n_players, n_episodes=200, model_path="models/player_0_latest.h5", save_interval=10, max_deals=25_000):

    players = [Agent(player_index) for player_index in range(n_players)]

    # Try to load existing model for the learning player (player 0)
    players[0].load_model(model_path)

    # Track cumulative statistics
    all_stats = {
        'deals_per_episode': [],
        'actions_per_episode': [],
        'max_deals_hits': 0,
        'positive_bet_percentages': []
    }

    print(f"\nStarting SARSA training with {n_episodes} episodes")
    print(f"Max deals per episode: {max_deals}")
    print(f"Saving models to: {model_path}")
    print("=" * 70)

    for episode in range(n_episodes):

        episode_stats = run_one_episode(episode, players, max_deals=max_deals)

        # Track statistics
        all_stats['deals_per_episode'].append(episode_stats['deals'])
        all_stats['actions_per_episode'].append(len(episode_stats['actions']))
        if episode_stats['hit_max_deals']:
            all_stats['max_deals_hits'] += 1

        # Calculate % of positive bets (>$0)
        actions = episode_stats['actions']
        positive_bets = sum(1 for a in actions if a > 0)
        pct_positive = 100 * positive_bets / len(actions) if actions else 0
        all_stats['positive_bet_percentages'].append(pct_positive)

        # Print progress every 10 episodes with diagnostics
        if episode % 10 == 0:
            recent_deals = all_stats['deals_per_episode'][-10:] if episode > 0 else [episode_stats['deals']]
            recent_positive_pct = all_stats['positive_bet_percentages'][-10:] if episode > 0 else [pct_positive]
            avg_deals = np.mean(recent_deals)
            avg_positive_pct = np.mean(recent_positive_pct)

            print(f"Episode {episode:4d} | Avg deals: {avg_deals:7.1f} | Positive bets: {avg_positive_pct:5.1f}% | Max deals hits: {all_stats['max_deals_hits']}")

        # Save model periodically
        if episode > 0 and episode % save_interval == 0:
            # Save latest
            players[0].save_model(model_path)

            # Save timestamped checkpoint
            timestamp = datetime.now().strftime("%Y-%m-%d")
            checkpoint_path = f"models/player_0_{timestamp}_ep{episode}.h5"
            players[0].save_model(checkpoint_path)

    # Save final model
    print("\n" + "=" * 70)
    print("Training complete! Saving final model...")
    players[0].save_model(model_path)

    # Save final timestamped checkpoint
    timestamp = datetime.now().strftime("%Y-%m-%d")
    final_checkpoint_path = f"models/player_0_{timestamp}_ep{n_episodes}_final.h5"
    players[0].save_model(final_checkpoint_path)

    # Print final statistics
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Total episodes: {n_episodes}")
    print(f"Times hit max deals limit: {all_stats['max_deals_hits']} ({100*all_stats['max_deals_hits']/n_episodes:.1f}%)")
    print(f"Average deals per episode: {np.mean(all_stats['deals_per_episode']):.1f}")
    print(f"Median deals per episode: {np.median(all_stats['deals_per_episode']):.1f}")
    print(f"Max deals in any episode: {np.max(all_stats['deals_per_episode'])}")
    print(f"Average % positive bets: {np.mean(all_stats['positive_bet_percentages']):.1f}%")
    print(f"Final 100 episodes avg deals: {np.mean(all_stats['deals_per_episode'][-100:]):.1f}")

    # Sanity check the learned strategy
    players[0].sanity_check_learned_strategy()

    players[0].describe_learned_q_function()


def main(n_players=3):

    run_sarsa(n_players)


if __name__ == "__main__":
    main()
