import numpy as np

from poker.state import State
from poker.agent import Agent


def run_one_episode(episode, players, initial_wealth=100):

    # Note: the probability of random (exploratory) actions decreases over time
    proba_random_action = 0.02 + 0.98 * np.exp(-episode / 500)

    state = State(n_players=len(players), initial_wealth=initial_wealth, verbose=False)

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

        wealth_before_action = state.wealth[learning_player]

        state.update(action)

        # Note: we need to make the other players act so that we can get back to the learning player
        #  Even though the other (non-learning) players are acting, it
        #  is possible that the learning player is forced
        #  to act (bet the small blind or big blind) in this block
        while state.current_player != learning_player:

            action = players[state.current_player].get_action(
                state, proba_random_action
            )
            state.update(action)

            # TODO Is this needed?
            if state.terminal:
                break

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

        continuation_value = players[learning_player].predicted_q(
            next_private_state, next_action
        )

        if state.terminal:
            print(f"Reached a terminal state: player wealths are {state.wealth}")
            updated_guess_for_q = reward

        else:
            updated_guess_for_q = reward + continuation_value

        players[learning_player].update_q(private_state, action, updated_guess_for_q)

        action = next_action
        private_state = next_private_state


def run_sarsa(n_players, n_episodes=2000):

    # This is (roughly) Sutton and Barto Figure 6.9
    # page 130, TODO compare to page 131
    # page 244

    players = [Agent(player_index) for player_index in range(n_players)]

    for episode in range(n_episodes):

        run_one_episode(episode, players)

    players[0].describe_learned_q_function()


def main(n_players=3):

    run_sarsa(n_players)


if __name__ == "__main__":
    main()
