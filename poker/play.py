import numpy as np

from poker.state import State


ACTION_FOLD = -1
ACTION_BET = 1


class SimpleRandomPlayer:
    def __init__(self):
        self.name = "Hi there, I'm Bob"

    def get_action(self, state):

        return np.random.choice([ACTION_FOLD, ACTION_BET], p=[0.1, 0.9])


def run_game(n_players, n_steps=20):

    players = [SimpleRandomPlayer() for _ in range(n_players)]

    state = State(n_players=n_players, verbose=True)

    for step in range(n_steps):

        current_player = state.current_player

        # TODO Need to prevent illegal actions
        #  For example, you cannot bet less than the amount you need to call
        action = players[current_player].get_action(state)
        state.update(action)


def main(n_players=3):

    run_game(n_players)


if __name__ == "__main__":
    main()
