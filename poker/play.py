import numpy as np

from poker.cards import FULL_DECK
from poker.state import GameStage, State


class SimpleRandomPlayer:
    def get_action(self, state):

        action_fold = -1
        action_bet = 1

        return np.random.choice([action_fold, action_bet], p=[0.1, 0.9])


class SimpleQFunctionPlayer:
    def __init__(self):

        n_game_stages = len(GameStage)
        n_cards = len(FULL_DECK)

        # Note: this is a very crude (incomplete) representation of the player's state:
        #  It tracks the game stage and the player's hole cards (i.e. their private cards),
        #  and nothing else
        self.q = np.zeros((n_game_stages, n_cards, n_cards, n_actions))


def run_game(n_players, n_steps=30):

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
