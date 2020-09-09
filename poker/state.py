from random import sample

from poker.cards import Suit, Rank, Card, FULL_DECK


class State:

    def __init__(self, n_players=3):

        self.n_players = n_players

        self.shuffled_deck = sample(FULL_DECK, k=len(FULL_DECK))


def get_next_state(current_state, action):

    # TODO Write this function
    return State(n_players=current_state.n_players)
