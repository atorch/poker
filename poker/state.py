from random import sample

from poker.cards import Suit, Rank, Card, FULL_DECK


class State:
    def __init__(self, n_players=3):

        self.n_players = n_players

        self.shuffled_deck = sample(FULL_DECK, k=len(FULL_DECK))

        # Note: these are the hidden (face down) cards which are hidden from other players
        #  Player i observes only hole_cards[i]
        self.hole_cards = self.deal_hole_cards()

    def deal_two_cards(self):

        first_card = self.shuffled_deck.pop()
        second_card = self.shuffled_deck.pop()

        return [first_card, second_card]

    def deal_hole_cards(self):

        return [self.deal_two_cards() for player in range(self.n_players)]


def get_next_state(current_state, action):

    # TODO Write this function
    return State(n_players=current_state.n_players)
