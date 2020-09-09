from enum import IntEnum
from random import sample

from poker.cards import Suit, Rank, Card, FULL_DECK


class GameStage(IntEnum):

    PRE_FLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


class State:
    def __init__(self, n_players=3):

        self.n_players = n_players

        self.shuffled_deck = sample(FULL_DECK, k=len(FULL_DECK))

        # Note: these are the hidden (face down) cards which are hidden from other players
        #  Player i observes only hole_cards[i]
        self.hole_cards = self.deal_hole_cards()

        self.game_stage = GameStage.PRE_FLOP

        self.dealer = 0

        # Note: this is the player to the left of the dealer
        self.current_player = 1

        self.has_folded = [False for player in range(self.n_players)]

        self.bets_by_stage = {
            stage: [[] for player in range(self.n_players)] for stage in GameStage
        }

    def get_next_player(self, current_player):

        next_player = (current_player + 1) % self.n_players
        move_to_next_stage = False

        while self.has_folded[next_player]:

            next_player = (next_player + 1) % self.n_players

            if next_player == current_player:
                break

        return next_player, move_to_next_stage

    def deal_two_cards(self):

        first_card = self.shuffled_deck.pop()
        second_card = self.shuffled_deck.pop()

        return [first_card, second_card]

    def deal_hole_cards(self):

        return [self.deal_two_cards() for player in range(self.n_players)]

    def update(self, action):

        player = self.current_player
        stage = self.game_stage

        if action < 0:
            # Note: negative bets indicate that the player is folding
            self.has_folded[player] = True

        else:
            self.bets_by_stage[stage][player].append(action)

        next_player, move_to_next_stage = self.get_next_player(player)

        if move_to_next_stage:
            self.game_stage = self.game_stage + 1

        else:
            self.current_player = next_player
