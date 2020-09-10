from enum import IntEnum
from random import sample

from poker.cards import Suit, Rank, Card, FULL_DECK


class GameStage(IntEnum):

    PRE_FLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


class State:
    def __init__(
        self,
        n_players=3,
        initial_wealth=100,
        big_blind=2,
        small_blind=1,
        initial_dealer=0,
    ):

        self.n_players = n_players

        self.shuffled_deck = sample(FULL_DECK, k=len(FULL_DECK))

        # Note: these are the private (face down) cards which are hidden from other players
        #  Player i observes only hole_cards[i]
        self.hole_cards = self.deal_hole_cards()

        self.public_cards = []

        self.game_stage = GameStage.PRE_FLOP

        self.dealer = initial_dealer

        self.has_folded = [False for player in range(self.n_players)]

        # Note: this is the player to the left of the dealer
        self.current_player = self.get_next_player(self.dealer)

        self.wealth = [initial_wealth for player in range(self.n_players)]

        self.bets_by_stage = {
            stage: [[] for player in range(self.n_players)] for stage in GameStage
        }

    def get_next_player(self, current_player):

        next_player = (current_player + 1) % self.n_players

        while self.has_folded[next_player]:

            next_player = (next_player + 1) % self.n_players

            if next_player == current_player:
                break

        return next_player

    def deal_k_cards(self, k=2):

        return [self.shuffled_deck.pop() for _ in range(k)]

    def deal_hole_cards(self):

        return [self.deal_k_cards(2) for player in range(self.n_players)]

    def stage_is_complete(self, next_player):

        if len(self.bets_by_stage[self.game_stage][self.current_player]) == 0:
            return False

        # Note: these are totals for the current stage only (e.g. player 1 has bet a total of $40 during the turn)
        total_bet_current_player = sum(
            self.bets_by_stage[self.game_stage][self.current_player]
        )
        total_bet_next_player = sum(self.bets_by_stage[self.game_stage][next_player])

        return total_bet_current_player == total_bet_next_player

    def update_has_folded_or_bets(self, action):

        if action < 0:
            # Note: negative bets indicate that the player is folding
            self.has_folded[self.current_player] = True

        else:
            self.bets_by_stage[self.game_stage][self.current_player].append(action)

    def update(self, action):

        self.update_has_folded_or_bets(action)

        over_due_to_folding = sum(self.has_folded) >= self.n_players - 1

        next_player = self.get_next_player(self.current_player)

        if not over_due_to_folding and self.stage_is_complete(next_player):

            # TODO This won't work if we've reached this river :-) need to find the strongest hand
            self.game_stage += 1

            # Note: after moving to the next stage, the first person to act is
            #  the first person left of the dealer (ignoring players who have already folded)
            self.current_player = self.get_next_player(self.dealer)

            if self.game_stage == GameStage.FLOP:
                self.public_cards.extend(self.deal_k_cards(3))
            elif (
                self.game_stage == GameStage.TURN or self.game_stage == GameStage.RIVER
            ):
                self.public_cards.extend(self.deal_k_cards(1))

        elif over_due_to_folding:
            # TODO Finish this: if all other players have folded, the remaining player wins their bets
            # TODO Remove wealth from other players
            winning_player = next_player

            losing_players = [
                player for player in range(self.n_players) if player != winning_player
            ]

            for losing_player in losing_players:

                for stage in GameStage:
                    total_bet_by_losing_player = sum(
                        self.bets_by_stage[stage][losing_player]
                    )
                    self.wealth[winning_player] += total_bet_by_losing_player
                    self.wealth[losing_player] -= total_bet_by_losing_player

        else:
            self.current_player = next_player
