from enum import IntEnum
from random import sample

import numpy as np

from poker.cards import Suit, Rank, Card, FULL_DECK
from poker.hands import best_hand_strength
from poker.utils import argmax


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
        verbose=False,
        deck=None,
    ):

        self.n_players = n_players

        self.wealth = [initial_wealth for player in range(self.n_players)]

        self.initialize_pre_flop(dealer=initial_dealer, deck=deck)

        self.verbose = verbose

        if self.verbose:
            print(
                f"Initialized game with {self.n_players} players each with wealth ${initial_wealth}"
            )

        # TODO Implement small blind and big blind,
        #  first person to act pre-flop is player after the big blind

    def __str__(self):

        return f"State: game stage {self.game_stage.name}, total pot ${self.total_bets()}, player {self.current_player} is next to act"

    def total_bets(self):

        total = 0
        for stage_bets in self.bets_by_stage.values():
            total += sum(sum(player_bets) for player_bets in stage_bets)

        return total

    def initialize_pre_flop(self, dealer, deck=None):

        if deck is None:
            self.shuffled_deck = sample(FULL_DECK, k=len(FULL_DECK))
        else:
            self.shuffled_deck = deck

        self.public_cards = []

        # Note: these are the private (face down) cards which are hidden from other players
        #  Player i observes only hole_cards[i]
        self.hole_cards = self.deal_hole_cards()

        self.game_stage = GameStage.PRE_FLOP

        self.dealer = dealer

        self.has_folded = [False for player in range(self.n_players)]

        # Note: this is the player to the left of the dealer
        self.current_player = self.get_next_player(self.dealer)

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

        if len(self.bets_by_stage[self.game_stage][next_player]) == 0:
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

            if self.verbose:
                print(f"Player {self.current_player} folds")

        else:
            self.bets_by_stage[self.game_stage][self.current_player].append(action)

            if self.verbose:
                print(f"Player {self.current_player} bets ${action}")

    def redistribute_wealth_and_reinitialize(self, winning_players):

        losing_players = set(range(self.n_players)).difference(winning_players)

        for losing_player in losing_players:

            for stage in GameStage:
                total_bet_by_losing_player = sum(
                    self.bets_by_stage[stage][losing_player]
                )
                self.wealth[losing_player] -= total_bet_by_losing_player

                for winning_player in winning_players:
                    # Note: if there are multiple winning players, they split the pot
                    self.wealth[winning_player] += total_bet_by_losing_player / len(
                        winning_players
                    )

        if self.verbose:
            print(f"Player wealths are now {self.wealth}")

        # Now that we have redistributed wealth, we assign a new dealer,
        #  deal new cards and go back to the initial stage
        next_dealer = (self.dealer + 1) % self.n_players
        self.initialize_pre_flop(dealer=next_dealer)

    def calculate_best_hand_strengths(self):

        hand_strengths, hand_descriptions = ([], [])

        for player in range(self.n_players):

            if self.has_folded[player]:
                # Note: players who have folded are given a negative hand strength,
                #  which prevents them from ever winning
                hand_strengths.append(-1)
                hand_descriptions.append("player has folded")

            else:
                hand_strength, hand_description = best_hand_strength(
                    self.public_cards, self.hole_cards[player]
                )
                hand_strengths.append(hand_strength)
                hand_descriptions.append(hand_description)

        return hand_strengths, hand_descriptions

    def move_to_next_stage(self):

        self.game_stage = list(GameStage)[self.game_stage + 1]

        # Note: after moving to the next stage, the first person to act is
        #  the first person left of the dealer (ignoring players who have already folded)
        self.current_player = self.get_next_player(self.dealer)

        if self.game_stage == GameStage.FLOP:
            self.public_cards.extend(self.deal_k_cards(3))

        elif self.game_stage == GameStage.TURN or self.game_stage == GameStage.RIVER:
            self.public_cards.extend(self.deal_k_cards(1))

    def update(self, action):

        if self.verbose:
            print(self)

        self.update_has_folded_or_bets(action)

        over_due_to_folding = sum(self.has_folded) >= self.n_players - 1

        next_player = self.get_next_player(self.current_player)

        if not over_due_to_folding and self.stage_is_complete(next_player):

            if self.game_stage <= GameStage.TURN:

                self.move_to_next_stage()

            else:

                # Note: we've reached the river and the stage is complete,
                #  so we need to figure out who has the strongest hand
                hand_strengths, hand_descriptions = self.calculate_best_hand_strengths()

                # TODO This is incorrect if there are ties (multiple players with the same hand),
                #  in which case the winners split the pot
                winning_players = argmax(hand_strengths)

                if self.verbose:
                    print(f"Hands: {hand_descriptions}")
                    winning_hand_description = hand_descriptions[winning_players[0]]
                    print(
                        f"Player(s) {winning_players} win the hand with {winning_hand_description} (hand strength {max(hand_strengths)})"
                    )
                    print(f"Public cards: {self.public_cards}")
                    print(f"Hole cards: {self.hole_cards}")

                self.redistribute_wealth_and_reinitialize(winning_players)

        elif over_due_to_folding:

            if self.verbose:
                print(
                    f"Player {next_player} wins the hand because everyone else has folded"
                )

            winning_players = [next_player]
            self.redistribute_wealth_and_reinitialize(winning_players)

        else:

            # Note: in this case, the current stage is not over
            #  and it is the next player's turn to act
            self.current_player = next_player
