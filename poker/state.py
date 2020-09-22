from enum import IntEnum
from random import sample

import numpy as np

from poker.cards import Suit, Rank, Card, FULL_DECK
from poker.hands import best_hand_strength, sort_hand
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
        initial_wealth=100.0,
        big_blind=2,
        small_blind=1,
        initial_dealer=0,
        verbose=False,
        deck=None,
    ):

        self.n_players = n_players
        self.big_blind = big_blind
        self.small_blind = small_blind
        self.wealth = [initial_wealth for player in range(self.n_players)]
        self.verbose = verbose

        if self.verbose:
            print(
                f"Initialized game with {self.n_players} players each with wealth ${initial_wealth}"
            )

        self.initialize_pre_flop(dealer=initial_dealer, deck=deck)

        self.terminal = False

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

        # Note: the first player to act is forced to bet the small blind,
        #  and the second player to act is forced to bet the big blind,
        #  as long as the blinds are not larger than anyone's wealth
        # TODO Forcing the player to act in this way might create odd results for Q function
        #  and for rewards. Might be better to let the player act but _constrain_ their
        #  actions so that they are forced to play the blinds. Does this matter?
        min_wealth = min(self.wealth)

        small_blind = min(self.small_blind, min_wealth)
        self.update(small_blind)

        big_blind = min(self.big_blind, min_wealth)
        self.update(big_blind)

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

        # Note: we sort the hole cards to reduce number of duplicate states
        #  The order of a player's hole cards does not affect the strength of their hand
        return [sort_hand(self.deal_k_cards(2)) for player in range(self.n_players)]

    def total_bet_by_player(self, player_index):

        return sum(sum(self.bets_by_stage[stage][player_index]) for stage in GameStage)

    def maximum_legal_bet(self):

        # Note: we don't allow bets that would put any non-folded player past all in
        #  That way, we can keep things simple and ignore side pots (because they can't happen)

        total_bet_by_current_player = self.total_bet_by_player(self.current_player)

        return min(
            self.wealth[player] - total_bet_by_current_player
            for player in range(self.n_players)
            if not self.has_folded[player]
        )

    def minimum_legal_bet(self):

        total_bet_current_player = sum(
            self.bets_by_stage[self.game_stage][self.current_player]
        )

        # Note: we want the previous player who has not folded
        previous_player = (self.current_player - 1) % self.n_players
        while self.has_folded[previous_player]:
            previous_player = (previous_player - 1) % self.n_players

        total_bet_previous_player = sum(
            self.bets_by_stage[self.game_stage][previous_player]
        )

        return total_bet_previous_player - total_bet_current_player

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

        # TODO This is incorrect pre-flop, when the big blind is allowed to raise
        # TODO Could this lead to an infinite loop / never-ending stage if a player does not have enough wealth to complete the bet?
        # TODO Pytest edge cases where one player has low (but positive) wealth
        return total_bet_current_player == total_bet_next_player

    def update_has_folded_or_bets(self, action):

        # Note: negative bets indicate that the player is folding
        if action < 0:

            self.has_folded[self.current_player] = True

            if self.verbose:
                print(f"Player {self.current_player} folds")

        else:

            minimum_legal_bet = self.minimum_legal_bet()
            maximum_legal_bet = self.maximum_legal_bet()

            # Note: blow up if the player tries to take an illegal action
            assert minimum_legal_bet <= action <= maximum_legal_bet

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

                if self.wealth[losing_player] <= 0:
                    # Note: for simplicity, the game ends as soon as any player runs out of money
                    self.terminal = True

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

        if self.verbose:
            print(f"Public cards are {self.public_cards}")

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
