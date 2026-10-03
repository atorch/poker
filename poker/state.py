from enum import IntEnum
from random import sample

import numpy as np

from poker.cards import Suit, Rank, Card, FULL_DECK
from poker.config import DEFAULT_ACTIONS, MAX_RAISES_PER_STAGE
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
        max_raises_per_stage=MAX_RAISES_PER_STAGE,
    ):

        self.n_players = n_players
        self.big_blind = big_blind
        self.small_blind = small_blind
        # Note: caps the number of voluntary raises (including the opening bet) per stage,
        #  as in limit hold'em. Without a cap, betting escalates until someone is all in.
        #  Forced blinds do not count toward the cap.
        self.max_raises_per_stage = max_raises_per_stage
        self.wealth = [initial_wealth for player in range(self.n_players)]
        self.verbose = verbose

        if self.verbose:
            print(
                f"Initialized game with {self.n_players} players each with wealth ${initial_wealth}"
            )

        # Note: we track the number of times the deck has been shuffled
        #  (i.e. the number of rounds that have been played)
        # TODO: Clarify semantics - n_deals is incremented in initialize_pre_flop,
        #   so it starts at 1 (not 0). Should comment say "deals started" not "deals completed"?
        #   Or should we move the increment to after round completes?
        self.n_deals = 0

        # Track last deal result for UI display
        self.last_deal_winners = None
        self.last_deal_won_by_fold = False
        self.last_deal_pot = 0
        self.last_deal_public_cards = []
        self.last_deal_hole_cards = []
        self.last_deal_hand_strengths = None
        self.last_deal_hand_descriptions = None
        self.last_deal_folded_players = []
        self.last_deal_bets_by_player = []

        self.initialize_pre_flop(dealer=initial_dealer, deck=deck)

        self.terminal = False

    def __str__(self):

        return f"State: game stage {self.game_stage.name}, total pot ${self.total_bets()}, player {self.current_player} is next to act"

    def initialize_pre_flop(self, dealer, deck=None):

        self.n_deals += 1

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

        # Note: number of voluntary bets/raises made so far in each stage (see max_raises_per_stage)
        self.raises_by_stage = {stage: 0 for stage in GameStage}

        # Note: the first player to act is forced to bet the small blind,
        #  and the second player to act is forced to bet the big blind,
        #  as long as the blinds are not larger than anyone's wealth
        # TODO Forcing the player to act in this way might create odd results for Q function
        #  and for rewards. Might be better to let the player act but _constrain_ their
        #  actions so that they are forced to play the blinds. Does this matter?
        min_wealth = min(self.wealth)

        small_blind = min(self.small_blind, min_wealth)
        self.update(small_blind, forced=True)

        big_blind = min(self.big_blind, min_wealth)
        self.update(big_blind, forced=True)

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

    def total_bets(self):

        total = 0
        for stage_bets in self.bets_by_stage.values():
            total += sum(sum(player_bets) for player_bets in stage_bets)

        return total

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

    def remaining_betting_room(self):
        """
        Chips each active player can still add this deal before the shortest active stack is all in.

        At the end of a stage all active players have bet the same total, so when this is
        zero no further betting is possible and the remaining cards can be dealt immediately.
        """
        return min(
            self.wealth[player] - self.total_bet_by_player(player)
            for player in range(self.n_players)
            if not self.has_folded[player]
        )

    def raise_cap_reached(self):

        return self.raises_by_stage[self.game_stage] >= self.max_raises_per_stage

    def is_legal(self, action):
        """
        Whether the current player may take `action` (negative = fold, otherwise chips to add).

        Rules:
        - Folding is only allowed when facing a bet (folding when you can check for free is
          never better than checking, and folding when all in just forfeits the pot)
        - Calling/betting must be between the minimum and maximum legal bet
        - Bets above the call amount (raises) are not allowed once the raise cap is reached
        """
        minimum_legal_bet = self.minimum_legal_bet()

        if action < 0:
            return minimum_legal_bet > 0

        if action > minimum_legal_bet and self.raise_cap_reached():
            return False

        return minimum_legal_bet <= action <= self.maximum_legal_bet()

    def legal_actions(self, actions=DEFAULT_ACTIONS):
        """The subset of `actions` the current player may take (never empty for DEFAULT_ACTIONS)."""

        return [action for action in actions if self.is_legal(action)]

    def stage_is_complete(self):

        active_players = [player for player in range(self.n_players) if not self.has_folded[player]]
        stage_bets = self.bets_by_stage[self.game_stage]

        # For pre-flop, ensure players who posted forced blinds have acted voluntarily
        # (more than just the forced blind action)
        if self.game_stage == GameStage.PRE_FLOP:
            # Small blind is player left of dealer
            sb_player = (self.dealer + 1) % self.n_players
            # Big blind is player left of small blind
            bb_player = (self.dealer + 2) % self.n_players
            blind_players = [sb_player, bb_player]
        else:
            blind_players = []

        # Every active player must have acted voluntarily at least once in this stage
        # (SB and BB need 2 actions pre-flop: forced blind + voluntary decision)
        for player in active_players:
            n_required_actions = 2 if player in blind_players else 1
            if len(stage_bets[player]) < n_required_actions:
                return False

        # Note: the stage is complete when every active player has bet the same total in this stage.
        #  We compare all active players (not just the current and next player) so that the stage
        #  also ends when the last player to act folds rather than calls
        return len({sum(stage_bets[player]) for player in active_players}) == 1

    def update_has_folded_or_bets(self, action, forced=False):

        # Note: forced blinds skip the legality checks and do not count as raises
        if not forced:
            # Note: blow up if the player tries to take an illegal action
            assert self.is_legal(action), (
                f"Illegal action {action} for player {self.current_player}: "
                f"min bet {self.minimum_legal_bet()}, max bet {self.maximum_legal_bet()}, "
                f"raise cap reached {self.raise_cap_reached()}"
            )

        # Note: negative bets indicate that the player is folding
        if action < 0:

            self.has_folded[self.current_player] = True

            if self.verbose:
                print(f"Player {self.current_player} folds")

        else:

            if not forced and action > self.minimum_legal_bet():
                self.raises_by_stage[self.game_stage] += 1

            self.bets_by_stage[self.game_stage][self.current_player].append(action)

            if self.verbose:
                print(f"Player {self.current_player} bets ${action}")

    def redistribute_wealth_and_reinitialize(self, winning_players, won_by_fold=False, hand_strengths=None, hand_descriptions=None):

        # Store last deal result for UI display (before state is reset)
        self.last_deal_winners = list(winning_players)
        self.last_deal_won_by_fold = won_by_fold
        self.last_deal_pot = self.total_bets()
        self.last_deal_public_cards = list(self.public_cards) if self.public_cards else []
        self.last_deal_hole_cards = [list(cards) for cards in self.hole_cards]  # Deep copy
        self.last_deal_hand_strengths = list(hand_strengths) if hand_strengths else None
        self.last_deal_hand_descriptions = list(hand_descriptions) if hand_descriptions else None
        self.last_deal_folded_players = [i for i in range(self.n_players) if self.has_folded[i]]
        self.last_deal_bets_by_player = [self.total_bet_by_player(i) for i in range(self.n_players)]

        losing_players = set(range(self.n_players)).difference(winning_players)

        total_lost = 0
        for losing_player in losing_players:

            total_bet_by_losing_player = self.total_bet_by_player(losing_player)
            self.wealth[losing_player] -= total_bet_by_losing_player
            total_lost += total_bet_by_losing_player

            if self.wealth[losing_player] <= 0:
                # Note: for simplicity, the game ends as soon as any player runs out of money
                self.terminal = True

        # Note: if there are multiple winning players, they split the pot in whole chips.
        #  Any odd chips go one at a time to the winners closest to the dealer's left
        #  (the standard rule), which keeps every stack a whole number of chips.
        winners_in_seat_order = sorted(
            winning_players, key=lambda player: (player - self.dealer - 1) % self.n_players
        )
        share, odd_chips = divmod(total_lost, len(winners_in_seat_order))
        for seat_index, winning_player in enumerate(winners_in_seat_order):
            self.wealth[winning_player] += share + (1 if seat_index < odd_chips else 0)

        if self.verbose:
            print(f"Player wealths are now {self.wealth}")

        # Only start a new round if the game hasn't ended
        if not self.terminal:
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

        # Note: Don't set current_player here - let update() handle it
        # to avoid the player who completed the stage acting again

        if self.game_stage == GameStage.FLOP:
            self.public_cards.extend(self.deal_k_cards(3))

        elif self.game_stage == GameStage.TURN or self.game_stage == GameStage.RIVER:
            self.public_cards.extend(self.deal_k_cards(1))

        if self.verbose:
            print(f"Public cards are {self.public_cards}")

    def showdown(self):

        # Note: we've reached the river and the stage is complete (or nobody can bet any more),
        #  so we need to figure out who has the strongest hand
        hand_strengths, hand_descriptions = self.calculate_best_hand_strengths()

        # Note: ties (multiple players with equally strong hands) split the pot
        winning_players = argmax(hand_strengths)

        if self.verbose:
            print(f"Hands: {hand_descriptions}")
            winning_hand_description = hand_descriptions[winning_players[0]]
            print(
                f"Player(s) {winning_players} win the hand with {winning_hand_description} (hand strength {max(hand_strengths)})"
            )
            print(f"Public cards: {self.public_cards}")
            print(f"Hole cards: {self.hole_cards}")

        self.redistribute_wealth_and_reinitialize(
            winning_players,
            won_by_fold=False,
            hand_strengths=hand_strengths,
            hand_descriptions=hand_descriptions
        )

    def update(self, action, forced=False):

        if self.verbose:
            print(self)

        self.update_has_folded_or_bets(action, forced=forced)

        over_due_to_folding = sum(self.has_folded) >= self.n_players - 1

        next_player = self.get_next_player(self.current_player)

        if not over_due_to_folding and self.stage_is_complete():

            if self.game_stage <= GameStage.TURN and self.remaining_betting_room() <= 0:

                # Note: the shortest active stack is all in, so nobody can bet any more:
                #  deal the remaining public cards and go straight to the showdown
                #  (rather than asking players to check through each remaining stage)
                while self.game_stage < GameStage.RIVER:
                    self.move_to_next_stage()

                self.showdown()

            elif self.game_stage <= GameStage.TURN:

                self.move_to_next_stage()

                # After stage transition, first player to act is left of dealer
                self.current_player = self.get_next_player(self.dealer)

            else:

                self.showdown()

        elif over_due_to_folding:

            if self.verbose:
                print(
                    f"Player {next_player} wins the hand because everyone else has folded"
                )

            winning_players = [next_player]
            self.redistribute_wealth_and_reinitialize(winning_players, won_by_fold=True)

        else:

            # Note: in this case, the current stage is not over
            #  and it is the next player's turn to act
            self.current_player = next_player
