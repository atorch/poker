"""
Tests for the betting rules enforced by State:
- folding is only legal when facing a bet
- a per-stage cap on voluntary raises
- automatic run-out to showdown once nobody can bet
- the stage ends when the last player to act folds
- split pots are paid in whole chips (odd chips to the first winner left of the dealer)
"""
import random

import numpy as np
import pytest

from poker.cards import Card, Rank, Suit, FULL_DECK
from poker.config import DEFAULT_ACTIONS, MAX_RAISES_PER_STAGE
from poker.random_agent import RandomAgent
from poker.state import GameStage, State

FOLD, CHECK = -1, 0


def cards(text):
    """Parse a compact hand like "Js 7h Ad" into Card objects (T = ten)."""
    ranks = {"2": Rank.TWO, "3": Rank.THREE, "4": Rank.FOUR, "5": Rank.FIVE, "6": Rank.SIX,
             "7": Rank.SEVEN, "8": Rank.EIGHT, "9": Rank.NINE, "T": Rank.TEN, "J": Rank.JACK,
             "Q": Rank.QUEEN, "K": Rank.KING, "A": Rank.ACE}
    suits = {"h": Suit.HEARTS, "d": Suit.DIAMONDS, "c": Suit.CLUBS, "s": Suit.SPADES}
    return [Card(ranks[token[0]], suits[token[1]]) for token in text.split()]


def stacked_deck(hole_cards_by_player, board):
    """
    Build a deck that deals the given hole cards (player 0 first) and then the board.

    Cards are popped off the end of the deck, so the deal order is reversed.
    """
    dealt_in_order = [card for hole in hole_cards_by_player for card in cards(hole)] + cards(board)
    rest = [card for card in FULL_DECK if card not in dealt_in_order]
    return rest + list(reversed(dealt_in_order))


def check_through_preflop(state):
    """With dealer 0 in a 3-player game: dealer calls, small blind completes, big blind checks."""
    state.update(state.big_blind)
    state.update(state.big_blind - state.small_blind)
    state.update(CHECK)
    assert state.game_stage == GameStage.FLOP


def test_fold_is_illegal_when_checking_is_free():
    state = State(n_players=3, initial_wealth=20, initial_dealer=0)
    check_through_preflop(state)

    assert state.minimum_legal_bet() == 0
    assert not state.is_legal(FOLD)
    assert FOLD not in state.legal_actions()
    with pytest.raises(AssertionError):
        state.update(FOLD)


def test_fold_is_legal_when_facing_a_bet():
    state = State(n_players=3, initial_wealth=20, initial_dealer=0)

    # Note: the dealer faces the big blind pre-flop
    assert state.minimum_legal_bet() == state.big_blind
    assert state.is_legal(FOLD)
    assert state.legal_actions() == [FOLD, 2, 3]


def test_raise_cap_limits_raises_per_stage():
    state = State(n_players=3, initial_wealth=100, initial_dealer=0, max_raises_per_stage=3)
    check_through_preflop(state)

    # Note: player 1 bets $1, player 2 raises to $2, player 0 raises to $3: 3 voluntary raises
    state.update(1)
    state.update(2)
    state.update(3)
    assert state.raises_by_stage[GameStage.FLOP] == 3
    assert state.raise_cap_reached()

    # Note: player 1 owes $2; calling is legal but raising is not
    assert state.current_player == 1
    assert state.legal_actions() == [FOLD, 2]
    with pytest.raises(AssertionError):
        state.update(3)
    state.update(2)

    # Note: player 2 owes $1
    assert state.legal_actions() == [FOLD, 1]
    state.update(1)
    assert state.game_stage == GameStage.TURN


def test_raise_cap_resets_each_stage_and_ignores_blinds():
    state = State(n_players=3, initial_wealth=100, initial_dealer=0, max_raises_per_stage=1)

    # Note: forced blinds don't count, so the dealer may still raise pre-flop
    assert state.raises_by_stage[GameStage.PRE_FLOP] == 0
    assert 3 in state.legal_actions()
    state.update(3)
    assert state.raise_cap_reached()

    # Note: small blind owes $2 and big blind owes $1: call or fold only
    assert state.legal_actions() == [FOLD, 2]
    state.update(2)
    assert state.legal_actions() == [FOLD, 1]
    state.update(1)

    assert state.game_stage == GameStage.FLOP
    assert not state.raise_cap_reached()
    assert state.legal_actions() == [CHECK, 1, 2, 3]


def test_default_raise_cap_is_used():
    state = State(n_players=3)
    assert state.max_raises_per_stage == MAX_RAISES_PER_STAGE


def test_stage_ends_when_last_player_to_act_folds():
    # Regression: the stage used to stay open when the last player folded instead of calling,
    #  giving the bettor an extra decision (in which they could even raise again)
    state = State(n_players=3, initial_wealth=20, initial_dealer=0)
    check_through_preflop(state)

    state.update(CHECK)  # player 1
    state.update(2)  # player 2 bets
    state.update(2)  # player 0 calls
    assert state.current_player == 1
    state.update(FOLD)

    assert state.game_stage == GameStage.TURN
    assert state.current_player == 2


def test_all_in_runs_out_the_board_to_showdown():
    state = State(n_players=3, initial_wealth=6, initial_dealer=0)

    # Note: each action adds at most $3, so it takes a few raises to put everyone all in
    state.update(3)  # dealer: total 3
    state.update(3)  # small blind: total 4
    state.update(2)  # big blind: total 4
    state.update(3)  # dealer: total 6 (all in)
    state.update(2)  # small blind calls: total 6
    n_deals_before = state.n_deals
    state.update(2)  # big blind calls: total 6

    # Note: nobody can bet any more, so the flop, turn and river are dealt at once
    #  and nobody is asked to check through the remaining stages
    assert state.n_deals == n_deals_before + 1 or state.terminal
    assert len(state.last_deal_public_cards) == 5
    assert not state.last_deal_won_by_fold
    assert sum(state.wealth) == 18


def test_no_run_out_while_players_can_still_bet():
    state = State(n_players=3, initial_wealth=20, initial_dealer=0)
    check_through_preflop(state)

    assert state.game_stage == GameStage.FLOP
    assert len(state.public_cards) == 3
    assert state.remaining_betting_room() == 18


def test_board_two_pair_ace_kicker_wins_whole_pot():
    # Regression for an interactive game that was scored as a three-way split:
    #  the board is two pair (jacks and sevens) and player 2's ace kicker plays
    deck = stacked_deck(["3h Kd", "2d 8h", "6d Ac"], "7s Jh Ts Js 7h")
    state = State(n_players=3, initial_wealth=20, initial_dealer=0, deck=deck)
    assert state.hole_cards == [cards("3h Kd"), cards("2d 8h"), cards("6d Ac")]

    check_through_preflop(state)
    for _ in range(3):  # flop, turn, river
        for _ in range(state.n_players):
            state.update(CHECK)

    assert state.last_deal_winners == [2]
    assert state.wealth == [18, 18, 24]


def test_odd_chip_goes_to_first_winner_left_of_dealer():
    # Note: the board is a royal flush, so players 1 and 2 tie and player 0 folds
    deck = stacked_deck(["2c 3d", "4c 5d", "6c 7d"], "Ts Js Qs Ks As")
    state = State(n_players=3, initial_wealth=20, initial_dealer=0, deck=deck)

    state.update(3)  # dealer raises to $3 total
    state.update(2)  # small blind calls (total $3)
    state.update(1)  # big blind calls (total $3)
    assert state.game_stage == GameStage.FLOP

    state.update(1)  # player 1 bets $1
    state.update(1)  # player 2 calls
    state.update(FOLD)  # player 0 folds: loses $3, which can't be split evenly

    for _ in range(2):  # turn and river
        for _ in range(2):
            state.update(CHECK)

    assert state.last_deal_winners == [1, 2]
    # Note: player 1 is closest to the dealer's left, so they get the odd chip
    assert state.wealth == [17, 22, 21]


def test_random_play_respects_rules():
    """Play many random games and check invariants at every decision."""
    random.seed(0)
    np.random.seed(0)

    for _ in range(200):
        initial_wealth = np.random.randint(2, 40)
        state = State(n_players=3, initial_wealth=initial_wealth, initial_dealer=np.random.randint(3))
        players = [RandomAgent(player_index=index) for index in range(3)]

        while not state.terminal and state.n_deals < 50:
            legal_actions = state.legal_actions(DEFAULT_ACTIONS)
            assert legal_actions, "there must always be at least one legal action"

            # Note: folding is legal exactly when facing a bet
            assert (FOLD in legal_actions) == (state.minimum_legal_bet() > 0)

            assert state.raises_by_stage[state.game_stage] <= state.max_raises_per_stage

            # Note: once betting is closed the board is run out, so post-flop nobody is
            #  ever asked to act when checking is their only option
            if state.game_stage != GameStage.PRE_FLOP:
                assert legal_actions != [CHECK]

            state.update(players[state.current_player].get_action(state))

            assert sum(state.wealth) == 3 * initial_wealth
            assert all(float(wealth).is_integer() for wealth in state.wealth)
