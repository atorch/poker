"""
Tests for the baseline agents (MaxRaise, CallStation, TagAgent) and the card/hand helpers they use.
"""
import random

import numpy as np
import pytest

from poker.baseline_agents import (
    MEDIUM, STRONG, WEAK, CallStationAgent, MaxRaiseAgent, TagAgent, postflop_hand_class, preflop_hand_class,
)
from poker.cards import Rank, parse_cards, stacked_deck
from poker.hands import HandCategory, best_hand_strength, hand_category, leading_rank, made_hand_category
from poker.random_agent import RandomAgent
from poker.state import GameStage, State

FOLD, CHECK = -1, 0


def test_stacked_deck_deals_given_cards():
    deck = stacked_deck([parse_cards("As Kd"), None, parse_cards("7h 2c")], parse_cards("Ah 7c 2d Ts 9s"),
                        rng=random.Random(0))
    assert len(deck) == 52
    state = State(n_players=3, initial_wealth=20, deck=deck)
    assert state.hole_cards[0] == parse_cards("Kd As")  # hole cards are sorted by rank
    assert state.hole_cards[2] == parse_cards("2c 7h")
    assert state.shuffled_deck[-3:] == list(reversed(parse_cards("Ah 7c 2d")))


def test_hand_category_helpers():
    strength_value, _ = best_hand_strength(parse_cards("Ah 7c 2d"), parse_cards("As Kd"))
    assert hand_category(strength_value) == HandCategory.PAIR
    assert leading_rank(strength_value) == Rank.ACE

    assert made_hand_category(parse_cards("Ah 7c 2d")) == HandCategory.HIGH_CARD
    assert made_hand_category(parse_cards("7h 7c 2d")) == HandCategory.PAIR
    assert made_hand_category(parse_cards("7h 7c 2d 2s")) == HandCategory.TWO_PAIR
    assert made_hand_category(parse_cards("7h 7c 7d 2s")) == HandCategory.THREE_OF_A_KIND
    assert made_hand_category(parse_cards("5h 6c 7d 8s 9h")) == HandCategory.STRAIGHT


@pytest.mark.parametrize("hole, expected", [
    ("Ah Ad", STRONG), ("7h 7d", STRONG), ("6h 6d", MEDIUM), ("2h 2d", MEDIUM),
    ("Ah Kd", STRONG), ("Ah Qd", STRONG), ("Ah Jh", STRONG), ("Ah Jd", MEDIUM), ("Kh Qh", STRONG),
    ("Kh Qd", MEDIUM), ("Jh Td", MEDIUM), ("Th 9d", WEAK), ("Ah 2d", MEDIUM),
    ("6h 5h", MEDIUM), ("5h 4h", WEAK), ("7h 2d", WEAK),
])
def test_preflop_hand_class(hole, expected):
    assert preflop_hand_class(parse_cards(hole)) == expected


@pytest.mark.parametrize("hole, board, expected", [
    ("As Kd", "Ah 7c 2d", STRONG),          # top pair
    ("Ks Kd", "Qh 7c 2d", STRONG),          # overpair
    ("8s 8d", "Ah 7c 2d", MEDIUM),          # underpair
    ("7s Kd", "Ah 7c 2d", MEDIUM),          # middle pair
    ("Ks Qd", "Ah 7c 2d", WEAK),            # nothing
    ("7s 2s", "Ah 7c 2d", STRONG),          # two pair
    ("Ks Qd", "7h 7c 2d", WEAK),            # the board's pair plays
    ("Ks Kd", "7h 7c 2d", STRONG),          # two pair, kings up
    ("Ad Kc", "5h 6c 7d 8s 9h", WEAK),      # the board's straight plays
    # Known limitation: only categories are compared, so a higher straight on a straight board is WEAK
    ("Td Kc", "5h 6c 7d 8s 9h", WEAK),
])
def test_postflop_hand_class(hole, board, expected):
    assert postflop_hand_class(parse_cards(hole), parse_cards(board)) == expected


def preflop_state_with(hole, seat=0):
    """A fresh 3-player deal with `hole` dealt to `seat` and the dealer at seat 0 (first to act)."""
    holes = [None, None, None]
    holes[seat] = parse_cards(hole)
    return State(n_players=3, initial_wealth=20, initial_dealer=0, deck=stacked_deck(holes, rng=random.Random(0)))


def test_tag_agent_preflop_decisions():
    # Dealer (seat 0) is first to act preflop, facing the $2 big blind
    assert TagAgent(player_index=0).get_action(preflop_state_with("Ah Ad")) == 3
    assert TagAgent(player_index=0).get_action(preflop_state_with("Ah 9d")) == 2
    assert TagAgent(player_index=0).get_action(preflop_state_with("7h 2d")) == FOLD

    # Big blind (seat 2) with a weak hand checks when nobody raised
    state = preflop_state_with("7h 2d", seat=2)
    state.update(2)  # dealer calls
    state.update(1)  # small blind completes
    assert state.current_player == 2
    assert TagAgent(player_index=2).get_action(state) == CHECK


def test_tag_agent_calldown_variant_never_folds_after_the_flop():
    deck = stacked_deck([parse_cards("Ah Kd"), parse_cards("Qs Qc"), parse_cards("7h 2d")],
                        parse_cards("Ks Kc 9d 3s 4s"), rng=random.Random(0))
    state = State(n_players=3, initial_wealth=20, initial_dealer=0, deck=deck)
    for action in [2, 1, 0]:  # dealer calls, small blind completes, big blind checks
        state.update(action)
    assert state.game_stage == GameStage.FLOP
    state.update(3)  # small blind (seat 1) bets
    assert state.current_player == 2
    assert TagAgent(player_index=2).get_action(state) == FOLD
    assert TagAgent(player_index=2, fold_postflop=False).get_action(state) == 3


def test_baseline_agents_only_take_legal_actions():
    np.random.seed(0)
    random.seed(0)
    for seats in [
        [TagAgent(0), TagAgent(1, fold_postflop=False), RandomAgent(2)],
        [MaxRaiseAgent(0), TagAgent(1), CallStationAgent(2)],
    ]:
        for _ in range(150):
            state = State(n_players=3, initial_wealth=random.randint(5, 35), initial_dealer=random.randrange(3))
            while state.n_deals == 1 and not state.terminal:
                # Note: State.update asserts that the action is legal
                state.update(seats[state.current_player].get_action(state))
