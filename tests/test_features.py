"""
Tests for poker.features: layout, value ranges, invariances and a few hand-checked values.
"""
import random

import numpy as np
import pytest

from poker.cards import parse_cards, stacked_deck
from poker.features import FEATURE_NAMES, N_FEATURES, OFFSETS, OPPONENT_FIELDS, describe, encode
from poker.preflop_equity import preflop_equity
from poker.random_agent import RandomAgent
from poker.state import State


def opponent_block(features, i):
    base = OFFSETS[f"opponent{i}_present"]
    return dict(zip(OPPONENT_FIELDS, features[base:base + len(OPPONENT_FIELDS)].tolist()))


def test_layout():
    assert N_FEATURES == len(FEATURE_NAMES) == 172
    assert len(set(FEATURE_NAMES)) == N_FEATURES


def test_first_to_act_preflop():
    deck = stacked_deck([parse_cards("Ah Ad"), None, None], rng=random.Random(0))
    state = State(n_players=3, initial_wealth=20, initial_dealer=0, deck=deck)
    named = describe(encode(state, seat=0))
    assert named["stage_pre_flop"] == 1 and named["position_0"] == 1 and named["made_pair"] == 1
    assert named["owed"] == pytest.approx(0.1)        # $2 = 1 BB, over MONEY_SCALE = 10
    assert named["pot"] == pytest.approx(0.15)        # $3
    assert named["pot_odds"] == pytest.approx(0.4)    # 2 / (3 + 2)
    assert named["stack_behind"] == pytest.approx(1.0)
    assert named["preflop_equity"] == pytest.approx(preflop_equity("AA", 2))
    assert "committed_deal" not in named               # the dealer hasn't put anything in
    assert named["n_players"] == pytest.approx(0.5) and named["n_active"] == pytest.approx(0.5)


def test_relative_seat_order_and_padding():
    state = State(n_players=3, initial_wealth=[5, 20, 35], initial_dealer=0)
    features = encode(state, seat=1)  # the small blind
    first, second = opponent_block(features, 1), opponent_block(features, 2)
    assert first["stack_behind"] == pytest.approx((35 - 2) / 20)   # seat 2, the big blind
    assert first["committed_deal"] == pytest.approx(2 / 20)
    assert second["stack_behind"] == pytest.approx(5 / 20)         # seat 0, the dealer
    assert first["present"] == second["present"] == 1
    for i in range(3, 6):
        assert all(value == 0 for value in opponent_block(features, i).values())
    assert describe(features)["effective_behind"] == pytest.approx(5 / 20)


def test_hole_card_order_does_not_matter():
    state = State(n_players=3, initial_wealth=20, initial_dealer=0)
    before = encode(state, seat=0)
    state.hole_cards[0] = list(reversed(state.hole_cards[0]))
    assert np.array_equal(before, encode(state, seat=0))


def test_raises_and_board_pair():
    deck = stacked_deck([parse_cards("Kh Qd"), None, None], parse_cards("7h 7c 2d"), rng=random.Random(0))
    state = State(n_players=3, initial_wealth=20, initial_dealer=0, deck=deck)
    state.update(3)  # dealer raises
    state.update(3)  # small blind re-raises
    state.update(2)  # big blind calls
    state.update(1)  # dealer calls: on to the flop
    state.update(0)  # small blind checks
    state.update(3)  # big blind bets
    named = describe(encode(state, seat=0))
    assert named["stage_flop"] == 1
    assert named["made_pair"] == 1 and "improves_board" not in named   # the board's pair plays
    assert named["owed"] == pytest.approx(3 / 20)
    sb, bb = opponent_block(encode(state, seat=0), 1), opponent_block(encode(state, seat=0), 2)
    assert sb["raises_deal"] == pytest.approx(1 / 4) and sb["raises_street"] == 0
    assert bb["raises_street"] == pytest.approx(1 / 4)
    assert named["raises_street"] == pytest.approx(1 / 4)


def test_values_stay_in_range_in_random_games():
    random.seed(0)
    np.random.seed(0)
    agents = [RandomAgent(player_index=i) for i in range(3)]
    for _ in range(200):
        state = State(n_players=3, initial_wealth=[random.randint(5, 35) for _ in range(3)],
                      initial_dealer=random.randrange(3))
        while state.n_deals == 1 and not state.terminal:
            features = encode(state, state.current_player)
            assert features.dtype == np.float32 and np.all(np.isfinite(features))
            assert features.min() >= 0 and features.max() <= 6
            assert features[OFFSETS["hole"]:OFFSETS["hole"] + 52].sum() == 2
            state.update(agents[state.current_player].get_action(state))
