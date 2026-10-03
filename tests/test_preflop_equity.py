"""
Tests for the preflop equity table (poker/data/preflop_equity.json) and hand-class helpers.
"""
import pytest

from poker.cards import parse_cards
from poker.preflop_equity import (
    HAND_CLASSES, class_combos, hand_class, load_table, preflop_equity, representative_hole_cards,
)


def test_hand_classes():
    assert len(HAND_CLASSES) == len(set(HAND_CLASSES)) == 169
    assert sum(class_combos(name) for name in HAND_CLASSES) == 1326
    assert hand_class(parse_cards("Kd As")) == "AKo"
    assert hand_class(parse_cards("2h 7h")) == "72s"
    assert hand_class(parse_cards("Tc Td")) == "TT"
    for name in HAND_CLASSES:
        assert hand_class(representative_hole_cards(name)) == name


def test_table_covers_every_class():
    table = load_table()
    assert set(table["equity"]) == {"1", "2"}
    for n_opponents in ["1", "2"]:
        assert set(table["equity"][n_opponents]) == set(HAND_CLASSES)


@pytest.mark.parametrize("n_opponents, fair_share", [(1, 1 / 2), (2, 1 / 3)])
def test_average_equity_is_the_fair_share(n_opponents, fair_share):
    """Weighted by combinations, a random hand's equity is exactly 1/(n_opponents + 1)."""
    total = sum(class_combos(name) * preflop_equity(name, n_opponents) for name in HAND_CLASSES) / 1326
    assert total == pytest.approx(fair_share, abs=0.003)


@pytest.mark.parametrize("name, expected", [
    # Published heads-up equities against a random hand
    ("AA", 0.852), ("KK", 0.824), ("AKs", 0.670), ("AKo", 0.653), ("22", 0.503), ("72o", 0.346), ("32o", 0.323),
])
def test_known_heads_up_equities(name, expected):
    assert preflop_equity(name, n_opponents=1) == pytest.approx(expected, abs=0.01)


def test_ordering():
    for n_opponents in [1, 2]:
        equities = {name: preflop_equity(name, n_opponents) for name in HAND_CLASSES}
        assert max(equities, key=equities.get) == "AA"
        assert equities["AKs"] > equities["AKo"]
        assert preflop_equity(parse_cards("As Kh"), n_opponents) == equities["AKo"]
