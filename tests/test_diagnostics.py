"""
Tests for poker.diagnostics: rank correlation, preflop profiles and the flop reference spots.
"""
import numpy as np
import pytest

from poker.benchmark import SeatedAgent, agent_factory
from poker.diagnostics import REFERENCE_SPOTS, equity_groups, fingerprint, format_fingerprint, spearman


def test_equity_groups():
    equity = [0.9, 0.7, 0.5, 0.3, 0.1]
    weights = np.full(5, 0.2)
    top, bottom = equity_groups(equity, weights, top_fraction=0.2, bottom_fraction=0.4)
    assert top.tolist() == [True, False, False, False, False]
    assert bottom.tolist() == [False, False, False, True, True]


def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1)
    assert spearman([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(1)  # ties
    assert spearman([1, 1, 1], [1, 2, 3]) is None


def test_tag_agent_fingerprint():
    fp = fingerprint(agent_factory("tag"))
    preflop = fp["preflop"]
    assert preflop["by_class"]["AA"] == [0, 0, 1]
    assert preflop["by_class"]["72o"] == [1, 0, 0]
    # Note: raising is binary for TagAgent (only its ~7% strongest hands), which caps the rank correlation
    assert preflop["raise_vs_equity"] > 0.4
    assert preflop["fold_vs_equity"] < -0.6
    # TagAgent raises 77+, AQ+, AJs, KQs (part of the top 20% of combos) and none of the bottom 50%,
    #  and folds most of the bottom 50% (small pairs and suited connectors are "medium")
    assert 0.3 < preflop["raise_spread"] < 1
    assert 0.8 < preflop["fold_spread"] < 1

    spots = fp["spots"]
    assert spots["top pair, first to act"]["raise"] == 1
    assert spots["nothing, first to act"]["call"] == 1  # checks
    assert spots["top pair, facing $2 bet"]["raise"] == 1
    assert spots["nothing, facing $2 bet"]["fold"] == 1
    assert len(format_fingerprint(fp)) <= 6


def test_card_blind_fingerprint_has_no_correlation():
    fp = fingerprint(agent_factory("maxraise"))
    assert fp["preflop"]["raise_vs_equity"] is None and fp["preflop"]["p_raise"] == pytest.approx(1)
    assert fp["preflop"]["raise_spread"] == pytest.approx(0) and fp["preflop"]["fold_spread"] == pytest.approx(0)
    assert "n/a" in format_fingerprint(fp)[0]


def test_sampled_agents():
    fp = fingerprint(agent_factory("random"), n_samples=200)
    # Uniform over fold, call $2 and raise when first to act preflop
    assert fp["preflop"]["p_fold"] == pytest.approx(1 / 3, abs=0.02)


class FixedPolicyAgent:
    """Exposes its policy, so the fingerprint should read it exactly instead of sampling."""

    def __init__(self, player_index=0):
        self.player_index = player_index

    def action_probabilities(self, game_state):
        legal = game_state.legal_actions()
        return {action: 1 / len(legal) for action in legal}

    def get_action(self, game_state, proba_random_action=0.0):
        raise AssertionError("should not be sampled")


def test_exposed_policies_are_read_exactly():
    fp = fingerprint(lambda seat: SeatedAgent(FixedPolicyAgent(seat), seat))
    assert fp["preflop"]["p_fold"] == pytest.approx(1 / 3)
    assert set(fp["spots"]) == {spot["name"] for spot in REFERENCE_SPOTS}
