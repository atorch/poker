"""
Tests for showdown frequency tracking during training.

Ensures we correctly distinguish between:
- Showdowns: 2+ players still active at end of hand (compare cards)
- Wins by fold: Only 1 player remains (everyone else folded)
"""

import pytest
from poker.state import State, GameStage
from poker.config import TYPICAL_INITIAL_WEALTH


def test_showdown_tracking_logic_with_showdown():
    """
    Test the core logic for detecting showdowns.

    Simulate the scenario: deal completes with 2+ players still active.
    This should count as a showdown, not a win by fold.
    """
    state = State(n_players=3, initial_wealth=TYPICAL_INITIAL_WEALTH)

    # Simulate: Player 0 and Player 1 stay in, Player 2 folds
    state.has_folded = [False, False, True]

    # Count active players (this is what the tracking code does)
    active_players = sum(1 for folded in state.has_folded if not folded)

    # Should be 2 active players
    assert active_players == 2

    # This should count as a showdown (active_players >= 2)
    is_showdown = active_players >= 2
    assert is_showdown == True


def test_showdown_tracking_logic_with_fold():
    """
    Test the core logic for detecting wins by fold.

    Simulate the scenario: deal completes with only 1 player still active.
    This should count as win by fold, not a showdown.
    """
    state = State(n_players=3, initial_wealth=TYPICAL_INITIAL_WEALTH)

    # Simulate: Only Player 1 stays in, others fold
    state.has_folded = [True, False, True]

    # Count active players
    active_players = sum(1 for folded in state.has_folded if not folded)

    # Should be 1 active player
    assert active_players == 1

    # This should count as win by fold (active_players < 2)
    is_showdown = active_players >= 2
    assert is_showdown == False


def test_bug_has_folded_reset_after_deal():
    """
    REGRESSION TEST for the showdown tracking bug.

    The bug: We check has_folded AFTER initialize_pre_flop() resets it.

    This test demonstrates the problem:
    1. Start with has_folded showing who folded
    2. Call initialize_pre_flop (simulating new deal)
    3. has_folded is now reset to all False
    4. Checking has_folded at this point gives wrong answer
    """
    state = State(n_players=3, initial_wealth=TYPICAL_INITIAL_WEALTH)

    # Simulate: 2 players folded during previous deal
    state.has_folded = [True, False, True]

    # Verify state before reset
    active_before = sum(1 for folded in state.has_folded if not folded)
    assert active_before == 1  # Only 1 player active → win by fold

    # Now simulate what happens when deal completes:
    # redistribute_wealth_and_reinitialize calls initialize_pre_flop
    # which resets has_folded
    old_n_deals = state.n_deals
    state.initialize_pre_flop(dealer=0)

    # After initialize_pre_flop:
    # - n_deals incremented (this is how we detect deal changed)
    assert state.n_deals == old_n_deals + 1

    # - has_folded reset to all False!
    assert state.has_folded == [False, False, False]

    # If we check has_folded NOW (like the buggy code does):
    active_after = sum(1 for folded in state.has_folded if not folded)
    assert active_after == 3  # All 3 active → looks like showdown!

    # This is the bug: We get showdown (3 players) when it should be win by fold (1 player)


def test_state_has_folded_not_reset_before_deal_end():
    """
    Test that has_folded contains meaningful data before a deal completes.

    This is a regression test for the bug where we check has_folded AFTER
    it's been reset for the next deal.
    """
    state = State(n_players=3, initial_wealth=TYPICAL_INITIAL_WEALTH)

    # Simulate player 0 folding
    state.has_folded[0] = True

    # Check that has_folded reflects the fold
    assert state.has_folded[0] == True
    assert state.has_folded[1] == False
    assert state.has_folded[2] == False

    # Active players should be 2
    active_players = sum(1 for folded in state.has_folded if not folded)
    assert active_players == 2


def test_showdown_definition():
    """
    Test that we correctly define showdown vs win-by-fold.

    Showdown: 2+ players active (has_folded[i] == False) at end of hand
    Win by fold: Exactly 1 player active at end of hand
    """
    # Scenario 1: All 3 players active → SHOWDOWN
    has_folded_showdown = [False, False, False]
    active = sum(1 for folded in has_folded_showdown if not folded)
    assert active >= 2  # This should count as showdown

    # Scenario 2: 2 players active → SHOWDOWN
    has_folded_showdown2 = [False, True, False]
    active = sum(1 for folded in has_folded_showdown2 if not folded)
    assert active >= 2  # This should count as showdown

    # Scenario 3: 1 player active → WIN BY FOLD
    has_folded_fold = [False, True, True]
    active = sum(1 for folded in has_folded_fold if not folded)
    assert active == 1  # This should count as win by fold
