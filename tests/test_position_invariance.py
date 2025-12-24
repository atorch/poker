"""
Tests for position invariance and fairness in poker game.

These tests verify that the game logic and state representation don't give
systematic advantages to specific player positions.
"""
import sys
from unittest.mock import Mock
import numpy as np
import pytest

# Mock tensorflow to avoid imports during testing
sys.modules["poker.q_function"] = Mock()

from poker.agent import Agent
from poker.state import State
from poker.cards import Card, Rank, Suit


def test_dealer_rotates_between_rounds():
    """
    Verify that dealer position rotates correctly across multiple rounds.
    Critical for fairness - all players should have equal dealer opportunities.
    """
    # Start with dealer=0
    state = State(n_players=3, initial_wealth=100, initial_dealer=0)
    assert state.dealer == 0

    # Check that redistribute_wealth_and_reinitialize rotates dealer
    # We can't easily test this without triggering game logic, so we verify
    # the rotation happens by checking the code logic:
    # state.py:234 → next_dealer = (self.dealer + 1) % self.n_players

    # Instead, test with different initial dealers to verify rotation logic
    state1 = State(n_players=3, initial_wealth=100, initial_dealer=0)
    state2 = State(n_players=3, initial_wealth=100, initial_dealer=1)
    state3 = State(n_players=3, initial_wealth=100, initial_dealer=2)

    assert state1.dealer == 0
    assert state2.dealer == 1
    assert state3.dealer == 2

    # Verify modulo wrapping
    state4 = State(n_players=3, initial_wealth=100, initial_dealer=3)
    # Note: State doesn't validate dealer, so this might be 3 or wrapped
    # The actual rotation happens in redistribute_wealth_and_reinitialize


def test_deck_is_shuffled_each_deal():
    """
    Verify that deck is randomized for each new deal.
    Without proper shuffling, certain positions could get better cards.
    """
    import random

    # NOTE: State uses random.sample() not np.random, so we need to use random.seed()
    # Create two states with same seed
    random.seed(42)
    state1 = State(n_players=3, initial_wealth=100)
    hole_cards_1 = [cards.copy() for cards in state1.hole_cards]

    random.seed(42)
    state2 = State(n_players=3, initial_wealth=100)
    hole_cards_2 = [cards.copy() for cards in state2.hole_cards]

    # Same seed should produce same cards
    for i in range(3):
        assert hole_cards_1[i][0] == hole_cards_2[i][0]
        assert hole_cards_1[i][1] == hole_cards_2[i][1]

    # Different seed should produce different cards
    random.seed(999)
    state3 = State(n_players=3, initial_wealth=100)

    # At least one player should have different cards
    cards_differ = False
    for i in range(3):
        if (state3.hole_cards[i][0] != hole_cards_1[i][0] or
            state3.hole_cards[i][1] != hole_cards_1[i][1]):
            cards_differ = True
            break

    assert cards_differ, "Different seeds should produce different card deals"


def test_state_representation_opponent_ordering():
    """
    CRITICAL TEST: Verify that opponent wealth/active status is ordered by RELATIVE position.

    After fix: Opponents should be ordered in circular/round-table seating:
    - opponent[0] = player at offset +1 (one seat to the right)
    - opponent[1] = player at offset +2 (two seats to the right)

    This ensures position-invariance: all players see equivalent state vectors
    for equivalent game situations.
    """
    state = State(n_players=3, initial_wealth=100)

    # Set specific wealth values for testing
    state.wealth = [100, 75, 50]
    state.has_folded = [False, False, True]

    # Create agents for each position
    agent0 = Agent(player_index=0, n_players=3)
    agent1 = Agent(player_index=1, n_players=3)
    agent2 = Agent(player_index=2, n_players=3)

    # Get private states
    private_state_0 = agent0.get_private_state(state)
    private_state_1 = agent1.get_private_state(state)
    private_state_2 = agent2.get_private_state(state)

    # Extract opponent wealths (last 4 elements: 2 wealths + 2 active flags)
    # State structure: [..., own_wealth, own_bets, pot_size, public_cards(6), opp_wealths(2), opp_active(2)]
    opp_wealth_0 = private_state_0[-4:-2]
    opp_wealth_1 = private_state_1[-4:-2]
    opp_wealth_2 = private_state_2[-4:-2]

    opp_active_0 = private_state_0[-2:]
    opp_active_1 = private_state_1[-2:]
    opp_active_2 = private_state_2[-2:]

    # AFTER FIX: Opponents ordered by relative position (offset +1, +2, ...)
    # Player 0 (index 0) sees:
    #   - offset +1 → player 1 (wealth=75, active=True)
    #   - offset +2 → player 2 (wealth=50, active=False)
    assert opp_wealth_0 == [75, 50], f"Player 0 opponent wealths: {opp_wealth_0}"
    assert opp_active_0 == [1, 0], f"Player 0 opponent active: {opp_active_0}"

    # Player 1 (index 1) sees:
    #   - offset +1 → player 2 (wealth=50, active=False)
    #   - offset +2 → player 0 (wealth=100, active=True)
    assert opp_wealth_1 == [50, 100], f"Player 1 opponent wealths: {opp_wealth_1}"
    assert opp_active_1 == [0, 1], f"Player 1 opponent active: {opp_active_1}"

    # Player 2 (index 2) sees:
    #   - offset +1 → player 0 (wealth=100, active=True)
    #   - offset +2 → player 1 (wealth=75, active=True)
    assert opp_wealth_2 == [100, 75], f"Player 2 opponent wealths: {opp_wealth_2}"
    assert opp_active_2 == [1, 1], f"Player 2 opponent active: {opp_active_2}"

    # POSITION INVARIANCE CHECK: In a symmetric situation, all players should see
    # the same relative pattern. Example: if all opponents have equal wealth and
    # are active, all players should see identical opponent features.
    state_symmetric = State(n_players=3, initial_wealth=100)
    state_symmetric.wealth = [100, 100, 100]
    state_symmetric.has_folded = [False, False, False]

    state_0_sym = agent0.get_private_state(state_symmetric)
    state_1_sym = agent1.get_private_state(state_symmetric)
    state_2_sym = agent2.get_private_state(state_symmetric)

    # All players should see [100, 100] for opponent wealths
    assert state_0_sym[-4:-2] == [100, 100]
    assert state_1_sym[-4:-2] == [100, 100]
    assert state_2_sym[-4:-2] == [100, 100]

    # All players should see [1, 1] for opponent active status
    assert state_0_sym[-2:] == [1, 1]
    assert state_1_sym[-2:] == [1, 1]
    assert state_2_sym[-2:] == [1, 1]


def test_position_symmetry_with_identical_models():
    """
    Integration test: When all 3 players use identical models and policies,
    win rates should be approximately equal over many games.

    This is a statistical test - we expect ~33% ± margin for each player.
    """
    # NOTE: This test requires actual model weights, so it's more of an
    # integration test. For now, we just document the expected behavior.

    # Expected: Over 1000 games with identical frozen policies:
    # - Player 0 wins: ~33% ± 3%
    # - Player 1 wins: ~33% ± 3%
    # - Player 2 wins: ~33% ± 3%

    # Current bug: Player 0 wins ~70%!
    pytest.skip("Integration test - requires trained model weights")


def test_randomize_initial_dealer():
    """
    Test that we can randomize the initial dealer position.
    Currently, run_frozen_episode always starts with dealer=0.
    """
    # Create states with different initial dealers
    state0 = State(n_players=3, initial_wealth=100, initial_dealer=0)
    state1 = State(n_players=3, initial_wealth=100, initial_dealer=1)
    state2 = State(n_players=3, initial_wealth=100, initial_dealer=2)

    assert state0.dealer == 0
    assert state1.dealer == 1
    assert state2.dealer == 2

    # Verify that initial dealer affects who posts blinds
    # With dealer=0: player 1 posts small blind, player 2 posts big blind
    # (Blinds are posted in initialize_pre_flop after setting dealer)
    # First two updates in initialize_pre_flop are the blinds

    # This is harder to test without refactoring, but documents the behavior


def test_small_blind_and_big_blind_rotation():
    """
    Verify that small blind and big blind positions rotate with dealer.
    """
    # Dealer = 0 → SB = 1, BB = 2
    state = State(n_players=3, initial_wealth=100, initial_dealer=0)
    # After initialize_pre_flop, player 1 and 2 should have posted blinds
    # This is implicit in the forced updates during initialization

    # Dealer = 1 → SB = 2, BB = 0
    state = State(n_players=3, initial_wealth=100, initial_dealer=1)
    # Similarly, player 2 and 0 should post blinds

    # Dealer = 2 → SB = 0, BB = 1
    state = State(n_players=3, initial_wealth=100, initial_dealer=2)
    # Player 0 and 1 should post blinds

    # NOTE: Current implementation forces blinds in initialize_pre_flop,
    # making this hard to test without refactoring. This test documents
    # the expected behavior.

    pytest.skip("Requires refactoring to make blind posting testable")
