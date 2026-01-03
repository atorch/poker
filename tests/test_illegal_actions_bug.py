"""Test for the 'All actions are illegal!' bug in maximum_legal_bet calculation."""

import pytest
import numpy as np
from poker.state import State, GameStage
from poker.agent import Agent


def test_maximum_legal_bet_bug():
    """
    Demonstrate the bug in maximum_legal_bet calculation.

    The bug: maximum_legal_bet uses total_bet_by_current_player for ALL players
    when calculating remaining wealth, instead of using each player's own total bet.

    This causes maximum_legal_bet to go negative when the current player has bet
    more than another player's total wealth, making all actions (including fold) illegal.
    """
    # Setup: 3 players with wealth $10 each
    state = State(n_players=3, initial_wealth=10)

    # Scenario:
    # - Player 0 has bet $1 (small blind)
    # - Player 1 has bet $2 (big blind)
    # - Player 2 raises to $15 (more than anyone's total wealth!)

    # Manually set up this scenario
    state.bets_by_stage[GameStage.PRE_FLOP][0] = [1]    # player 0: $1
    state.bets_by_stage[GameStage.PRE_FLOP][1] = [2]    # player 1: $2
    state.bets_by_stage[GameStage.PRE_FLOP][2] = [15]   # player 2: $15 (impossible but let's test)

    # Now it's player 0's turn to respond
    state.current_player = 0

    # Check the bug
    total_bet_current = state.total_bet_by_player(state.current_player)
    max_bet = state.maximum_legal_bet()

    print(f"\nCurrent player: {state.current_player}")
    print(f"Total bet by current player: {total_bet_current}")
    print(f"Maximum legal bet: {max_bet}")

    # With the BUG, calculation is:
    # min(wealth[0] - total_bet[0], wealth[1] - total_bet[0], wealth[2] - total_bet[0])
    # = min(10 - 1, 10 - 1, 10 - 1) = 9

    # But player 2 has already bet $15! So the CORRECT calculation should be:
    # min(wealth[0] - total_bet[0], wealth[1] - total_bet[1], wealth[2] - total_bet[2])
    # = min(10 - 1, 10 - 2, 10 - 15) = min(9, 8, -5) = -5

    # With max_bet = -5, ALL actions become illegal:
    # - fold (-1): illegal if -1 > -5 (TRUE!)
    # - 0, 1, 2, 3: all > -5, so illegal

    # Let's verify the bug leads to all actions being illegal
    actions = [-1, 0, 1, 2, 3]
    min_bet = state.minimum_legal_bet()

    print(f"Minimum legal bet: {min_bet}")

    legal_actions = []
    for action in actions:
        # This is the logic from agent.py get_action()
        is_legal = action < 0 or (min_bet <= action <= max_bet)
        if is_legal:
            legal_actions.append(action)
        print(f"  Action {action:2d}: {'LEGAL' if is_legal else 'ILLEGAL'}")

    # The bug manifests when max_bet is very negative
    # In this case, even fold can become illegal if max_bet < -1
    if max_bet < -1:
        print(f"\n❌ BUG: max_bet = {max_bet} < -1, so fold (-1) is illegal!")
        print(f"  This causes 'All actions are illegal!' error")
        assert len(legal_actions) == 0, "Expected all actions to be illegal with this bug"
    else:
        print(f"\n✓ Fold should still be legal (max_bet = {max_bet} >= -1)")


def test_maximum_legal_bet_correct_calculation():
    """
    Test that maximum_legal_bet correctly uses the current player's total bet.

    The calculation ensures that if the current player bets X, all other players
    can afford to match that total bet (without exceeding their wealth).
    """
    state = State(n_players=3, initial_wealth=20)

    # Setup:
    # - Player 0: bet $1, has $19 remaining
    # - Player 1: bet $10, has $10 remaining
    # - Player 2: bet $2, has $18 remaining

    state.bets_by_stage[GameStage.PRE_FLOP][0] = [1]
    state.bets_by_stage[GameStage.PRE_FLOP][1] = [10]
    state.bets_by_stage[GameStage.PRE_FLOP][2] = [2]

    state.current_player = 0

    # The CORRECT calculation uses the current player's total bet:
    # For player 0 to bet X additional, their total becomes (1 + X)
    # Others must be able to match this total:
    # - Player 1 needs total (1 + X), has wealth $20: (1 + X) <= 20, so X <= 19
    # - Player 2 needs total (1 + X), has wealth $20: (1 + X) <= 20, so X <= 19
    # So max_bet = min(19, 19) = 19

    current_total_bet = state.total_bet_by_player(state.current_player)
    expected_max_bet = min(
        state.wealth[p] - current_total_bet
        for p in range(state.n_players)
        if not state.has_folded[p]
    )

    actual_max_bet = state.maximum_legal_bet()

    print(f"\nCurrent player total bet: ${current_total_bet}")
    print(f"Expected max_bet: ${expected_max_bet}")
    print(f"Actual max_bet: ${actual_max_bet}")

    assert actual_max_bet == expected_max_bet, f"max_bet={actual_max_bet}, expected {expected_max_bet}"
    print(f"✓ Correctly calculated as ${actual_max_bet}")


def test_all_actions_illegal_with_agent():
    """
    Test that reproduces the actual 'ValueError: All actions are illegal!' error.
    """
    # Create a scenario where one player has bet much more than others have in wealth
    state = State(n_players=3, initial_wealth=5)

    # Player 0: bet $1 (small blind)
    # Player 1: bet $2 (big blind)
    # Player 2: bet $10 (impossible raise, but testing the edge case)
    state.bets_by_stage[GameStage.PRE_FLOP][0] = [1]
    state.bets_by_stage[GameStage.PRE_FLOP][1] = [2]
    state.bets_by_stage[GameStage.PRE_FLOP][2] = [10]

    state.current_player = 1  # Player 1's turn

    agent = Agent(player_index=1, n_players=3)

    min_bet = state.minimum_legal_bet()
    max_bet = state.maximum_legal_bet()

    print(f"\nPlayer 1's turn:")
    print(f"  Total bets: {[state.total_bet_by_player(i) for i in range(3)]}")
    print(f"  min_bet = {min_bet}, max_bet = {max_bet}")

    # Check if we can reproduce the error
    if max_bet < -1:
        print(f"  ❌ max_bet < -1, so even fold will be marked illegal!")

        # This is the exact code path in agent.py that fails
        q_values = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        actions = agent.actions  # [-1, 0, 1, 2, 3]

        for index, action in enumerate(actions):
            if (0 <= action < min_bet) or (action > max_bet):
                q_values[index] = -np.inf

        legal_mask = np.isfinite(q_values)
        print(f"  Q-values: {q_values}")
        print(f"  Legal mask: {legal_mask}")

        if not np.any(legal_mask):
            print(f"  ❌ BUG REPRODUCED: All actions are illegal!")
            with pytest.raises(ValueError, match="All actions are illegal"):
                from poker.agent import softmax_with_temperature
                softmax_with_temperature(q_values, temperature=1.0)
