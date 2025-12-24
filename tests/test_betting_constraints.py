"""
Test betting constraints: wealthy players cannot bet more than poor players can afford to match.
"""

import pytest
from poker.state import State, GameStage


def test_wealthy_vs_poor_two_players():
    """
    With 2 players, the wealthy player should NOT be able to bet more than
    the poor player's total wealth, since that would make calling impossible.
    """
    print("\n" + "="*70)
    print("TEST: 2 players, one wealthy, one poor")
    print("="*70)

    # 2 players: one wealthy ($100), one poor ($10)
    state = State(n_players=2, initial_wealth=100)

    # Manually set player 1's wealth to $10 to simulate a poor player
    state.wealth[1] = 10

    # Post blinds
    # Player 0 (small blind): bet $1
    # Player 1 (big blind): bet $2
    state.bets_by_stage[GameStage.PRE_FLOP][0] = [1]
    state.bets_by_stage[GameStage.PRE_FLOP][1] = [2]

    # Player 0's turn (the wealthy player)
    state.current_player = 0

    max_bet = state.maximum_legal_bet()

    print(f"Player 0 (wealthy): wealth=${state.wealth[0]}, bet=${state.total_bet_by_player(0)}")
    print(f"Player 1 (poor):    wealth=${state.wealth[1]}, bet=${state.total_bet_by_player(1)}")
    print(f"\nPlayer 0's maximum_legal_bet: ${max_bet}")

    # The wealthy player should only be able to bet up to the poor player's wealth
    # Poor player has $10 total, so wealthy player can bet up to $10 total
    # Wealthy player already bet $1, so can bet $9 more
    # This forces poor player all-in ($10 total) but doesn't exceed their wealth

    expected_max = state.wealth[1] - state.total_bet_by_player(0)
    print(f"Expected max_bet: ${expected_max}")
    print(f"  Calculation: poor_player_wealth - wealthy_player_total_bet")
    print(f"             = ${state.wealth[1]} - ${state.total_bet_by_player(0)}")
    print(f"             = ${expected_max}")

    if max_bet == expected_max:
        print(f"\n✓ Correct! Wealthy player can bet up to ${max_bet}")
        print(f"  This would make wealthy player's total: ${state.total_bet_by_player(0) + max_bet}")
        print(f"  Poor player could call by going all-in: ${state.wealth[1]}")
    else:
        print(f"\n❌ Wrong! max_bet=${max_bet}, expected ${expected_max}")
        pytest.fail(f"maximum_legal_bet incorrect: {max_bet} != {expected_max}")


def test_three_players_one_poor():
    """
    With 3 players (2 wealthy, 1 poor), the maximum bet should be constrained
    by the poor player's ability to call.
    """
    print("\n" + "="*70)
    print("TEST: 3 players, 2 wealthy, 1 poor")
    print("="*70)

    state = State(n_players=3, initial_wealth=100)

    # Player 2 is poor
    state.wealth[2] = 10

    # Post blinds
    state.bets_by_stage[GameStage.PRE_FLOP][0] = [1]   # small blind
    state.bets_by_stage[GameStage.PRE_FLOP][1] = [2]   # big blind
    state.bets_by_stage[GameStage.PRE_FLOP][2] = [0]   # hasn't acted yet

    # Player 2 (poor player) acts first
    state.current_player = 2

    max_bet_poor = state.maximum_legal_bet()
    print(f"Player 2 (poor) max_bet: ${max_bet_poor}")
    print(f"  Can bet up to: ${state.wealth[2]} (their total wealth)")

    # Now player 0 (wealthy) acts
    state.current_player = 0
    max_bet_wealthy = state.maximum_legal_bet()

    print(f"\nPlayer 0 (wealthy) max_bet: ${max_bet_wealthy}")

    # Wealthy player should be constrained by poor player's wealth
    # Poor player has $10, so wealthy player can bet up to $10 total
    # Wealthy player already bet $1, so can bet $9 more?
    # No wait - let's think about this carefully

    print(f"\nWealth and bets:")
    for i in range(3):
        total_bet = state.total_bet_by_player(i)
        remaining = state.wealth[i] - total_bet
        print(f"  Player {i}: wealth=${state.wealth[i]}, bet=${total_bet}, remaining=${remaining}")

    # For player 0 to bet $X total:
    # - Player 1 must be able to match: $X <= wealth[1] = $100 ✓
    # - Player 2 must be able to match: $X <= wealth[2] = $10
    # So player 0 can bet up to $10 total, which is $9 additional (since already bet $1)

    expected_max = min(
        state.wealth[1] - state.total_bet_by_player(0),  # constraint from player 1
        state.wealth[2] - state.total_bet_by_player(0),  # constraint from player 2
    )

    print(f"\nExpected max_bet for player 0: ${expected_max}")
    print(f"  Player 1 constraint: ${state.wealth[1]} - ${state.total_bet_by_player(0)} = ${state.wealth[1] - state.total_bet_by_player(0)}")
    print(f"  Player 2 constraint: ${state.wealth[2]} - ${state.total_bet_by_player(0)} = ${state.wealth[2] - state.total_bet_by_player(0)}")

    if max_bet_wealthy == expected_max:
        print(f"\n✓ Correct! Wealthy player can bet up to ${max_bet_wealthy}")
    else:
        print(f"\n❌ Wrong! max_bet={max_bet_wealthy}, expected {expected_max}")
        pytest.fail(f"maximum_legal_bet incorrect: {max_bet_wealthy} != {expected_max}")


def test_poor_player_can_always_call_or_fold():
    """
    After a wealthy player bets, the poor player should always have at least
    2 legal actions: fold OR call (by going all-in if necessary).

    They should NEVER be in a situation where fold is the only legal action.
    """
    print("\n" + "="*70)
    print("TEST: Poor player can always call or fold")
    print("="*70)

    state = State(n_players=2, initial_wealth=100)
    state.wealth[1] = 10  # Player 1 is poor

    # Post blinds
    state.bets_by_stage[GameStage.PRE_FLOP][0] = [1]
    state.bets_by_stage[GameStage.PRE_FLOP][1] = [2]

    # Player 0 (wealthy) bets the maximum allowed
    state.current_player = 0
    max_bet_p0 = state.maximum_legal_bet()

    print(f"Player 0 bets ${max_bet_p0} (the maximum allowed)")

    # Simulate the bet
    state.bets_by_stage[GameStage.PRE_FLOP][0].append(max_bet_p0)

    # Now it's player 1's turn
    state.current_player = 1
    min_bet_p1 = state.minimum_legal_bet()
    max_bet_p1 = state.maximum_legal_bet()

    print(f"\nPlayer 1's turn:")
    print(f"  Wealth: ${state.wealth[1]}")
    print(f"  Total bet so far: ${state.total_bet_by_player(1)}")
    print(f"  min_bet: ${min_bet_p1} (to call)")
    print(f"  max_bet: ${max_bet_p1}")

    # Check if player 1 can call
    can_call = min_bet_p1 <= max_bet_p1

    print(f"\n  Can player 1 call? {can_call}")

    if can_call:
        print(f"  ✓ Player 1 has 2 legal actions: fold OR call (bet ${min_bet_p1})")
    else:
        print(f"  ❌ PROBLEM: Player 1 cannot call! min_bet > max_bet")
        print(f"  This means fold is the ONLY legal action")
        print(f"  This violates the game design - poor player should always be able to call!")
        pytest.fail("Poor player cannot call - fold is only legal action!")

    # Verify player 1 can afford to call
    total_bet_after_call = state.total_bet_by_player(1) + min_bet_p1
    print(f"\n  If player 1 calls:")
    print(f"    Total bet: ${total_bet_after_call}")
    print(f"    Wealth: ${state.wealth[1]}")
    print(f"    Legal? {total_bet_after_call <= state.wealth[1]}")

    assert total_bet_after_call <= state.wealth[1], "Calling would exceed player 1's wealth!"


def test_all_players_equal_wealth():
    """
    Baseline: when all players have equal wealth, the betting should work normally.
    """
    print("\n" + "="*70)
    print("TEST: All players equal wealth (baseline)")
    print("="*70)

    state = State(n_players=3, initial_wealth=20)

    print(f"Wealth: {state.wealth}")
    print(f"Total bets after blinds: {[state.total_bet_by_player(i) for i in range(3)]}")

    # Player 2 acts first (UTG)
    state.current_player = 2
    max_bet = state.maximum_legal_bet()

    print(f"\nPlayer 2's max_bet: ${max_bet}")

    # With equal wealth, max_bet should be constrained by the player with most bets
    # Player 0: bet $0, remaining $20
    # Player 1: bet $1, remaining $19
    # Player 2: bet $2, remaining $18

    # For player 2 to bet $X total, others must match:
    # - Player 0 needs $X, has $20 ✓
    # - Player 1 needs $X, has $20 ✓
    # So player 2 can bet up to $20 total, which is $18 additional

    expected = min(
        state.wealth[0] - state.total_bet_by_player(2),
        state.wealth[1] - state.total_bet_by_player(2),
        state.wealth[2] - state.total_bet_by_player(2),
    )

    print(f"Expected: ${expected}")

    assert max_bet == expected, f"max_bet={max_bet}, expected {expected}"
    print(f"✓ Correct!")
