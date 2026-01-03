"""
Simple test to debug stage transition and current_player.
"""
from poker.state import State, GameStage


def test_simple_stage_transition():
    """Test that after pre-flop completes, player left of dealer acts first."""
    state = State(n_players=3, initial_wealth=100, initial_dealer=0, verbose=True)

    print(f"\n=== After initialization ===")
    print(f"Dealer: {state.dealer}")
    print(f"Current player: {state.current_player}")
    print(f"Game stage: {state.game_stage.name}")
    print(f"Bets: {state.bets_by_stage[state.game_stage]}")

    # Player 0 calls BB
    print(f"\n=== Player {state.current_player} calls BB (action=2) ===")
    state.update(2)
    print(f"After update: current_player={state.current_player}, stage={state.game_stage.name}")

    # Player 1 completes to BB
    print(f"\n=== Player {state.current_player} completes to BB (action=1) ===")
    state.update(1)
    print(f"After update: current_player={state.current_player}, stage={state.game_stage.name}")

    # Player 2 checks
    print(f"\n=== Player {state.current_player} checks (action=0) ===")
    print(f"Before update: current_player={state.current_player}, stage={state.game_stage.name}")
    state.update(0)
    print(f"After update: current_player={state.current_player}, stage={state.game_stage.name}")

    print(f"\n=== RESULT ===")
    print(f"Game stage: {state.game_stage.name}")
    print(f"Current player: {state.current_player}")
    print(f"Expected: 1 (player left of dealer)")

    assert state.game_stage == GameStage.FLOP, f"Should be at FLOP, got {state.game_stage.name}"
    assert state.current_player == 1, f"After pre-flop, current_player should be 1 (left of dealer), got {state.current_player}"
