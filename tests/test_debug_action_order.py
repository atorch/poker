"""
Debug test to trace exactly what happens with action order.
"""
from poker.state import State, GameStage


def test_trace_action_order():
    """Trace action order step by step to find the bug."""
    state = State(n_players=3, initial_wealth=100, initial_dealer=0, verbose=False)

    print(f"\n=== After initialization (dealer=0) ===")
    print(f"current_player: {state.current_player}")
    print(f"game_stage: {state.game_stage.name}")
    print(f"bets_by_stage[PRE_FLOP]: {state.bets_by_stage[GameStage.PRE_FLOP]}")

    # Player 0 calls big blind (2)
    print(f"\n=== Player {state.current_player} calls big blind (action=2) ===")
    state.update(2)
    print(f"After update: current_player={state.current_player}, stage={state.game_stage.name}")

    # Player 1 completes to big blind (1)
    print(f"\n=== Player {state.current_player} completes to big blind (action=1) ===")
    state.update(1)
    print(f"After update: current_player={state.current_player}, stage={state.game_stage.name}")

    # IMPORTANT: Stage should NOT transition yet - BB must act voluntarily
    assert state.game_stage == GameStage.PRE_FLOP, "Should still be at PRE_FLOP (BB needs to act)"
    assert state.current_player == 2, f"BB (player 2) should act next, got {state.current_player}"
    print(f"✓ CORRECT: BB gets option to act before stage transition")

    # Player 2 (BB) checks (0)
    print(f"\n=== Player {state.current_player} (BB) checks (action=0) ===")
    state.update(0)
    print(f"After update: current_player={state.current_player}, stage={state.game_stage.name}")

    # NOW the stage should transition to FLOP
    print(f"\n=== RESULT ===")
    print(f"After BB acted, stage transitioned to: {state.game_stage.name}")
    print(f"First to act on flop: player {state.current_player}")

    assert state.game_stage == GameStage.FLOP, "Should be at FLOP stage after BB acts"
    assert state.current_player == 1, f"First to act on flop should be player 1 (left of dealer), got {state.current_player}"
