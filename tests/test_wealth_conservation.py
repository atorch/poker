"""
Test wealth conservation across extended gameplay scenarios.

In this poker implementation, wealth updates happen at deal boundaries:
- During a deal: state.wealth represents wealth at deal start
- Bets are tracked separately in state.bets_by_stage
- After a deal completes: pot is distributed and state.wealth is updated

Basic wealth conservation (after single deals, folds, ties) is already covered
in test_state.py. This file extends that coverage to multi-deal sequences and
full games to terminal state.
"""
from poker.state import State, GameStage
from poker.random_agent import RandomAgent


def test_wealth_conservation_across_multiple_deals():
    """
    Test that wealth remains conserved across multiple consecutive deals.

    This extends the single-deal coverage in test_state.py to verify that
    wealth conservation holds over longer sequences of gameplay.
    """
    initial_wealth_per_player = 25
    n_players = 3
    expected_total = initial_wealth_per_player * n_players

    state = State(n_players=n_players, initial_wealth=initial_wealth_per_player)
    actions = [-1, 0, 1, 2]
    agents = [RandomAgent(player_index=i, actions=actions) for i in range(n_players)]

    max_deals = 10
    deals_completed = 0

    while not state.terminal and deals_completed < max_deals:
        deal_start_stage = state.game_stage

        # Play through one deal
        max_actions = 50
        for _ in range(max_actions):
            if state.terminal:
                break

            # Check if we completed a deal (returned to PRE_FLOP or terminal)
            if state.game_stage == GameStage.PRE_FLOP and deal_start_stage != GameStage.PRE_FLOP:
                break

            agent = agents[state.current_player]
            action = agent.get_action(state)
            state.update(action)

        deals_completed += 1

        # After each deal, wealth should be conserved
        assert abs(sum(state.wealth) - expected_total) < 0.01, \
            f"Wealth not conserved after deal {deals_completed}: {sum(state.wealth)} != {expected_total}"


def test_wealth_conservation_until_terminal_state():
    """
    Test wealth conservation over a complete game until terminal state.

    This verifies that wealth conservation holds through the full lifecycle
    of a game, including all intermediate deals and the final elimination.
    """
    initial_wealth_per_player = 20
    n_players = 3
    expected_total = initial_wealth_per_player * n_players

    state = State(n_players=n_players, initial_wealth=initial_wealth_per_player)
    actions = [-1, 0, 1, 2]
    agents = [RandomAgent(player_index=i, actions=actions) for i in range(n_players)]

    max_actions = 500
    action_count = 0

    while not state.terminal and action_count < max_actions:
        agent = agents[state.current_player]
        action = agent.get_action(state)
        state.update(action)
        action_count += 1

    # Game should reach terminal state
    assert state.terminal, "Game should reach terminal state"

    # Final wealth should equal initial total
    assert abs(sum(state.wealth) - expected_total) < 0.01, \
        f"Wealth not conserved at game end: {sum(state.wealth)} != {expected_total}"
