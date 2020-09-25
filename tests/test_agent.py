import sys
from unittest.mock import Mock
import numpy as np

# Note: we mock this module so that we don't import tensorflow during pytest
sys.modules["poker.q_function"] = Mock()

from poker.agent import Agent
from poker.state import State, GameStage


def test_agent():

    actions = [-1, 0, 1, 2, 3, 4]

    agent = Agent(player_index=0, actions=actions)

    game_state = State(n_players=3)

    private_state = agent.get_private_state(game_state)

    assert len(private_state) == agent.len_private_state
    assert private_state[0] == GameStage.PRE_FLOP
    assert private_state[1] == game_state.hole_cards[agent.player_index][0].rank
    assert private_state[2] == game_state.hole_cards[agent.player_index][0].suit
    assert private_state[3] == game_state.hole_cards[agent.player_index][1].rank
    assert private_state[4] == game_state.hole_cards[agent.player_index][1].suit

    model_input = agent.get_model_input(private_state, actions)
    assert model_input.shape == (len(actions), agent.n_inputs)

    assert np.allclose(model_input[:, -1], actions)
    assert np.allclose(model_input[:, 0], private_state[0])

    model_input_single_action = agent.get_model_input(private_state, actions=[-1])
    assert model_input_single_action.shape == (1, agent.n_inputs)
