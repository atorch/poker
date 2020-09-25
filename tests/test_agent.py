import sys
from unittest.mock import Mock
# Note: we mock this module so that we don't import tensorflow during pytest
sys.modules["poker.q_function"] = Mock()

from poker.agent import Agent
from poker.state import State


def test_agent():

    actions = [-1, 0, 1, 2, 3, 4]

    agent = Agent(player_index=0, actions=actions)

    game_state = State(n_players=3)

    private_state = agent.get_private_state(game_state)

    # TODO Assert on private_state
    # TODO Call get_model_input(self, private_state, actions) and assert on model_input
