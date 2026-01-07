"""
Tests for Action IntEnum to verify it works correctly in all contexts.

IntEnum should behave as integers in comparisons, arithmetic, NumPy arrays,
and neural network inputs while providing named constants for readability.
"""

import numpy as np
import pytest

from poker.config import Action, DEFAULT_ACTIONS, describe_action
from poker.agent import Agent
from poker.random_agent import RandomAgent
from poker.skillful_random_agent import SkillfulRandomAgent


def test_action_enum_values():
    """Verify Action enum has correct integer values."""
    assert Action.FOLD == -1
    assert Action.CHECK_CALL == 0
    assert Action.BET_1 == 1
    assert Action.BET_2 == 2
    assert Action.BET_3 == 3


def test_action_enum_comparisons():
    """Verify Action enum works in numeric comparisons."""
    # Less than comparisons (used for fold detection)
    assert Action.FOLD < 0
    assert not (Action.CHECK_CALL < 0)
    assert not (Action.BET_1 < 0)

    # Equality comparisons
    assert Action.FOLD == -1
    assert Action.CHECK_CALL == 0
    assert Action.BET_1 == 1

    # Greater than comparisons (used for bet detection)
    assert Action.BET_1 > 0
    assert Action.BET_2 > Action.BET_1
    assert Action.BET_3 > Action.BET_2


def test_action_enum_arithmetic():
    """Verify Action enum works in arithmetic operations."""
    # Addition
    assert Action.BET_1 + 1 == 2
    assert Action.CHECK_CALL + 5 == 5

    # Subtraction
    assert Action.BET_2 - 1 == 1
    assert Action.BET_1 - Action.CHECK_CALL == 1

    # Can use in range checks (minimum_bet <= action <= maximum_bet)
    min_bet = 0
    max_bet = 3
    assert min_bet <= Action.CHECK_CALL <= max_bet
    assert min_bet <= Action.BET_1 <= max_bet
    assert min_bet <= Action.BET_3 <= max_bet


def test_action_enum_to_int():
    """Verify Action enum converts to int correctly."""
    assert int(Action.FOLD) == -1
    assert int(Action.CHECK_CALL) == 0
    assert int(Action.BET_1) == 1
    assert int(Action.BET_2) == 2
    assert int(Action.BET_3) == 3


def test_action_enum_in_list():
    """Verify Action enum works in lists (for iteration)."""
    actions = [Action.FOLD, Action.CHECK_CALL, Action.BET_1, Action.BET_2, Action.BET_3]

    assert len(actions) == 5
    assert actions[0] == -1
    assert actions[1] == 0
    assert actions[2] == 1

    # List comprehension with filtering (used in agent code)
    positive_actions = [a for a in actions if a > 0]
    assert positive_actions == [1, 2, 3]

    fold_actions = [a for a in actions if a < 0]
    assert fold_actions == [-1]


def test_action_enum_in_numpy_array():
    """Verify Action enum works in NumPy arrays (critical for NN inputs)."""
    # Create NumPy array from Action enum values
    actions_array = np.array([Action.FOLD, Action.CHECK_CALL, Action.BET_1, Action.BET_2, Action.BET_3])

    # Should convert to integer array
    assert actions_array.dtype == np.int64 or actions_array.dtype == np.int32
    assert np.array_equal(actions_array, np.array([-1, 0, 1, 2, 3]))

    # Indexing should work
    assert actions_array[0] == -1
    assert actions_array[2] == 1

    # Broadcasting and comparisons should work
    mask = actions_array > 0
    assert np.array_equal(mask, np.array([False, False, True, True, True]))

    # Can use in argmax (used in agent code)
    q_values = np.array([0.5, 1.0, 0.8, 0.6, 0.4])
    best_action_idx = np.argmax(q_values)
    best_action = actions_array[best_action_idx]
    assert best_action == 0  # CHECK_CALL has highest Q-value


def test_default_actions_list():
    """Verify DEFAULT_ACTIONS is a proper list of Action enum values."""
    assert len(DEFAULT_ACTIONS) == 5
    assert DEFAULT_ACTIONS[0] == Action.FOLD
    assert DEFAULT_ACTIONS[1] == Action.CHECK_CALL
    assert DEFAULT_ACTIONS[2] == Action.BET_1
    assert DEFAULT_ACTIONS[3] == Action.BET_2
    assert DEFAULT_ACTIONS[4] == Action.BET_3

    # Should work in numeric comparisons
    assert DEFAULT_ACTIONS[0] < 0
    assert DEFAULT_ACTIONS[1] == 0
    assert all(a > 0 for a in DEFAULT_ACTIONS[2:])


def test_describe_action_with_enum():
    """Verify describe_action works with Action enum values."""
    # Fold
    assert describe_action(Action.FOLD) == "Fold"
    assert describe_action(Action.FOLD, min_bet=2) == "Fold"

    # Check (when min_bet=0)
    assert describe_action(Action.CHECK_CALL, min_bet=0) == "Check"

    # Call (when min_bet>0)
    assert describe_action(Action.CHECK_CALL, min_bet=2) == "Call $2"
    assert describe_action(Action.BET_2, min_bet=2) == "Call $2"

    # Bet (when min_bet=0)
    assert describe_action(Action.BET_1, min_bet=0) == "Bet $1"
    assert describe_action(Action.BET_2, min_bet=0) == "Bet $2"

    # Raise (when min_bet>0 and action > min_bet)
    assert describe_action(Action.BET_3, min_bet=1) == "Raise to $3"
    assert describe_action(Action.BET_2, min_bet=1) == "Raise to $2"


def test_describe_action_with_int():
    """Verify describe_action also works with plain integers."""
    # Should work the same as with enum
    assert describe_action(-1) == "Fold"
    assert describe_action(0, min_bet=0) == "Check"
    assert describe_action(0, min_bet=2) == "Call $2"
    assert describe_action(2, min_bet=0) == "Bet $2"
    assert describe_action(3, min_bet=1) == "Raise to $3"


def test_agent_uses_default_actions():
    """Verify Agent uses DEFAULT_ACTIONS when none provided."""
    agent = Agent(player_index=0, n_players=3)

    # Should have DEFAULT_ACTIONS
    assert len(agent.actions) == 5
    assert agent.actions[0] == Action.FOLD
    assert agent.actions[1] == Action.CHECK_CALL

    # Should work in numeric contexts
    assert agent.actions[0] < 0


def test_agent_accepts_custom_actions():
    """Verify Agent still accepts custom action sets."""
    custom_actions = [-1, 0, 1, 2, 3, 4, 5]
    agent = Agent(player_index=0, n_players=3, actions=custom_actions)

    assert len(agent.actions) == 7
    assert agent.actions[-1] == 5


def test_random_agent_uses_default_actions():
    """Verify RandomAgent uses DEFAULT_ACTIONS when none provided."""
    agent = RandomAgent(player_index=0)

    assert len(agent.actions) == 5
    assert agent.actions[0] == Action.FOLD
    assert agent.actions[1] == Action.CHECK_CALL


def test_skillful_random_agent_uses_default_actions():
    """Verify SkillfulRandomAgent uses DEFAULT_ACTIONS when none provided."""
    agent = SkillfulRandomAgent(player_index=0)

    assert len(agent.actions) == 5
    assert agent.actions[0] == Action.FOLD
    assert agent.actions[1] == Action.CHECK_CALL


def test_action_enum_in_agent_action_selection():
    """
    Integration test: Verify Action enum works in actual agent action selection.

    This tests that the enum works through the entire pipeline:
    - Agent stores actions as enum values
    - get_action returns enum values
    - Numeric comparisons work correctly
    """
    from poker.state import State

    agent = Agent(player_index=0, n_players=3)
    state = State(n_players=3, initial_wealth=25)

    # Get an action (will be random with high exploration)
    action = agent.get_action(state, proba_random_action=1.0)

    # Action should be one of the enum values
    assert action in DEFAULT_ACTIONS

    # Should work in numeric comparisons (fold detection)
    if action < 0:
        assert action == Action.FOLD

    # Should work in legality checks
    min_bet = state.minimum_legal_bet()
    max_bet = state.maximum_legal_bet()

    if action >= 0:
        assert min_bet <= action <= max_bet
