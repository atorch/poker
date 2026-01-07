"""
Configuration constants for poker agent training and evaluation.

These values define the initial wealth distribution at the start of each training episode.
During episodes, wealth will vary (down to 0 when losing, higher when winning).
Sanity checks should use initial wealth near this range to ensure in-distribution testing.
"""

from enum import IntEnum

# Training configuration: initial wealth range at episode start
# Each episode begins with wealth randomly sampled from [MIN, MAX]
# Lower bound set to $5 to ensure Q-values remain finite:
#   - Players with low wealth are more likely to go broke (terminal state)
#   - Terminal states have Q-values bounded by immediate reward
#   - This prevents unbounded value accumulation during training
MIN_INITIAL_WEALTH = 5
MAX_INITIAL_WEALTH = 35
TYPICAL_INITIAL_WEALTH = (MIN_INITIAL_WEALTH + MAX_INITIAL_WEALTH) // 2  # 20


class Action(IntEnum):
    """
    Poker action constants using IntEnum for type safety and readability.

    IntEnum members behave as integers in all contexts (comparisons, arithmetic,
    NumPy arrays, neural network inputs) while providing named constants for clarity.

    Actions represent betting amounts:
        FOLD: Fold hand (always legal)
        CHECK_CALL: Check if min_bet=0, Call otherwise
        BET_1, BET_2, BET_3: Raise/Bet to $1, $2, $3

    IMPORTANT: The neural network is trained with (state, action) inputs.
    Changing this enum requires retraining all models from scratch!
    """
    FOLD = -1
    CHECK_CALL = 0
    BET_1 = 1
    BET_2 = 2
    BET_3 = 3


# Default action set for agents
# This is a list for backwards compatibility and iteration
# Use Action enum for named access (e.g., Action.FOLD)
DEFAULT_ACTIONS = [Action.FOLD, Action.CHECK_CALL, Action.BET_1, Action.BET_2, Action.BET_3]


def describe_action(action, min_bet=0):
    """
    Get human-readable description of an action.

    Args:
        action: Action value (int or Action enum)
        min_bet: Minimum legal bet at this decision point

    Returns:
        str: Description like "Fold", "Check", "Call $2", "Raise to $3"
    """
    action_int = int(action)  # Convert IntEnum to int if needed

    if action_int < 0:
        return "Fold"
    elif action_int == 0:
        return "Check" if min_bet == 0 else f"Call ${min_bet}"
    elif action_int == min_bet and min_bet > 0:
        return f"Call ${action_int}"
    else:
        return f"Raise to ${action_int}" if min_bet > 0 else f"Bet ${action_int}"
