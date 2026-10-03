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

# Maximum number of voluntary bets/raises per betting stage (a bet plus three raises,
# as in limit hold'em). Forced blinds don't count. Without a cap, betting escalates
# until someone is all in.
MAX_RAISES_PER_STAGE = 4


class Action(IntEnum):
    """
    Poker action constants using IntEnum for type safety and readability.

    IntEnum members behave as integers in all contexts (comparisons, arithmetic,
    NumPy arrays, neural network inputs) while providing named constants for clarity.

    Actions represent the number of chips the player puts in with this action:
        FOLD: Fold hand (only legal when facing a bet)
        CHECK_CALL: Check (only legal when min_bet=0)
        BET_1, BET_2, BET_3: Put in $1, $2, $3. This is a call if it equals the amount owed
            (min_bet), and a raise by (amount - min_bet) if it exceeds it. For example,
            facing a $2 bet, BET_2 is a call and BET_3 is a raise by $1.

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
        str: Description like "Fold", "Check", "Call $2", "Bet $2", "Raise by $1 (put in $3)"
    """
    action_int = int(action)  # Convert IntEnum to int if needed
    min_bet = int(min_bet)

    if action_int < 0:
        return "Fold"
    elif action_int == 0:
        return "Check" if min_bet == 0 else f"Call ${min_bet}"
    elif action_int == min_bet:
        return f"Call ${action_int}"
    elif min_bet == 0:
        return f"Bet ${action_int}"
    else:
        return f"Raise by ${action_int - min_bet} (put in ${action_int})"
