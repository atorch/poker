"""
Simple fixed-strategy baseline agents for benchmarking.

These ignore their cards entirely, so any agent that has learned real poker should beat them.
They are useful precisely because they are hard to beat by accident:
- MaxRaiseAgent punishes agents that fold too much
- CallStationAgent punishes agents that bluff too much
"""
from poker.config import DEFAULT_ACTIONS


class MaxRaiseAgent:
    """Always puts in the largest legal amount (never folds)."""

    def __init__(self, player_index=0, actions=None):
        self.player_index = player_index
        self.actions = DEFAULT_ACTIONS if actions is None else actions

    def get_action(self, game_state, proba_random_action=0.0):
        return max(game_state.legal_actions(self.actions))


class CallStationAgent:
    """Always checks or calls (never folds, never raises)."""

    def __init__(self, player_index=0, actions=None):
        self.player_index = player_index
        self.actions = DEFAULT_ACTIONS if actions is None else actions

    def get_action(self, game_state, proba_random_action=0.0):
        non_fold_actions = [action for action in game_state.legal_actions(self.actions) if action >= 0]
        return min(non_fold_actions)
