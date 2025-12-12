import numpy as np


class RandomAgent:
    """
    A simple agent that selects legal actions uniformly at random.
    Useful as a baseline opponent for curriculum learning.
    """

    def __init__(self, player_index=0, actions=[-1, 0, 1, 2, 3]):
        self.player_index = player_index
        self.actions = actions

    def get_action(self, game_state, proba_random_action=1.0):
        """
        Select a legal action uniformly at random.

        Args:
            game_state: Current game state
            proba_random_action: Ignored (always random)

        Returns:
            A randomly selected legal action
        """
        minimum_legal_bet = game_state.minimum_legal_bet()
        maximum_legal_bet = game_state.maximum_legal_bet()

        legal_actions = []
        for action in self.actions:
            # Folding (negative action) is always legal
            if action < 0 or (minimum_legal_bet <= action <= maximum_legal_bet):
                legal_actions.append(action)

        return np.random.choice(legal_actions)
