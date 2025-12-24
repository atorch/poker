import numpy as np
from poker.cards import Rank


class SkillfulRandomAgent:
    """
    An agent that plays mostly randomly but has basic hand strength awareness.

    Key behavior:
    - For strong hands (pairs of face cards/aces, any hand with ace),
      folds with very low probability (5% instead of uniform random)
    - Otherwise plays completely randomly

    Purpose: Prevents catastrophic forgetting during self-play by maintaining
    a signal that strong hands have value. Acts as regularization.
    """

    def __init__(self, player_index=0, actions=[-1, 0, 1, 2, 3], strong_hand_fold_prob=0.05):
        """
        Args:
            player_index: Index of this player
            actions: Available actions (default: fold=-1, check=0, bets=1,2,3)
            strong_hand_fold_prob: Probability of folding a strong hand (default 5%)
        """
        self.player_index = player_index
        self.actions = actions
        self.strong_hand_fold_prob = strong_hand_fold_prob

    def is_strong_hand(self, cards):
        """
        Determine if a hand is strong enough to avoid folding.

        Strong hands:
        - Any hand with an Ace
        - Pair of face cards or tens (TT, JJ, QQ, KK)

        Args:
            cards: List of 2 Card objects

        Returns:
            bool: True if hand is strong
        """
        if len(cards) != 2:
            return False

        ranks = [card.rank for card in cards]

        # Any hand with an Ace is strong
        if Rank.ACE in ranks:
            return True

        # Pair of tens or better (TT, JJ, QQ, KK)
        if ranks[0] == ranks[1] and ranks[0] >= Rank.TEN:
            return True

        return False

    def get_action(self, game_state, proba_random_action=1.0):
        """
        Select a legal action, avoiding folding strong hands.

        Args:
            game_state: Current game state
            proba_random_action: Ignored (always uses hand-strength logic)

        Returns:
            A legal action, with reduced folding for strong hands
        """
        minimum_legal_bet = game_state.minimum_legal_bet()
        maximum_legal_bet = game_state.maximum_legal_bet()

        # Get all legal actions
        legal_actions = []
        for action in self.actions:
            # Folding (negative action) is always legal
            if action < 0 or (minimum_legal_bet <= action <= maximum_legal_bet):
                legal_actions.append(action)

        # Check if we have a strong hand
        my_cards = game_state.hole_cards[self.player_index]
        is_strong = self.is_strong_hand(my_cards)

        if is_strong and len(legal_actions) > 1:
            # For strong hands, only fold with low probability
            if np.random.random() < self.strong_hand_fold_prob:
                # Fold with low probability even with strong hand
                fold_actions = [a for a in legal_actions if a < 0]
                if fold_actions:
                    return np.random.choice(fold_actions)

            # Otherwise, choose randomly from non-fold actions
            non_fold_actions = [a for a in legal_actions if a >= 0]
            if non_fold_actions:
                return np.random.choice(non_fold_actions)

        # For weak hands or when only one action is legal, choose uniformly
        return np.random.choice(legal_actions)
