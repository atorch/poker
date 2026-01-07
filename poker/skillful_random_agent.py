import numpy as np
from poker.cards import Rank
from poker.config import DEFAULT_ACTIONS


class SkillfulRandomAgent:
    """
    An agent that plays mostly randomly but has basic hand strength awareness.

    Key behavior:
    - For strong hands (pairs of face cards/aces, any hand with ace),
      folds with very low probability (2% instead of uniform random)
    - For weak hands (everything else), folds 70% of the time immediately
      (prevents agents from learning to exploit cumulative fold probability)
    - Otherwise plays randomly from legal actions

    Purpose: Prevents catastrophic forgetting during self-play by maintaining
    a signal that strong hands have value. Acts as regularization.
    Improved to prevent "always bet" exploitation strategies.
    """

    def __init__(self, player_index=0, actions=None, strong_hand_fold_prob=0.02, weak_hand_fold_prob=0.70):
        """
        Args:
            player_index: Index of this player
            actions: Available actions (default: DEFAULT_ACTIONS)
            strong_hand_fold_prob: Probability of folding a strong hand (default 2%, reduced from 5%)
            weak_hand_fold_prob: Probability of folding weak hands immediately (default 70%, new param)
        """
        if actions is None:
            actions = DEFAULT_ACTIONS

        self.player_index = player_index
        self.actions = actions
        self.strong_hand_fold_prob = strong_hand_fold_prob
        self.weak_hand_fold_prob = weak_hand_fold_prob

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
        Select a legal action, protecting strong hands and folding weak hands more quickly.

        Args:
            game_state: Current game state
            proba_random_action: Ignored (always uses hand-strength logic)

        Returns:
            A legal action based on hand strength
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
            # For strong hands, almost never fold (reduced from 5% to 2%)
            if np.random.random() < self.strong_hand_fold_prob:
                # Fold with very low probability even with strong hand
                fold_actions = [a for a in legal_actions if a < 0]
                if fold_actions:
                    return np.random.choice(fold_actions)

            # Otherwise, choose randomly from non-fold actions
            non_fold_actions = [a for a in legal_actions if a >= 0]
            if non_fold_actions:
                return np.random.choice(non_fold_actions)

        else:
            # For weak hands, fold much more aggressively (new: 70% of the time)
            # This prevents the agent from learning to exploit easy folds over multiple rounds
            if len(legal_actions) > 1 and np.random.random() < self.weak_hand_fold_prob:
                fold_actions = [a for a in legal_actions if a < 0]
                if fold_actions:
                    return np.random.choice(fold_actions)

            # Otherwise, play randomly from all legal actions
            return np.random.choice(legal_actions)

        # Fallback (shouldn't reach here)
        return np.random.choice(legal_actions)
