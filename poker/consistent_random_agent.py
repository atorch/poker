"""
ConsistentRandomAgent: A random agent that prevents cumulative fold exploitation.

Key insight: Instead of making i.i.d. random decisions on every action,
commit to a playing style once per deal and stick with it.

This prevents agents from learning to exploit cumulative fold probability
(e.g., "always raise $3" exploits 20% fold per action → 67% fold after 5 rounds).

By committing upfront, we ensure agents see diverse showdowns and learn hand evaluation,
not just fold exploitation patterns.
"""

import numpy as np
from poker.cards import Rank
from poker.config import DEFAULT_ACTIONS


class ConsistentRandomAgent:
    """
    A random agent that commits to a playing style at the start of each deal.

    Strategy Distribution (committed once per deal):
    - Aggressive (30%): Plays strong, rarely folds
    - Passive (40%): Folds weak hands immediately, plays strong hands
    - Bluffing (30%): Plays weak hands like strong hands (always bets)

    IMPORTANT DESIGN CHOICE: Strategy is chosen RANDOMLY at the start of each deal,
    WITHOUT looking at hand strength. This means:
    - Pocket aces might get "passive" strategy (and fold to aggression)
    - 7-2 offsuit might get "bluffing" strategy (and bet aggressively all round)
    This creates maximum diversity in training scenarios. Future improvement could
    make strategy choice depend on hand strength for more realistic play.

    This prevents i.i.d. fold exploitation:
    - Can't exploit P(fold after n actions) = 1 - 0.8^n
    - Forces agent to learn hand strength evaluation
    - Sees diverse showdowns (both bluffs and strong hands)
    """

    def __init__(self, player_index=0, actions=None, strategy_probs=None):
        """
        Args:
            player_index: Index of this player
            actions: Available actions (default: DEFAULT_ACTIONS)
            strategy_probs: [aggressive, passive, bluffing] probabilities
                           (default: [0.3, 0.4, 0.3])
        """
        if actions is None:
            actions = DEFAULT_ACTIONS

        if strategy_probs is None:
            strategy_probs = [0.3, 0.4, 0.3]  # [aggressive, passive, bluffing]

        assert len(strategy_probs) == 3, "strategy_probs must be [aggressive, passive, bluffing]"
        assert abs(sum(strategy_probs) - 1.0) < 0.01, "strategy_probs must sum to 1.0"

        self.player_index = player_index
        self.actions = actions
        self.strategy_probs = strategy_probs

        # Track current deal's committed strategy
        self.current_deal_number = -1
        self.current_strategy = None  # Will be 'aggressive', 'passive', or 'bluffing'

    def is_strong_hand(self, cards):
        """
        Determine if a hand is strong.

        Strong hands:
        - Any hand with an Ace
        - Pair of tens or better (TT, JJ, QQ, KK, AA)

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

        # Pair of tens or better (TT, JJ, QQ, KK, AA)
        if ranks[0] == ranks[1] and ranks[0] >= Rank.TEN:
            return True

        return False

    def get_action(self, game_state, proba_random_action=1.0):
        """
        Select a legal action based on committed strategy for this deal.

        On new deal: Commit to a strategy (aggressive/passive/bluffing)
        On subsequent actions: Act consistently with that strategy

        This prevents i.i.d. exploitation - can't use cumulative fold probability.

        Args:
            game_state: Current game state
            proba_random_action: Ignored (always uses strategy logic)

        Returns:
            A legal action consistent with this deal's committed strategy
        """
        # On new deal, commit to a strategy
        # NOTE: Strategy is chosen RANDOMLY without looking at hand strength first.
        # This creates maximum diversity (e.g., bluffing with 7-2, being passive with AA).
        # Future improvement: Could make strategy choice conditional on hand strength.
        if game_state.n_deals != self.current_deal_number:
            self.current_strategy = np.random.choice(
                ['aggressive', 'passive', 'bluffing'],
                p=self.strategy_probs
            )
            self.current_deal_number = game_state.n_deals

        # Get legal actions
        minimum_legal_bet = game_state.minimum_legal_bet()
        maximum_legal_bet = game_state.maximum_legal_bet()

        legal_actions = []
        for action in self.actions:
            if action < 0 or (minimum_legal_bet <= action <= maximum_legal_bet):
                legal_actions.append(action)

        # Get hand strength
        my_cards = game_state.hole_cards[self.player_index]
        is_strong = self.is_strong_hand(my_cards)

        # Execute committed strategy
        if self.current_strategy == 'aggressive':
            # Aggressive: Play strong, rarely fold (even with weak hands sometimes)
            if is_strong:
                # Strong hand: Never fold, always bet/raise
                non_fold_actions = [a for a in legal_actions if a >= 0]
                if non_fold_actions:
                    return np.random.choice(non_fold_actions)
            else:
                # Weak hand: Fold 60% of time, bet 40%
                if len(legal_actions) > 1 and np.random.random() < 0.60:
                    fold_actions = [a for a in legal_actions if a < 0]
                    if fold_actions:
                        return np.random.choice(fold_actions)
                # Otherwise bet
                non_fold_actions = [a for a in legal_actions if a >= 0]
                if non_fold_actions:
                    return np.random.choice(non_fold_actions)

        elif self.current_strategy == 'passive':
            # Passive: Conservative - fold weak, bet strong
            if is_strong:
                # Strong hand: Always bet/raise
                non_fold_actions = [a for a in legal_actions if a >= 0]
                if non_fold_actions:
                    return np.random.choice(non_fold_actions)
            else:
                # Weak hand: Always fold immediately (70% overall fold rate for weak hands)
                fold_actions = [a for a in legal_actions if a < 0]
                if fold_actions and len(legal_actions) > 1:
                    return np.random.choice(fold_actions)

        elif self.current_strategy == 'bluffing':
            # Bluffing: Play weak hand like strong hand - always bet
            non_fold_actions = [a for a in legal_actions if a >= 0]
            if non_fold_actions:
                return np.random.choice(non_fold_actions)

        # Fallback: choose randomly from all legal actions
        return np.random.choice(legal_actions)
