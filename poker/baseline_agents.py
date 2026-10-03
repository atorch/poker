"""
Simple fixed-strategy baseline agents for benchmarking.

MaxRaiseAgent and CallStationAgent ignore their cards entirely, so any agent that has learned
real poker should beat them. They are useful precisely because they are hard to beat by accident:
- MaxRaiseAgent punishes agents that fold too much
- CallStationAgent punishes agents that bluff too much

TagAgent is a crude rule-based player that does use its cards (and the board), as a reference
point: a learned agent should beat it too.
"""
from poker.cards import Rank
from poker.config import DEFAULT_ACTIONS
from poker.hands import HandCategory, best_hand_strength, hand_category, leading_rank, made_hand_category
from poker.state import GameStage

STRONG, MEDIUM, WEAK = "strong", "medium", "weak"


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


def preflop_hand_class(hole_cards):
    """
    STRONG: 77+, AQ+, AJs, KQs
    MEDIUM: any other pair, any ace, two cards ten or higher, suited connectors 56s and up
    WEAK: everything else
    """
    low, high = sorted(card.rank for card in hole_cards)
    suited = hole_cards[0].suit == hole_cards[1].suit
    pair = low == high

    if (
        (pair and low >= Rank.SEVEN)
        or (high == Rank.ACE and low >= Rank.QUEEN)
        or (suited and (high, low) in [(Rank.ACE, Rank.JACK), (Rank.KING, Rank.QUEEN)])
    ):
        return STRONG
    if pair or high == Rank.ACE or low >= Rank.TEN or (suited and high - low == 1 and low >= Rank.FIVE):
        return MEDIUM
    return WEAK


def postflop_hand_class(hole_cards, public_cards):
    """
    STRONG: two pair or better, or top pair / an overpair (pair rank >= highest board card)
    MEDIUM: any other hand that improves on what the board alone makes
    WEAK: the board plays (the hole cards add nothing to the made-hand category)

    This deliberately ignores draws, kickers and board texture.
    """
    strength_value, _ = best_hand_strength(list(public_cards), list(hole_cards))
    category = hand_category(strength_value)

    if category <= made_hand_category(public_cards):
        return WEAK
    if category >= HandCategory.TWO_PAIR:
        return STRONG
    if category == HandCategory.PAIR and leading_rank(strength_value) >= max(card.rank for card in public_cards):
        return STRONG
    return MEDIUM


class TagAgent:
    """
    A rule-based "tight-aggressive" player: strong hands put in the maximum, medium hands
    check or call, weak hands check or fold (see preflop_hand_class / postflop_hand_class).

    With fold_postflop=False it never folds after the flop (weak hands check or call instead),
    which does much better against MaxRaiseAgent.
    """

    def __init__(self, player_index=0, actions=None, fold_postflop=True):
        self.player_index = player_index
        self.actions = DEFAULT_ACTIONS if actions is None else actions
        self.fold_postflop = fold_postflop

    def hand_class(self, game_state):
        hole_cards = game_state.hole_cards[self.player_index]
        if game_state.game_stage == GameStage.PRE_FLOP:
            return preflop_hand_class(hole_cards)
        return postflop_hand_class(hole_cards, game_state.public_cards)

    def get_action(self, game_state, proba_random_action=0.0):
        legal_actions = game_state.legal_actions(self.actions)
        hand_class = self.hand_class(game_state)
        check_or_call = min(action for action in legal_actions if action >= 0)

        if hand_class == STRONG:
            return max(legal_actions)
        if hand_class == MEDIUM:
            return check_or_call
        if not self.fold_postflop and game_state.game_stage != GameStage.PRE_FLOP:
            return check_or_call
        # Note: the smallest legal action is a fold when facing a bet, and a check otherwise
        return min(legal_actions)
