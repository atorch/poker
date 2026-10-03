"""
State -> feature vector for learned agents. Saved models record FEATURE_VERSION, so bump it
whenever the encoding changes (old models then fail to load instead of misreading inputs).

Everything is from the point of view of `seat` (normally the player to act):
- stage (one-hot), hole cards and board (52-dim binary each, so card order doesn't matter)
- the made-hand category of hole + board (one-hot), whether it improves on what the board
  alone makes, and the preflop equity of the hole cards against the active opponents
  (poker.preflop_equity; vs 1 or 2 random hands)
- money, in big blinds / MONEY_SCALE: pot, amount owed, pot odds, own stack behind, own chips
  committed this deal and this street, effective stack behind (the shortest active stack,
  since there are no side pots)
- street context: raises this street and own raises this street (over the cap), cap reached
- position: seat offset from the dealer (one-hot), number of players, number still in the hand
- one block per opponent in relative seat order (seat + 1, seat + 2, ...), padded to
  MAX_PLAYERS - 1 blocks: present, still in the hand, stack behind, committed this deal and this
  street, raises this street and this deal (over the cap)

Padding to MAX_PLAYERS means 2..MAX_PLAYERS-player games share one layout.
"""
from functools import lru_cache

import numpy as np

from poker.cards import FULL_DECK, card_index
from poker.hands import HandCategory, best_hand_strength, hand_category, made_hand_category
from poker.preflop_equity import preflop_equity
from poker.state import GameStage

FEATURE_VERSION = 1
MAX_PLAYERS = 6
MONEY_SCALE = 10.0  # in big blinds

OPPONENT_FIELDS = ["present", "active", "stack_behind", "committed_deal", "committed_street",
                   "raises_street", "raises_deal"]

LAYOUT = (
    [(f"stage_{stage.name.lower()}", 1) for stage in GameStage]
    + [("hole", 52), ("board", 52)]
    + [(f"made_{category.name.lower()}", 1) for category in HandCategory]
    + [("improves_board", 1), ("preflop_equity", 1)]
    + [(name, 1) for name in ["pot", "owed", "pot_odds", "stack_behind", "committed_deal", "committed_street",
                              "effective_behind"]]
    + [("raises_street", 1), ("own_raises_street", 1), ("raise_cap_reached", 1)]
    + [(f"position_{offset}", 1) for offset in range(MAX_PLAYERS)]
    + [("n_players", 1), ("n_active", 1)]
    + [(f"opponent{i + 1}_{field}", 1) for i in range(MAX_PLAYERS - 1) for field in OPPONENT_FIELDS]
)

OFFSETS = {}
_offset = 0
for _name, _size in LAYOUT:
    OFFSETS[_name] = _offset
    _offset += _size
N_FEATURES = _offset
FEATURE_NAMES = [name if size == 1 else f"{name}_{i}" for name, size in LAYOUT for i in range(size)]


@lru_cache(maxsize=200_000)
def _made_hand(hole_key, board_key):
    """(HandCategory, improves on the board alone) for sorted card-index tuples."""
    hole = [FULL_DECK[i] for i in hole_key]
    board = [FULL_DECK[i] for i in board_key]
    if board:
        category = hand_category(best_hand_strength(board, hole)[0])
    else:
        category = HandCategory.PAIR if hole[0].rank == hole[1].rank else HandCategory.HIGH_CARD
    return category, category > made_hand_category(board)


def encode(state, seat, out=None):
    """The feature vector (float32, length N_FEATURES) of `state` for the player in `seat`."""
    features = np.zeros(N_FEATURES, dtype=np.float32) if out is None else out
    if out is not None:
        features[:] = 0

    n_players = state.n_players
    assert n_players <= MAX_PLAYERS
    stage = state.game_stage
    big_blind = state.big_blind
    cap = state.max_raises_per_stage

    def money(chips):
        return chips / big_blind / MONEY_SCALE

    committed_deal = [state.total_bet_by_player(p) for p in range(n_players)]
    committed_street = [sum(state.bets_by_stage[stage][p]) for p in range(n_players)]
    behind = [state.wealth[p] - committed_deal[p] for p in range(n_players)]
    active = [not folded for folded in state.has_folded]
    active_players = [p for p in range(n_players) if active[p]]

    features[OFFSETS[f"stage_{stage.name.lower()}"]] = 1

    hole = state.hole_cards[seat]
    hole_key = tuple(sorted(card_index(card) for card in hole))
    board_key = tuple(sorted(card_index(card) for card in state.public_cards))
    for index in hole_key:
        features[OFFSETS["hole"] + index] = 1
    for index in board_key:
        features[OFFSETS["board"] + index] = 1

    category, improves = _made_hand(hole_key, board_key)
    features[OFFSETS[f"made_{category.name.lower()}"]] = 1
    features[OFFSETS["improves_board"]] = improves
    n_active_opponents = len(active_players) - (1 if active[seat] else 0)
    features[OFFSETS["preflop_equity"]] = preflop_equity(hole, n_opponents=min(2, max(1, n_active_opponents)))

    pot = state.total_bets()
    owed = max(committed_street[p] for p in active_players) - committed_street[seat]
    features[OFFSETS["pot"]] = money(pot)
    features[OFFSETS["owed"]] = money(owed)
    features[OFFSETS["pot_odds"]] = owed / (pot + owed) if owed > 0 else 0.0
    features[OFFSETS["stack_behind"]] = money(behind[seat])
    features[OFFSETS["committed_deal"]] = money(committed_deal[seat])
    features[OFFSETS["committed_street"]] = money(committed_street[seat])
    features[OFFSETS["effective_behind"]] = money(min(behind[p] for p in active_players))

    raises = state.raises_by_stage_and_player
    features[OFFSETS["raises_street"]] = state.raises_by_stage[stage] / cap
    features[OFFSETS["own_raises_street"]] = raises[stage][seat] / cap
    features[OFFSETS["raise_cap_reached"]] = state.raise_cap_reached()

    features[OFFSETS[f"position_{(seat - state.dealer) % n_players}"]] = 1
    features[OFFSETS["n_players"]] = n_players / MAX_PLAYERS
    features[OFFSETS["n_active"]] = len(active_players) / MAX_PLAYERS

    for i in range(n_players - 1):
        p = (seat + 1 + i) % n_players
        base = OFFSETS[f"opponent{i + 1}_present"]
        features[base:base + len(OPPONENT_FIELDS)] = [
            1.0,
            active[p],
            money(behind[p]),
            money(committed_deal[p]),
            money(committed_street[p]),
            raises[stage][p] / cap,
            sum(raises[s][p] for s in GameStage) / cap,
        ]
    return features


def describe(features):
    """{feature name: value} for the nonzero features (for debugging)."""
    return {FEATURE_NAMES[i]: float(value) for i, value in enumerate(features) if value != 0}
