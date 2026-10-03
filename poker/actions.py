"""
Relative action heads for learned agents, and their mapping to game actions.

The game's actions (poker.config.Action) are "chips put in now", so the same number can be a
call in one spot and a raise in another. Learned agents choose instead among heads that mean
the same thing in every spot:

    FOLD        fold (only legal when facing a bet)
    CHECK_CALL  put in exactly what is owed (a check when nothing is owed)
    RAISE_1..3  put in what is owed plus 1, 2 or 3

A head is legal when the chips it puts in are one of the game's actions (at most $3 per action)
and State.is_legal allows it, so the betting rules still live in poker/state.py. For example,
facing $1, RAISE_2 puts in $3 and RAISE_3 ($4) is illegal; facing $3, no raise is possible.
"""
from enum import IntEnum

import numpy as np

from poker.config import DEFAULT_ACTIONS


class Head(IntEnum):
    FOLD = 0
    CHECK_CALL = 1
    RAISE_1 = 2
    RAISE_2 = 3
    RAISE_3 = 4


HEADS = list(Head)
N_HEADS = len(HEADS)
HEAD_NAMES = [head.name for head in HEADS]


def game_action(head, owed):
    """The game action (chips to put in, or -1 to fold) for a head when `owed` chips are owed."""
    head = Head(head)
    if head == Head.FOLD:
        return -1
    return owed + (head - Head.CHECK_CALL)


def head_for_game_action(action, owed):
    """The head that a legal game action corresponds to."""
    if action < 0:
        return Head.FOLD
    return Head(Head.CHECK_CALL + action - owed)


def legal_mask(state):
    """Boolean array over HEADS for the player to act (consistent with State.legal_actions)."""
    owed = state.minimum_legal_bet()
    legal_actions = set(state.legal_actions(DEFAULT_ACTIONS))
    return np.array([game_action(head, owed) in legal_actions for head in HEADS])
