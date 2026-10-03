"""
Tests for the relative action heads in poker.actions.
"""
import random

import numpy as np

from poker.actions import Head, N_HEADS, game_action, head_for_game_action, legal_mask
from poker.config import DEFAULT_ACTIONS
from poker.random_agent import RandomAgent
from poker.state import State


def test_game_action_mapping():
    assert game_action(Head.FOLD, 2) == -1
    assert game_action(Head.CHECK_CALL, 0) == 0
    assert game_action(Head.CHECK_CALL, 2) == 2
    assert game_action(Head.RAISE_1, 2) == 3
    assert game_action(Head.RAISE_2, 1) == 3
    assert game_action(Head.RAISE_3, 0) == 3
    for owed in range(4):
        for head in Head:
            action = game_action(head, owed)
            assert head_for_game_action(action, owed) == head


def test_first_to_act_preflop():
    # The dealer acts first preflop, facing the $2 big blind: fold, call $2 or raise by 1 ($3)
    state = State(n_players=3, initial_wealth=20, initial_dealer=0)
    assert legal_mask(state).tolist() == [True, True, True, False, False]


def test_masks_agree_with_state_legal_actions_in_random_games():
    random.seed(0)
    np.random.seed(0)
    agents = [RandomAgent(player_index=i) for i in range(3)]
    n_checked = 0
    for _ in range(300):
        state = State(n_players=3, initial_wealth=[random.randint(5, 35) for _ in range(3)],
                      initial_dealer=random.randrange(3))
        while state.n_deals == 1 and not state.terminal:
            owed = state.minimum_legal_bet()
            mask = legal_mask(state)
            legal_from_heads = {game_action(head, owed) for head in range(N_HEADS) if mask[head]}
            assert legal_from_heads == set(state.legal_actions(DEFAULT_ACTIONS))
            n_checked += 1
            state.update(agents[state.current_player].get_action(state))
    assert n_checked > 1000
