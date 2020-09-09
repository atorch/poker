from poker.state import GameStage, State


def test_state():

    state = State()

    n_cards_in_deck_after_initial_deal = 52 - state.n_players * 2
    assert len(state.shuffled_deck) == n_cards_in_deck_after_initial_deal

    assert all(player_has_folded is False for player_has_folded in state.has_folded)

    # Note: an action of -1 means the current player has folded
    action_fold = -1

    # Note: this is the first player to take action
    initial_player = state.current_player

    state.update(action_fold)
    assert state.has_folded[initial_player]

    second_player = state.current_player
    assert second_player == initial_player + 1

    action_bet = 10

    state.update(action_bet)
    assert state.bets_by_stage[GameStage.PRE_FLOP][second_player] == [10]
