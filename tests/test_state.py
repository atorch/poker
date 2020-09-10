from poker.state import GameStage, State


def test_state():

    initial_wealth = 200
    state = State(n_players=3, initial_wealth=initial_wealth)

    n_cards_in_deck_after_initial_deal = 52 - state.n_players * 2
    assert len(state.shuffled_deck) == n_cards_in_deck_after_initial_deal

    assert len(state.public_cards) == 0

    assert all(player_has_folded is False for player_has_folded in state.has_folded)

    # Note: an action of -1 means the current player has folded
    action_fold = -1

    first_player = state.current_player

    state.update(action_fold)
    assert state.has_folded[first_player]

    second_player = state.current_player
    assert second_player == first_player + 1

    action_bet = 10

    state.update(action_bet)
    assert state.bets_by_stage[GameStage.PRE_FLOP][second_player] == [action_bet]
    assert state.game_stage == GameStage.PRE_FLOP

    # Note: the third player to act also bets $10. At this point,
    #  the first player to act has folded, and the other two have both bet $10
    state.update(action_bet)

    assert state.game_stage == GameStage.FLOP

    # Note: the first player has already folded, so the first person to act
    #  after the flop is the second player
    assert state.current_player == second_player

    assert len(state.public_cards) == 3
    assert len(state.shuffled_deck) == n_cards_in_deck_after_initial_deal - 3

    # Note: the second player bets $10, then the dealer bets $20,
    #  the first player has already folded and does nothing, and then the second player calls
    state.update(action_bet)
    state.update(2 * action_bet)
    state.update(action_bet)

    assert state.bets_by_stage[GameStage.FLOP][first_player] == []
    assert state.bets_by_stage[GameStage.FLOP][second_player] == [
        action_bet,
        action_bet,
    ]
    assert state.bets_by_stage[GameStage.FLOP][state.dealer] == [2 * action_bet]

    assert state.game_stage == GameStage.TURN
    assert len(state.public_cards) == 4

    assert state.current_player == second_player
    state.update(action_bet)

    assert state.current_player == state.dealer
    state.update(action_fold)

    total_bets_by_other_players = sum(
        state.bets_by_stage[GameStage.FLOP][state.dealer]
    ) + sum(state.bets_by_stage[GameStage.PRE_FLOP][state.dealer])
    assert state.wealth[second_player] == initial_wealth + total_bets_by_other_players
    assert state.wealth[first_player] == initial_wealth
    assert state.wealth[state.dealer] == initial_wealth - total_bets_by_other_players
