import numpy as np

from poker.state import GameStage, State


def test_state():

    initial_wealth = 200
    initial_dealer = 0
    state = State(
        n_players=3, initial_wealth=initial_wealth, initial_dealer=initial_dealer
    )

    n_cards_in_deck_after_initial_deal = 52 - state.n_players * 2
    assert len(state.shuffled_deck) == n_cards_in_deck_after_initial_deal

    assert state.game_stage == GameStage.PRE_FLOP
    assert len(state.public_cards) == 0

    assert all(player_has_folded is False for player_has_folded in state.has_folded)

    # Note: any action < 0 means the current player is folding
    action_fold = -1

    first_player = state.current_player

    state.update(action_fold)
    assert state.has_folded[first_player]

    second_player = state.current_player
    assert second_player == first_player + 1

    # Note: this is a bet of $10
    action_bet = 10

    state.update(action_bet)
    assert state.bets_by_stage[GameStage.PRE_FLOP][second_player] == [action_bet]
    assert state.game_stage == GameStage.PRE_FLOP

    # Note: the third player to act (i.e. the dealer) also bets $10. At this point,
    #  the first player to act has folded, and the other two have both bet $10.
    #  We move to the next stage (the flop) and the dealer deals three public cards
    state.update(action_bet)
    assert state.game_stage == GameStage.FLOP
    assert len(state.public_cards) == 3
    assert len(state.shuffled_deck) == n_cards_in_deck_after_initial_deal - 3

    # Note: the first player has already folded, so the first person to act
    #  after the flop is the second player
    assert state.current_player == second_player

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

    total_bets_by_dealer = action_bet * 3
    assert state.wealth[second_player] == initial_wealth + total_bets_by_dealer
    assert state.wealth[first_player] == initial_wealth
    assert state.wealth[initial_dealer] == initial_wealth - total_bets_by_dealer

    assert sum(state.wealth) == initial_wealth * state.n_players

    assert len(state.public_cards) == 0
    assert len(state.shuffled_deck) == n_cards_in_deck_after_initial_deal

    # Note: the first person to act in the first round is now the dealer
    assert state.dealer == initial_dealer + 1 == first_player

    assert all(player_has_folded is False for player_has_folded in state.has_folded)

    assert sum(state.wealth) == initial_wealth * state.n_players
    assert state.wealth[second_player] == initial_wealth + total_bets_by_dealer

    action_knock = 0
    action_small_bet = 10
    action_big_bet = 20

    assert state.game_stage == GameStage.PRE_FLOP

    for _ in range(state.n_players):
        # Note: all players bet $0 pre-flop (TODO Implement big and small blinds),
        #  and we move to the next stage
        state.update(action_knock)

    assert state.game_stage == GameStage.FLOP
    assert len(state.public_cards) == 3

    for _ in range(state.n_players):
        # Note: all players bet $10 after the flop
        #  and we move to the next stage
        state.update(action_small_bet)

    assert state.game_stage == GameStage.TURN
    assert len(state.public_cards) == 4

    for _ in range(state.n_players):
        # Note: all players bet $10 after the turn
        #  and we move to the next stage
        state.update(action_small_bet)

    assert state.game_stage == GameStage.RIVER
    assert len(state.public_cards) == 5

    # Note: even though the stage has not ended, all public cards have
    #  been dealt and we can calculate hand strengths
    hand_strengths = state.calculate_best_hand_strengths()
    winning_player = np.argmax(hand_strengths)

    wealth_before_winning = state.wealth[winning_player]

    # Note: no players have folded, so all hand strengths must be positive
    assert max(hand_strengths) > 0

    state.update(action_small_bet)
    state.update(action_big_bet)
    state.update(action_big_bet)

    # Note: this player initially made a small bet, and the next person raised them with a large bet
    #  The first person to act now completes their bet (calls the raise)
    state.update(action_big_bet - action_small_bet)

    # Note: the bets were small during the flop and turn, and big during the river
    amount_won = (state.n_players - 1) * (2 * action_small_bet + action_big_bet)
    assert state.wealth[winning_player] == wealth_before_winning + amount_won
