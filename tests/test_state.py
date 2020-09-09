from poker.state import get_next_state, State


def test_get_next_state():

    initial_state = State()

    n_cards_in_deck_after_initial_deal = 52 - initial_state.n_players * 2
    assert len(initial_state.shuffled_deck) == n_cards_in_deck_after_initial_deal

    # TODO Actions should represent betting or folding
    action = 123

    next_state = get_next_state(initial_state, action)

    assert next_state
