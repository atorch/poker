from poker.state import get_next_state, State


def test_get_next_state():

    initial_state = State()

    # TODO Actions should represent betting or folding
    action = 123

    next_state = get_next_state(initial_state, action)

    assert next_state
