import sys
from unittest.mock import Mock
import numpy as np
import pytest

# Note: we mock this module so that we don't import tensorflow during pytest
sys.modules["poker.q_function"] = Mock()

from poker.agent import Agent, softmax_with_temperature
from poker.state import State, GameStage


def test_agent():

    actions = [-1, 0, 1, 2, 3, 4]
    n_players = 3

    agent = Agent(player_index=0, actions=actions, n_players=n_players)

    game_state = State(n_players=n_players)

    private_state = agent.get_private_state(game_state)

    assert len(private_state) == agent.len_private_state
    assert private_state[0] == GameStage.PRE_FLOP
    assert private_state[1] == game_state.hole_cards[agent.player_index][0].rank
    assert private_state[2] == game_state.hole_cards[agent.player_index][0].suit
    assert private_state[3] == game_state.hole_cards[agent.player_index][1].rank
    assert private_state[4] == game_state.hole_cards[agent.player_index][1].suit

    model_input = agent.get_model_input(private_state, actions)
    assert model_input.shape == (len(actions), agent.n_inputs)

    assert np.allclose(model_input[:, -1], actions)
    assert np.allclose(model_input[:, 0], private_state[0])

    model_input_single_action = agent.get_model_input(private_state, actions=[-1])
    assert model_input_single_action.shape == (1, agent.n_inputs)


def test_softmax_basic():
    """Test that softmax produces valid probability distribution."""
    q_values = np.array([1.0, 2.0, 3.0, 4.0])
    probs = softmax_with_temperature(q_values, temperature=1.0)

    # Check probabilities sum to 1
    assert np.isclose(np.sum(probs), 1.0)

    # Check all probabilities are non-negative
    assert np.all(probs >= 0)

    # Check higher Q-values have higher probabilities
    assert probs[3] > probs[2] > probs[1] > probs[0]


def test_softmax_with_illegal_actions():
    """Test that illegal actions (Q=-inf) get probability 0."""
    q_values = np.array([1.0, -np.inf, 3.0, -np.inf, 2.0])
    probs = softmax_with_temperature(q_values, temperature=1.0)

    # Check probabilities sum to 1
    assert np.isclose(np.sum(probs), 1.0)

    # Check illegal actions have probability 0
    assert probs[1] == 0.0
    assert probs[3] == 0.0

    # Check only legal actions have non-zero probability
    assert probs[0] > 0
    assert probs[2] > 0
    assert probs[4] > 0

    # Check highest Q-value has highest probability
    assert probs[2] > probs[4] > probs[0]


def test_softmax_temperature_low():
    """Test that low temperature approaches argmax (greedy)."""
    q_values = np.array([1.0, 2.0, 5.0, 3.0])
    probs = softmax_with_temperature(q_values, temperature=0.01)

    # With very low temperature, should put almost all probability on max Q
    assert probs[2] > 0.99
    assert np.sum(probs) <= 1.0


def test_softmax_temperature_high():
    """Test that high temperature approaches uniform distribution."""
    q_values = np.array([1.0, 2.0, 3.0, 4.0])
    probs = softmax_with_temperature(q_values, temperature=100.0)

    # With very high temperature, should be nearly uniform
    expected_uniform = 0.25
    for prob in probs:
        assert np.isclose(prob, expected_uniform, atol=0.01)


def test_softmax_all_illegal_raises_error():
    """Test that all illegal actions raises an error."""
    q_values = np.array([-np.inf, -np.inf, -np.inf])

    with pytest.raises(ValueError, match="All actions are illegal"):
        softmax_with_temperature(q_values, temperature=1.0)


def test_softmax_equal_q_values():
    """Test that equal Q-values produce uniform distribution over legal actions."""
    q_values = np.array([2.0, -np.inf, 2.0, 2.0, -np.inf])
    probs = softmax_with_temperature(q_values, temperature=1.0)

    # Three legal actions with equal Q should have equal probability
    expected_prob = 1.0 / 3.0
    assert np.isclose(probs[0], expected_prob)
    assert np.isclose(probs[2], expected_prob)
    assert np.isclose(probs[3], expected_prob)

    # Illegal actions should have probability 0
    assert probs[1] == 0.0
    assert probs[4] == 0.0


def test_state_representation_size():
    """Test that state representation has correct size after turn/river bug fix."""
    actions = [-1, 0, 1, 2, 3]
    n_players = 3

    agent = Agent(player_index=0, actions=actions, n_players=n_players)

    # For 3 players: 18 + 2*(3-1) = 22 features
    # Breakdown:
    # - game_stage(1) + hole_cards(4) + own_wealth(1) + own_bets(1) + pot_size(1) = 8
    # - public_cards(10) = 5 cards × 2 features
    # - opponent_wealths(2) + opponent_active(2) = 4
    # Total: 8 + 10 + 4 = 22
    expected_state_size = 22
    assert agent.len_private_state == expected_state_size, \
        f"Expected state size {expected_state_size}, got {agent.len_private_state}"

    # Verify state size in practice
    game_state = State(n_players=n_players)
    private_state = agent.get_private_state(game_state)
    assert len(private_state) == expected_state_size


def test_own_wealth_index():
    """Test that own_wealth is at index 5 (critical for pre-training)."""
    actions = [-1, 0, 1, 2, 3]
    n_players = 3

    agent = Agent(player_index=0, actions=actions, n_players=n_players)
    game_state = State(n_players=n_players, initial_wealth=50)

    private_state = agent.get_private_state(game_state)

    # Index breakdown:
    # 0: game_stage
    # 1-4: hole_cards (2 cards × 2 features)
    # 5: own_wealth ← THIS IS CRITICAL!
    # 6: own_bets
    # 7: pot_size
    # 8-17: public_cards (5 cards × 2 features)
    # 18-19: opponent_wealths
    # 20-21: opponent_active

    WEALTH_INDEX = 5
    assert private_state[WEALTH_INDEX] == game_state.wealth[agent.player_index], \
        f"own_wealth should be at index {WEALTH_INDEX}"


def test_public_cards_all_stages():
    """Test that all 5 public cards are encoded correctly across game stages."""
    from poker.cards import Card, Rank, Suit

    actions = [-1, 0, 1, 2, 3]
    n_players = 3
    agent = Agent(player_index=0, actions=actions, n_players=n_players)

    # Test PRE_FLOP: no public cards visible
    game_state = State(n_players=n_players)
    assert game_state.game_stage == GameStage.PRE_FLOP
    private_state = agent.get_private_state(game_state)

    # Indices 8-17 should all be -1 (no cards visible)
    public_cards_slice = private_state[8:18]
    assert all(val == -1 for val in public_cards_slice), \
        "Pre-flop should have all public cards as -1"

    # Test FLOP: 3 cards visible
    game_state.game_stage = GameStage.FLOP
    game_state.public_cards = [
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.KING, Suit.SPADES),
        Card(Rank.QUEEN, Suit.DIAMONDS),
    ]
    private_state = agent.get_private_state(game_state)

    # First 3 cards should be encoded
    assert private_state[8] == Rank.ACE
    assert private_state[9] == Suit.HEARTS
    assert private_state[10] == Rank.KING
    assert private_state[11] == Suit.SPADES
    assert private_state[12] == Rank.QUEEN
    assert private_state[13] == Suit.DIAMONDS
    # Last 2 cards should be -1 (not dealt yet)
    assert private_state[14] == -1
    assert private_state[15] == -1
    assert private_state[16] == -1
    assert private_state[17] == -1

    # Test TURN: 4 cards visible
    game_state.game_stage = GameStage.TURN
    game_state.public_cards.append(Card(Rank.JACK, Suit.CLUBS))
    private_state = agent.get_private_state(game_state)

    # First 4 cards should be encoded
    assert private_state[8] == Rank.ACE
    assert private_state[14] == Rank.JACK
    assert private_state[15] == Suit.CLUBS
    # Last card should be -1 (river not dealt yet)
    assert private_state[16] == -1
    assert private_state[17] == -1

    # Test RIVER: all 5 cards visible
    game_state.game_stage = GameStage.RIVER
    game_state.public_cards.append(Card(Rank.TEN, Suit.HEARTS))
    private_state = agent.get_private_state(game_state)

    # All 5 cards should be encoded (no -1 values)
    assert private_state[8] == Rank.ACE
    assert private_state[14] == Rank.JACK
    assert private_state[16] == Rank.TEN
    assert private_state[17] == Suit.HEARTS
    # No -1 values in public cards
    public_cards_slice = private_state[8:18]
    assert -1 not in public_cards_slice, \
        "River stage should have all 5 public cards encoded (no -1 values)"
