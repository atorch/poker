"""
Test suite for SkillfulRandomAgent behavior.

Verifies that:
1. Strong hands (with aces, high pairs) rarely fold
2. Weak hands fold with normal probability
3. Only legal actions are selected
"""
import pytest
from poker.skillful_random_agent import SkillfulRandomAgent
from poker.state import State
from poker.cards import Card, Rank, Suit


def test_is_strong_hand_with_ace():
    """Test that any hand with an ace is considered strong."""
    agent = SkillfulRandomAgent(player_index=0)

    # Ace-King
    cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)]
    assert agent.is_strong_hand(cards) == True

    # Ace-2 (worst ace hand)
    cards = [Card(Rank.ACE, Suit.CLUBS), Card(Rank.TWO, Suit.DIAMONDS)]
    assert agent.is_strong_hand(cards) == True

    # Two aces
    cards = [Card(Rank.ACE, Suit.HEARTS), Card(Rank.ACE, Suit.DIAMONDS)]
    assert agent.is_strong_hand(cards) == True


def test_is_strong_hand_with_high_pairs():
    """Test that pairs of tens or better are considered strong."""
    agent = SkillfulRandomAgent(player_index=0)

    # Pocket Kings
    cards = [Card(Rank.KING, Suit.HEARTS), Card(Rank.KING, Suit.DIAMONDS)]
    assert agent.is_strong_hand(cards) == True

    # Pocket Queens
    cards = [Card(Rank.QUEEN, Suit.SPADES), Card(Rank.QUEEN, Suit.CLUBS)]
    assert agent.is_strong_hand(cards) == True

    # Pocket Jacks
    cards = [Card(Rank.JACK, Suit.HEARTS), Card(Rank.JACK, Suit.SPADES)]
    assert agent.is_strong_hand(cards) == True

    # Pocket Tens
    cards = [Card(Rank.TEN, Suit.DIAMONDS), Card(Rank.TEN, Suit.CLUBS)]
    assert agent.is_strong_hand(cards) == True


def test_is_not_strong_hand_low_pairs():
    """Test that low pairs are not considered strong."""
    agent = SkillfulRandomAgent(player_index=0)

    # Pocket Nines
    cards = [Card(Rank.NINE, Suit.HEARTS), Card(Rank.NINE, Suit.DIAMONDS)]
    assert agent.is_strong_hand(cards) == False

    # Pocket Deuces
    cards = [Card(Rank.TWO, Suit.SPADES), Card(Rank.TWO, Suit.CLUBS)]
    assert agent.is_strong_hand(cards) == False


def test_is_not_strong_hand_weak():
    """Test that weak unpaired hands are not considered strong."""
    agent = SkillfulRandomAgent(player_index=0)

    # 7-2 offsuit (worst hand)
    cards = [Card(Rank.SEVEN, Suit.HEARTS), Card(Rank.TWO, Suit.DIAMONDS)]
    assert agent.is_strong_hand(cards) == False

    # King-Queen (decent but not strong by our definition)
    cards = [Card(Rank.KING, Suit.SPADES), Card(Rank.QUEEN, Suit.HEARTS)]
    assert agent.is_strong_hand(cards) == False


def test_only_selects_legal_actions():
    """Test that SkillfulRandomAgent only selects legal actions."""
    agent = SkillfulRandomAgent(player_index=0)
    state = State(n_players=3, initial_wealth=20.0)

    # Run 100 iterations to ensure we cover different random choices
    for _ in range(100):
        action = agent.get_action(state)

        min_bet = state.minimum_legal_bet()
        max_bet = state.maximum_legal_bet()

        # Verify action is legal
        assert action < 0 or (min_bet <= action <= max_bet), \
            f"Illegal action {action}, legal range: fold or [{min_bet}, {max_bet}]"


def test_strong_hands_rarely_fold():
    """Test that strong hands fold with low probability (5%)."""
    agent = SkillfulRandomAgent(player_index=0, strong_hand_fold_prob=0.05)

    # Create a state with pocket aces for player 0
    state = State(n_players=3, initial_wealth=20.0)
    state.hole_cards[0] = [Card(Rank.ACE, Suit.SPADES), Card(Rank.ACE, Suit.HEARTS)]

    # After blinds, next player needs to act
    # Pocket aces should rarely fold
    fold_count = 0
    n_trials = 1000

    for _ in range(n_trials):
        action = agent.get_action(state)
        if action < 0:  # Fold
            fold_count += 1

    fold_rate = fold_count / n_trials

    # With 5% fold probability, expect ~5% folds (allow some variance)
    # Use generous bounds for statistical test: [0%, 15%]
    assert fold_rate < 0.15, \
        f"Strong hand folded too often: {fold_rate:.1%} (expected ~5%)"


def test_weak_hands_fold_normally():
    """Test that weak hands can fold with normal probability."""
    agent = SkillfulRandomAgent(player_index=0)

    # Create a state with 7-2 offsuit (worst hand) for player 0
    state = State(n_players=3, initial_wealth=20.0)
    state.hole_cards[0] = [Card(Rank.SEVEN, Suit.HEARTS), Card(Rank.TWO, Suit.DIAMONDS)]

    # Weak hands should be able to fold (not restricted)
    fold_count = 0
    n_trials = 500

    for _ in range(n_trials):
        action = agent.get_action(state)
        if action < 0:  # Fold
            fold_count += 1

    fold_rate = fold_count / n_trials

    # With uniform random selection over legal actions, expect some folds
    # Just verify we're not preventing folds on weak hands (fold_rate > 0)
    assert fold_rate > 0, "Weak hands should be able to fold"


def test_respects_betting_constraints():
    """Test that SkillfulRandomAgent respects min/max betting constraints."""
    agent = SkillfulRandomAgent(player_index=0)

    # Create a state with specific betting constraints
    state = State(n_players=3, initial_wealth=10.0)

    # Run many trials to test constraint compliance
    for _ in range(200):
        action = agent.get_action(state)

        min_bet = state.minimum_legal_bet()
        max_bet = state.maximum_legal_bet()

        if action >= 0:  # Non-fold action
            assert min_bet <= action <= max_bet, \
                f"Bet {action} outside legal range [{min_bet}, {max_bet}]"
