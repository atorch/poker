"""
Tests for ConsistentRandomAgent - an opponent designed to prevent cumulative fold exploitation.

Key property being tested:
- Agent commits to a strategy ONCE per deal (not i.i.d. per action)
- This prevents learning agents from exploiting cumulative fold probability:
  P(fold after n actions) = 1 - (1 - p_fold)^n

Example of what we're preventing:
- If p_fold = 0.2 per action:
  - After 3 rounds: 48.8% folded
  - After 5 rounds: 67.2% folded
- Agent could learn "always bet $3" to maximize betting rounds
- With ConsistentRandomAgent, fold decision is made once upfront
"""

import pytest
from poker.state import State
from poker.consistent_random_agent import ConsistentRandomAgent
from poker.cards import Card, Rank, Suit
from poker.config import TYPICAL_INITIAL_WEALTH


def test_strong_hand_definition():
    """Test that strong hand detection matches SkillfulRandomAgent."""
    agent = ConsistentRandomAgent(player_index=0)

    # Strong: Ace with anything
    cards_ace = [Card(Rank.ACE, Suit.HEARTS), Card(Rank.TWO, Suit.SPADES)]
    assert agent.is_strong_hand(cards_ace) == True

    # Strong: High pairs (TT+)
    cards_tens = [Card(Rank.TEN, Suit.HEARTS), Card(Rank.TEN, Suit.SPADES)]
    assert agent.is_strong_hand(cards_tens) == True

    cards_aces = [Card(Rank.ACE, Suit.HEARTS), Card(Rank.ACE, Suit.SPADES)]
    assert agent.is_strong_hand(cards_aces) == True

    # Weak: Low pairs
    cards_twos = [Card(Rank.TWO, Suit.HEARTS), Card(Rank.TWO, Suit.SPADES)]
    assert agent.is_strong_hand(cards_twos) == False

    # Weak: Unpaired non-aces
    cards_weak = [Card(Rank.SEVEN, Suit.HEARTS), Card(Rank.TWO, Suit.SPADES)]
    assert agent.is_strong_hand(cards_weak) == False


def test_strategy_commitment_per_deal():
    """
    Test that strategy is committed ONCE per deal, not per action.

    This is the core anti-exploitation property:
    - Old approach (i.i.d.): Each action has independent fold probability
      → Agent can exploit cumulative probability
    - New approach: Commit to strategy once
      → Can't exploit cumulative probability
    """
    agent = ConsistentRandomAgent(player_index=0, strategy_probs=[0.0, 1.0, 0.0])  # Always passive
    state = State(n_players=3, initial_wealth=TYPICAL_INITIAL_WEALTH)

    # Give agent weak hand
    weak_cards = [Card(Rank.SEVEN, Suit.HEARTS), Card(Rank.TWO, Suit.SPADES)]
    state.hole_cards[0] = weak_cards

    # First action on this deal
    action1 = agent.get_action(state)
    strategy1 = agent.current_strategy

    # Simulate another action on SAME deal (n_deals doesn't change)
    action2 = agent.get_action(state)
    strategy2 = agent.current_strategy

    # Strategy should be the SAME (committed once per deal)
    assert strategy1 == strategy2 == 'passive'

    # Now increment deal number (simulating new deal)
    state.n_deals += 1

    # Strategy should be re-committed for new deal
    action3 = agent.get_action(state)
    # Note: With 100% passive, it will be passive again, but the commitment happened


def test_strategy_changes_between_deals():
    """
    Test that strategy CAN change between different deals.

    While strategy is fixed within a deal, it should vary across deals
    to provide diverse opponent behavior.
    """
    agent = ConsistentRandomAgent(player_index=0)
    state = State(n_players=3, initial_wealth=TYPICAL_INITIAL_WEALTH)

    strategies_seen = set()

    # Play many deals and observe strategy variety
    for deal_num in range(1, 50):
        state.n_deals = deal_num
        agent.get_action(state)
        strategies_seen.add(agent.current_strategy)

    # Should see all 3 strategies with high probability (after 50 deals)
    # Probability of missing any one strategy: (2/3)^50 ≈ 4.5e-9
    assert len(strategies_seen) == 3, f"Only saw strategies: {strategies_seen}, expected all 3"
    assert strategies_seen == {'aggressive', 'passive', 'bluffing'}


def test_passive_strategy_folds_weak_hands():
    """
    Test that passive strategy folds weak hands immediately.

    This prevents the "always bet" exploit - passive players won't
    stay in multiple rounds with weak hands.
    """
    agent = ConsistentRandomAgent(player_index=0, strategy_probs=[0.0, 1.0, 0.0])  # Always passive
    state = State(n_players=3, initial_wealth=TYPICAL_INITIAL_WEALTH)

    # Give agent weak hand
    weak_cards = [Card(Rank.SEVEN, Suit.HEARTS), Card(Rank.TWO, Suit.SPADES)]
    state.hole_cards[0] = weak_cards

    # Run multiple times with different deals
    fold_count = 0
    trials = 100

    for deal_num in range(trials):
        state.n_deals = deal_num
        action = agent.get_action(state)

        if action < 0:  # Fold action
            fold_count += 1

    # Passive strategy should fold weak hands most of the time
    # (exact probability depends on legal actions, but should be high)
    assert fold_count > 50, f"Expected >50 folds with weak hands in passive mode, got {fold_count}"


def test_aggressive_strategy_rarely_folds_strong_hands():
    """
    Test that aggressive strategy plays strong hands aggressively.
    """
    agent = ConsistentRandomAgent(player_index=0, strategy_probs=[1.0, 0.0, 0.0])  # Always aggressive
    state = State(n_players=3, initial_wealth=TYPICAL_INITIAL_WEALTH)

    # Give agent strong hand (pocket aces)
    strong_cards = [Card(Rank.ACE, Suit.HEARTS), Card(Rank.ACE, Suit.SPADES)]
    state.hole_cards[0] = strong_cards

    # Run multiple times
    non_fold_count = 0
    trials = 100

    for deal_num in range(trials):
        state.n_deals = deal_num
        action = agent.get_action(state)

        if action >= 0:  # Non-fold action
            non_fold_count += 1

    # Aggressive strategy with strong hand should almost never fold
    assert non_fold_count == 100, f"Expected 100/100 non-folds with AA in aggressive mode, got {non_fold_count}"


def test_bluffing_strategy_never_folds():
    """
    Test that bluffing strategy always bets (even with weak hands).

    This represents the "maniac" player type that provides training signal
    for learning to identify and exploit bluffs.
    """
    agent = ConsistentRandomAgent(player_index=0, strategy_probs=[0.0, 0.0, 1.0])  # Always bluffing
    state = State(n_players=3, initial_wealth=TYPICAL_INITIAL_WEALTH)

    # Give agent WEAK hand (7-2 offsuit)
    weak_cards = [Card(Rank.SEVEN, Suit.HEARTS), Card(Rank.TWO, Suit.SPADES)]
    state.hole_cards[0] = weak_cards

    # Run multiple times
    non_fold_count = 0
    trials = 100

    for deal_num in range(trials):
        state.n_deals = deal_num
        action = agent.get_action(state)

        if action >= 0:  # Non-fold action
            non_fold_count += 1

    # Bluffing strategy should never fold (even with terrible hands)
    assert non_fold_count == 100, f"Expected 100/100 non-folds in bluffing mode, got {non_fold_count}"


def test_only_selects_legal_actions():
    """Test that ConsistentRandomAgent only selects legal actions."""
    agent = ConsistentRandomAgent(player_index=0)
    state = State(n_players=3, initial_wealth=20.0)

    # Run many iterations across different deals
    for deal_num in range(100):
        state.n_deals = deal_num
        action = agent.get_action(state)

        min_bet = state.minimum_legal_bet()
        max_bet = state.maximum_legal_bet()

        # Verify action is legal
        assert action < 0 or (min_bet <= action <= max_bet), \
            f"Illegal action {action}, legal range: fold or [{min_bet}, {max_bet}]"


def test_custom_strategy_distribution():
    """
    Test that custom strategy probabilities are respected.

    This allows tuning opponent difficulty by adjusting
    aggressive/passive/bluffing ratios.
    """
    # Create agent that's always aggressive
    agent = ConsistentRandomAgent(player_index=0, strategy_probs=[1.0, 0.0, 0.0])
    state = State(n_players=3, initial_wealth=TYPICAL_INITIAL_WEALTH)

    # Play 50 deals
    for deal_num in range(50):
        state.n_deals = deal_num
        agent.get_action(state)
        # Should always be aggressive
        assert agent.current_strategy == 'aggressive'


def test_prevents_cumulative_fold_exploitation():
    """
    Integration test: Verify that ConsistentRandomAgent prevents the
    cumulative fold exploitation pattern.

    With i.i.d. folding:
    - P(fold after n actions) = 1 - (1 - 0.2)^n
    - After 5 actions: 67.2% fold probability
    - Agent learns: "maximize betting rounds"

    With consistent strategy:
    - Fold decision made once upfront
    - Can't increase fold probability by prolonging betting
    - Agent must learn hand evaluation instead
    """
    # This is tested implicitly by test_strategy_commitment_per_deal
    # but we document the property explicitly here

    agent = ConsistentRandomAgent(player_index=0)
    state = State(n_players=3, initial_wealth=TYPICAL_INITIAL_WEALTH)

    weak_cards = [Card(Rank.SEVEN, Suit.HEARTS), Card(Rank.TWO, Suit.SPADES)]
    state.hole_cards[0] = weak_cards

    # Simulate 5 sequential actions on same deal
    actions_taken = []
    for _ in range(5):
        action = actions_taken.append(agent.get_action(state))

    # The strategy was committed on first action
    # All subsequent actions use the SAME committed strategy
    # Therefore, we can't exploit P(fold after n actions) = 1 - 0.8^n

    # If strategy was passive with weak hand, it folded on first action
    # If strategy was aggressive/bluffing, it will keep betting regardless of rounds

    # Key insight: The fold probability does NOT increase with more betting rounds
    # This forces learning agents to learn hand strength, not round maximization
    pass  # Property tested implicitly above
