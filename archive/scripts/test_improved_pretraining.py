#!/usr/bin/env python3
"""
Quick test to verify improved pre-training teaches the right concepts:
1. Q ≈ wealth (baseline)
2. Q(bet | strong hand) >> Q(fold | strong hand)
3. Q(bet | strong hand, large pot) gets extra bonus
"""

import numpy as np
from poker.agent import Agent
from poker.state import State
from poker.cards import Card, Rank, Suit
from poker.config import TYPICAL_INITIAL_WEALTH, Action

def test_pretraining():
    print("Testing improved pre-training function...\n")

    # Create agent and run pre-training
    agent = Agent(player_index=0, n_players=3, actions=[-1, 0, 1, 2, 3])
    agent.pretrain_on_wealth_heuristic(n_samples=1000, n_epochs=15, verbose=0)

    print("\n" + "="*70)
    print("VERIFICATION: Testing learned Q-values after pre-training")
    print("="*70)

    # Test 1: Strong hand (pocket aces) should have Q(bet) > Q(fold)
    print("\n[Test 1] Strong hand (AA): Q(bet) should >> Q(fold)")
    state = State(n_players=3, initial_wealth=TYPICAL_INITIAL_WEALTH)
    state.hole_cards[0] = [
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.ACE, Suit.SPADES)
    ]

    private_state = agent.get_private_state(state)
    model_input = agent.get_model_input(private_state, agent.actions)
    q_values = agent.model.predict(model_input, verbose=0)[:, 0]

    q_fold = q_values[agent.actions.index(Action.FOLD)]
    q_bet3 = q_values[agent.actions.index(Action.BET_3)]

    print(f"  Q(fold | AA) = {q_fold:.2f}")
    print(f"  Q(bet $3 | AA) = {q_bet3:.2f}")
    print(f"  Ratio: Q(bet)/Q(fold) = {q_bet3/q_fold:.2f}x")

    if q_bet3 > q_fold * 2:
        print(f"  ✓ PASS: Q(bet) is {q_bet3/q_fold:.2f}x higher than Q(fold)")
    else:
        print(f"  ✗ FAIL: Q(bet) should be >> Q(fold)")

    # Test 2: Weak hand should have Q(fold) > Q(bet) (fold is better)
    print("\n[Test 2] Weak hand (7-2): Q(fold) should > Q(bet)")
    state.hole_cards[0] = [
        Card(Rank.SEVEN, Suit.HEARTS),
        Card(Rank.TWO, Suit.SPADES)
    ]

    private_state = agent.get_private_state(state)
    model_input = agent.get_model_input(private_state, agent.actions)
    q_values = agent.model.predict(model_input, verbose=0)[:, 0]

    q_fold = q_values[agent.actions.index(Action.FOLD)]
    q_check = q_values[agent.actions.index(Action.CHECK_CALL)]
    q_bet3 = q_values[agent.actions.index(Action.BET_3)]

    print(f"  Q(fold | 7-2) = {q_fold:.2f}")
    print(f"  Q(check | 7-2) = {q_check:.2f}")
    print(f"  Q(bet $3 | 7-2) = {q_bet3:.2f}")
    print(f"  Ratio: Q(bet)/Q(fold) = {q_bet3/q_fold:.2f}x")

    if q_fold > q_bet3 and q_check >= q_bet3:
        print(f"  ✓ PASS: Q(fold) and Q(check) > Q(bet) - weak hands shouldn't bet")
    else:
        print(f"  ✗ FAIL: Weak hand should prefer fold/check over betting")

    # Test 3: Large pot with strong hand should boost Q(bet) even more
    print("\n[Test 3] AA with large pot: Q(bet) should get extra bonus")
    state.hole_cards[0] = [
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.ACE, Suit.SPADES)
    ]
    # Simulate large pot (both opponents bet $3)
    state.bets_by_stage[state.game_stage][1] = [3]
    state.bets_by_stage[state.game_stage][2] = [3]

    private_state = agent.get_private_state(state)
    pot_size = private_state[7]  # POT_SIZE_INDEX = 7
    wealth = private_state[5]  # WEALTH_INDEX = 5

    model_input = agent.get_model_input(private_state, agent.actions)
    q_values = agent.model.predict(model_input, verbose=0)[:, 0]

    q_bet3 = q_values[agent.actions.index(Action.BET_3)]

    print(f"  Current wealth: ${wealth:.1f}")
    print(f"  Pot size: ${pot_size:.1f} ({pot_size/wealth*100:.0f}% of wealth)")
    print(f"  Q(bet $3 | AA, large pot) = {q_bet3:.2f}")

    # With large pot, Q should be > wealth * 1.5
    if q_bet3 > wealth * 1.5:
        print(f"  ✓ PASS: Q(bet) exceeds wealth×1.5, showing pot awareness")
    else:
        print(f"  ✗ FAIL: Q(bet) should be boosted by large pot")

    print("\n" + "="*70)
    print("Pre-training verification complete!")
    print("="*70)

if __name__ == "__main__":
    test_pretraining()
