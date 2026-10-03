#!/usr/bin/env python3
"""
Quick test to verify transformer architecture works.
"""
import numpy as np
from poker.agent import Agent

def test_transformer_architecture():
    """Test that transformer models can be created and make predictions."""

    print("="*70)
    print("Testing Transformer Architecture")
    print("="*70)

    # Test 1: transformer_small
    print("\n[Test 1] Creating transformer_small agent...")
    agent_small = Agent(
        player_index=0,
        n_players=3,
        temperature=1.0,
        learning_rate=0.0003,
        hidden_layers='transformer_small'
    )
    print("✓ transformer_small created successfully")

    # Test 2: transformer (full)
    print("\n[Test 2] Creating transformer agent...")
    agent_full = Agent(
        player_index=0,
        n_players=3,
        temperature=1.0,
        learning_rate=0.0003,
        hidden_layers='transformer'
    )
    print("✓ transformer created successfully")

    # Test 3: Make predictions with transformer_small
    print("\n[Test 3] Testing transformer_small predictions...")
    # Create a fake state: 19 features
    fake_state = [
        0,  # game_stage
        12, 0,  # hole_card_1: rank, suit
        12, 1,  # hole_card_2: rank, suit (pocket aces)
        25, 2, 5,  # wealth, bet, pot
        -1, -1, -1, -1, -1, -1,  # public cards (not revealed)
        25, 25,  # opponent wealths
        1, 1,  # opponent active
    ]

    # Test prediction for each action
    actions = agent_small.actions
    model_input = agent_small.get_model_input(fake_state, actions)
    q_values = agent_small.model.predict(model_input, verbose=0)[:, 0]

    print(f"  Input shape: {model_input.shape}")
    print(f"  Q-values shape: {q_values.shape}")
    print(f"  Q-values: {q_values}")
    assert q_values.shape == (len(actions),), f"Expected {len(actions)} Q-values, got {q_values.shape}"
    print("✓ transformer_small predictions work correctly")

    # Test 4: Make predictions with transformer (full)
    print("\n[Test 4] Testing transformer predictions...")
    model_input = agent_full.get_model_input(fake_state, actions)
    q_values = agent_full.model.predict(model_input, verbose=0)[:, 0]

    print(f"  Input shape: {model_input.shape}")
    print(f"  Q-values shape: {q_values.shape}")
    print(f"  Q-values: {q_values}")
    assert q_values.shape == (len(actions),), f"Expected {len(actions)} Q-values, got {q_values.shape}"
    print("✓ transformer predictions work correctly")

    # Test 5: Training update (single step)
    print("\n[Test 5] Testing transformer training update...")
    agent_small.update_q(fake_state, action=2, updated_guess_for_q=0.5)
    print("✓ Single update works")

    # Test 6: Batch update
    print("\n[Test 6] Testing transformer batch update...")
    batch_states = np.array([fake_state, fake_state, fake_state])
    batch_actions = np.array([0, 1, 2])
    batch_targets = np.array([0.1, 0.2, 0.3])
    agent_small.update_q_batch(batch_states, batch_actions, batch_targets)
    print("✓ Batch update works")

    print("\n" + "="*70)
    print("ALL TESTS PASSED! ✓")
    print("="*70)
    print("\nTransformer architecture is ready for grid search!")


if __name__ == "__main__":
    test_transformer_architecture()
