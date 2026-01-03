from tensorflow.keras import losses, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Input,
    MultiHeadAttention,
    LayerNormalization,
    Dropout,
    Concatenate,
    GlobalAveragePooling1D,
)
import tensorflow as tf
import numpy as np


def get_transformer_model(n_inputs, learning_rate=0.001, d_model=128, num_heads=4, num_layers=2, dropout_rate=0.1):
    """
    Create a Transformer-based Q-function for poker.

    State representation breakdown (n_inputs=19 for 3 players):
        - game_stage (1)
        - hole_card_1: rank, suit (2)
        - hole_card_2: rank, suit (2)
        - own_wealth, total_bet_by_self, pot_size (3)
        - public_cards (6): 3 cards × (rank, suit) = 6 values (-1 if not revealed)
        - opponent_wealths (n_players-1 = 2)
        - opponent_active (n_players-1 = 2)
        - action (1)

    Transformer approach:
        1. Structure cards as sequence: [hole_1, hole_2, public_1, public_2, public_3]
        2. Each card token = (rank, suit) embedded to d_model dimensions
        3. Apply multi-head self-attention to learn card relationships (straights, flushes, pairs)
        4. Pool card representations (GlobalAverage)
        5. Concatenate with global features (wealth, pot, bets, action)
        6. Final MLP to output Q-value

    Args:
        n_inputs: Input dimension (must be 19 for 3 players)
        learning_rate: Learning rate for optimizer
        d_model: Transformer embedding dimension (default 128)
        num_heads: Number of attention heads (default 4)
        num_layers: Number of transformer blocks (default 2)
        dropout_rate: Dropout probability (default 0.1)

    Returns:
        Compiled Keras model
    """
    input_layer = Input(shape=(n_inputs,), name='input')

    # Extract components from flat input
    # State layout: [game_stage, hole1_rank, hole1_suit, hole2_rank, hole2_suit,
    #                wealth, bet, pot, pub1_rank, pub1_suit, pub2_rank, pub2_suit,
    #                pub3_rank, pub3_suit, opp1_wealth, opp2_wealth, opp1_active, opp2_active, action]

    # Cards: positions [1,2] (hole1), [3,4] (hole2), [8,9] (pub1), [10,11] (pub2), [12,13] (pub3)
    hole_card_1 = tf.stack([input_layer[:, 1], input_layer[:, 2]], axis=1)  # (batch, 2)
    hole_card_2 = tf.stack([input_layer[:, 3], input_layer[:, 4]], axis=1)  # (batch, 2)
    public_card_1 = tf.stack([input_layer[:, 8], input_layer[:, 9]], axis=1)  # (batch, 2)
    public_card_2 = tf.stack([input_layer[:, 10], input_layer[:, 11]], axis=1)  # (batch, 2)
    public_card_3 = tf.stack([input_layer[:, 12], input_layer[:, 13]], axis=1)  # (batch, 2)

    # Stack cards into sequence: (batch, 5, 2) where 5 = num_cards
    card_sequence = tf.stack([hole_card_1, hole_card_2, public_card_1, public_card_2, public_card_3], axis=1)

    # Embed each card (rank, suit) to d_model dimensions
    # Input: (batch, 5, 2) -> Output: (batch, 5, d_model)
    card_embeddings = Dense(d_model, activation='relu', name='card_embedding')(card_sequence)

    # Add positional encoding (fixed): hole cards vs public cards
    # Position 0,1 = hole cards (private info), Position 2,3,4 = public cards (shared info)
    position_embedding = tf.keras.layers.Embedding(
        input_dim=5,  # 5 card positions
        output_dim=d_model,
        name='position_embedding'
    )
    positions = tf.range(start=0, limit=5, delta=1)
    position_encodings = position_embedding(positions)  # (5, d_model)

    # Add positional encodings (broadcast over batch)
    x = card_embeddings + position_encodings  # (batch, 5, d_model)

    # Apply transformer blocks
    for i in range(num_layers):
        # Multi-head self-attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
            name=f'attention_{i}'
        )(x, x)

        # Skip connection + LayerNorm
        x = LayerNormalization(name=f'norm1_{i}')(x + attention_output)

        # Feed-forward network
        ffn = Dense(d_model * 2, activation='relu', name=f'ffn1_{i}')(x)
        ffn = Dropout(dropout_rate)(ffn)
        ffn = Dense(d_model, name=f'ffn2_{i}')(ffn)

        # Skip connection + LayerNorm
        x = LayerNormalization(name=f'norm2_{i}')(x + ffn)

    # Pool card representations: (batch, 5, d_model) -> (batch, d_model)
    card_features = GlobalAveragePooling1D(name='pool_cards')(x)

    # Extract global features (non-card state)
    # Positions: [0]=stage, [5,6,7]=wealth/bet/pot, [14,15]=opp_wealth, [16,17]=opp_active, [18]=action
    global_features = tf.concat([
        tf.expand_dims(input_layer[:, 0], axis=1),   # game_stage
        input_layer[:, 5:8],                         # wealth, bet, pot
        input_layer[:, 14:18],                       # opponent wealths and active status
        tf.expand_dims(input_layer[:, 18], axis=1),  # action
    ], axis=1, name='global_features')  # (batch, 8)

    # Combine card features with global features
    combined = Concatenate(name='combine')([card_features, global_features])  # (batch, d_model + 8)

    # Final MLP to output Q-value
    x = Dense(d_model, activation='relu', name='mlp1')(combined)
    x = Dropout(dropout_rate)(x)
    x = Dense(d_model // 2, activation='relu', name='mlp2')(x)
    output_layer = Dense(1, activation='linear', name='output')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    nadam = optimizers.Nadam(learning_rate=learning_rate)
    model.compile(
        optimizer=nadam,
        loss=losses.mean_squared_error,
        metrics=['mean_squared_error'],
    )

    print(model.summary())

    return model


def get_model(n_actions, n_inputs, n_units=64, learning_rate=0.001, hidden_layers=None):
    """
    Create Q-function neural network.

    Args:
        n_actions: Number of actions (not used, kept for compatibility)
        n_inputs: Input dimension
        n_units: Default number of units per layer (used if hidden_layers not specified)
        learning_rate: Learning rate for optimizer
        hidden_layers: Tuple of hidden layer sizes, e.g., (32, 32) or (64, 64, 64, 64)
                       OR string to specify architecture type:
                       - 'transformer_small': Transformer with 2 heads, 64 dim, 1 layer
                       - 'transformer': Transformer with 4 heads, 128 dim, 2 layers
                       If None, uses default (64, 64, 64, 64) for backward compatibility

    Returns:
        Compiled Keras model
    """
    # Note: this is a model for Q(private_state, action), i.e. an action-value function

    # Check if transformer architecture requested
    if isinstance(hidden_layers, str):
        if hidden_layers == 'transformer_small':
            return get_transformer_model(
                n_inputs=n_inputs,
                learning_rate=learning_rate,
                d_model=64,
                num_heads=2,
                num_layers=1,
                dropout_rate=0.1
            )
        elif hidden_layers == 'transformer':
            return get_transformer_model(
                n_inputs=n_inputs,
                learning_rate=learning_rate,
                d_model=128,
                num_heads=4,
                num_layers=2,
                dropout_rate=0.1
            )
        else:
            raise ValueError(f"Unknown architecture type: {hidden_layers}")

    # Default to original MLP architecture if not specified
    if hidden_layers is None:
        hidden_layers = (n_units, n_units, n_units, n_units)

    # TODO Would it be cleaner to have a multi-input model, give it some structure,
    #  have card array as one input, bets as another, etc?
    #  Should also try one hot encoding discrete variables like the game stage
    input_layer = Input(shape=(n_inputs,))

    # Build hidden layers dynamically
    x = input_layer
    for i, layer_size in enumerate(hidden_layers):
        x = Dense(layer_size, activation="relu", name=f"hidden_{i+1}")(x)

    output_layer = Dense(1, activation="linear", name="output")(x)

    model = Model(inputs=input_layer, outputs=[output_layer])

    nadam = optimizers.Nadam(learning_rate=learning_rate)

    model.compile(
        optimizer=nadam, loss=losses.mean_squared_error, metrics=["mean_squared_error"],
    )

    print(model.summary())

    return model
