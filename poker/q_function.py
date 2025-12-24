from tensorflow.keras import losses, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Input,
)


def get_model(n_actions, n_inputs, n_units=64, learning_rate=0.001, hidden_layers=None):
    """
    Create Q-function neural network.

    Args:
        n_actions: Number of actions (not used, kept for compatibility)
        n_inputs: Input dimension
        n_units: Default number of units per layer (used if hidden_layers not specified)
        learning_rate: Learning rate for optimizer
        hidden_layers: Tuple of hidden layer sizes, e.g., (32, 32) or (64, 64, 64, 64)
                       If None, uses default (64, 64, 64, 64) for backward compatibility

    Returns:
        Compiled Keras model
    """
    # Note: this is a model for Q(private_state, action), i.e. an action-value function

    # Default to original architecture if not specified
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
