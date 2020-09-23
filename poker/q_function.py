from tensorflow.keras import losses, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Input,
)


def get_q_function_model(n_actions, n_inputs=5, n_units=32):

    # We want a function of the form Q(private_state, action), i.e. an action-value function
    # For now, let's keep it simple and let the private state be
    #  (game_stage, first_hole_card_rank, first_hole_card_suit, second_hole_card_rank, second_hole_card_suit)
    #  Is the input simply a vector of length (5, )?
    # TODO Add public cards, total bets so far, etc to private state

    input_layer = Input(shape=(n_inputs, ))

    layer1 = Dense(n_units, activation="relu")(input_layer)
    layer2 = Dense(n_units, activation="relu")(layer1)

    output_layer = Dense(1, activation="linear")(layer2)

    model = Model(inputs=input_layer, outputs=[output_layer])

    nadam = optimizers.Nadam()

    model.compile(
        optimizer=nadam, loss=losses.mean_squared_error, metrics=["mean_squared_error"],
    )

    print(model.summary())

    return model
