from tensorflow.keras import losses, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Input,
)


def get_q_function_model(n_actions, n_inputs, n_units=64):

    # This is a model for Q(private_state, action), i.e. an action-value function

    input_layer = Input(shape=(n_inputs,))

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
