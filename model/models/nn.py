"""
Neural Network package.

Assembles the creation of all our models in simple and
easy-to-use function calls.
"""

from tensorflow import keras


def create_model(input_size, output_size):
    """Creates a simple, "default" nn model.

    The neural network have a single hidden layer with 16 neurons.

    Returns:
        keras.Sequential: Compiled model.
    """

    model = keras.Sequential()

    model.add(keras.Input(shape=input_size))
    model.add(keras.layers.Dense(16, activation="relu"))
    model.add(keras.layers.Dense(output_size, activation="softmax"))

    model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])

    return model


def create_rnn_model(input_size, output_size):
    """Creates a simple RNN model.

    Returns:
        keras.Sequential: Compiled model.
    """

    model = keras.Sequential()

    model.add(keras.layers.Embedding(
            input_dim=input_size,
            output_dim=int(16)))

    # With all these layers or with the uncommented one
    # the results were the same :(.
    # model.add(keras.layers.SimpleRNN(16, return_sequences=True))
    # model.add(keras.layers.SimpleRNN(64, return_sequences=True))
    # model.add(keras.layers.SimpleRNN(64, return_sequences=True))
    model.add(keras.layers.SimpleRNN(32))
    model.add(keras.layers.Dense(output_size, activation="softmax"))

    model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])

    return model
