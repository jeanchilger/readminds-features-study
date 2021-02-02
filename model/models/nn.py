"""
Neural Network package (we needed a name for the file...)
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
