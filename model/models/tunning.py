from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch
from tensorflow import keras

from tensorflow.keras import backend as K


class TunableModel(HyperModel):
    def __init__(self, input_size, num_classes, metrics=["acc"]):
        self.input_size = input_size
        self.num_classes = num_classes
        self.metrics = metrics

    def build(self, hp):
        model = keras.Sequential()

        # Input Layer
        model.add(keras.Input(shape=self.input_size))

        # Hidden Layers
        num_layers = hp.Int("num_layers", 0, 2)
        for i in range(num_layers):
            model.add(
                    keras.layers.Dense(units=hp.Int(
                            "units_" + str(i), min_value=4,
                            max_value=32, step=4), activation="relu"))

        # Output Layer
        model.add(keras.layers.Dense(self.num_classes, activation="softmax"))

        model.compile(
                optimizer="adam", loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])

        return model
