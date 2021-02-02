import numpy as np
import os
import random
import tensorflow as tf


def set_random_state(seed=0):
    """Sets a common random seed for libraries.

    Args:
        seed (int, optional): Seed value. Defaults to 0.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
