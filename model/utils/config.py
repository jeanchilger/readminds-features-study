import numpy as np
import os
import random
import tensorflow as tf
# from tensorflow.keras import backend as K


def set_random_state(seed=0):
    """[summary]

    Args:
        seed (int, optional): [description]. Defaults to 0.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)