from tensorflow.keras import backend as K


def recall(y_true, y_pred):
    """Computes the recall metric.

    Args:
        y_true (list): [description]
        y_pred (list): [description]

    Returns:
        float: Recall value.
    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    return true_positives / (possible_positives + K.epsilon())


def precision(y_true, y_pred):
    """Computes the precision metric.

    Args:
        y_true (list): [description]
        y_pred (list): [description]

    Returns:
        float: Precision value.
    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    return true_positives / (predicted_positives + K.epsilon())


def f1_score(y_true, y_pred):
    """Computes the f1 metric.

    Args:
        y_true (list): [description]
        y_pred (list): [description]

    Returns:
        float: F1 value.
    """

    precision_ = precision(y_true, y_pred)
    recall_ = recall(y_true, y_pred)

    return 2 * ((precision_ * recall_) / (precision_ + recall_ + K.epsilon()))
