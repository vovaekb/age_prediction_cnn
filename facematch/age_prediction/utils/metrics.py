from keras import backend as K


def earth_movers_distance(y_true, y_pred):
    """
    Calculates the Earth Movers Distance (EMD) between two probability distributions.

    Parameters:
    - y_true (Tensor): True probability distribution.
    - y_pred (Tensor): Predicted probability distribution.

    Returns:
    - Tensor: The mean EMD between the true and predicted distributions.
    """
    cdf_true = K.cumsum(y_true, axis=-1)
    cdf_pred = K.cumsum(y_pred, axis=-1)
    emd = K.sqrt(K.mean(K.square(cdf_true - cdf_pred), axis=-1))
    return K.mean(emd)


def age_mae(y_true, y_pred):
    """
    Calculate the mean absolute error (MAE) between the true age and the predicted age.

    Parameters:
    - y_true: The true age values. A tensor of shape (batch_size, 100).
    - y_pred: The predicted age values. A tensor of shape (batch_size, 100).

    Returns:
    - mae: The mean absolute error between the true age and the predicted age. A scalar value.
    """
    true_age = K.sum(y_true * K.arange(0, 100, dtype="float32"), axis=-1)
    pred_age = K.sum(y_pred * K.arange(0, 100, dtype="float32"), axis=-1)
    mae = K.mean(K.abs(true_age - pred_age))
    return mae
