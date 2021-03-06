import numpy as np


def compute_rmse(y_pred, y_true):
    """ compute RMSE metric """
    return -np.sqrt(((y_pred - y_true) ** 2).mean())
