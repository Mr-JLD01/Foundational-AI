import numpy as np
from .LossFunction import LossFunction

class SquaredError(LossFunction):

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        error_matrix: np.ndarray = y_pred - y_true
        squared_error: np.ndarray = np.vectorize(lambda element: element**2)(error_matrix)
        sum_error: np.ndarray = np.sum(squared_error, axis=0)
        return np.average((.5 * sum_error) / len(y_true))


    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_true = np.reshape(y_true, y_pred.shape)
        return y_pred - y_true