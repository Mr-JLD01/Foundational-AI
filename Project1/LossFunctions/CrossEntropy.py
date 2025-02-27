import numpy as np
from .LossFunction import LossFunction

class CrossEntropy(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        #NOTE - The commented out error matrix was to test if it worked with the mpg dataset since I can get that to work fine
        # error_matrix: np.ndarray = y_true * np.log(np.abs(y_pred))
        error_matrix: np.ndarray = y_true * np.log(np.abs(y_pred))
        sum_error: np.ndarray = np.sum(error_matrix, axis=0)
        return np.average((-sum_error) / len(y_true))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_true = np.reshape(y_true, y_pred.shape)
        return - y_true / y_pred