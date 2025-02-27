import numpy as np
from .ActivationFunction import ActivationFunction

class Softmax(ActivationFunction):

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x) / np.exp(x).sum(axis=0)

    # TODO handle batch
    def derivative(self, x: np.ndarray) -> np.ndarray:
        softmax_computation = self.forward(x)
        return np.diagflat(softmax_computation) - (softmax_computation @ softmax_computation.T)