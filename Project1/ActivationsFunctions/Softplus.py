import numpy as np
from .ActivationFunction import ActivationFunction

class Softplus(ActivationFunction):

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.log(1 + np.exp(x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-1 * x))