import numpy as np
from .ActivationFunction import ActivationFunction

class Tanh(ActivationFunction):

    def forward(self, x: np.ndarray) -> np.ndarray:
        return (np.exp(x) - np.exp(-1 * x)) / (np.exp(x) + np.exp(-1 * x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - self.forward(x) * self.forward(x)