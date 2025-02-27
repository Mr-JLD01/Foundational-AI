import numpy as np
from .ActivationFunction import ActivationFunction

class Sigmoid(ActivationFunction):

    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x) * (1 - self.forward(x))