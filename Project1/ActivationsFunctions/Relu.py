import numpy as np
from .ActivationFunction import ActivationFunction

class Relu(ActivationFunction):

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda element: max(0, element))(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        # I chose to evaluate the point discontinuity at 0 as 0
        return np.vectorize(lambda element: 1 if element > 0 else 0)(x)