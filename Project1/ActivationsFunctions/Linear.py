import numpy as np
from .ActivationFunction import ActivationFunction

class Linear(ActivationFunction):
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones(x.shape)