import numpy as np
from .ActivationFunction import ActivationFunction
from .Tanh import Tanh
from .Softplus import Softplus

class Mish(ActivationFunction):

    def __init__(self):
        self.tanh: Tanh = Tanh()
        self.softplus: Softplus = Softplus()

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x * self.tanh.forward(self.softplus.forward(x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return self.tanh.forward(self.softplus.forward(x)) + x * self.tanh.derivative(self.softplus.forward(x)) * self.softplus.derivative(x)