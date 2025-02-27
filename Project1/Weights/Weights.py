import numpy as np
from abc import ABC, abstractmethod

class Weights(ABC):

    @classmethod
    @abstractmethod
    def generate(cls, in_neurons: int, out_neurons: int) -> np.ndarray:
        """Generate weights matrix for given size

        Args:
            in_neurons (int): number of inputs coming in
            out_neurons (int): number of outputs for the weights

        Returns:
            np.ndarray: matrix of dimension in_neurons by out_neurons
        """
        ...