import numpy as np
from abc import ABC, abstractmethod

class Weights(ABC):

    @classmethod
    @abstractmethod
    def generate(cls, in_neurons: int, out_neurons: int) -> np.ndarray:
        ...