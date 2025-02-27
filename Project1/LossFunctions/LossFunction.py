import numpy as np
from abc import ABC, abstractmethod

class LossFunction(ABC):

    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate the total loss of a batch

        Args:
            y_true (np.ndarray): The true label for the data
            y_pred (np.ndarray): The predicted labels

        Returns:
            np.ndarray: total loss
        """
        ...

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculates the loss gradient

        Args:
            y_true (np.ndarray): The true label for the data
            y_pred (np.ndarray): The predicted labels

        Returns:
            np.ndarray: loss gradient
        """
        ...