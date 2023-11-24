from abc import abstractmethod
import numpy as np


class RobustMeanEstimator:
    def __init__(self, data, mu: np.ndarray, sigma: np.ndarray, epsilon: float):
        self.data = data
        self.mu = mu
        self.sigma = sigma
        self.epsilon = epsilon

        assert len(self.mu.shape) == 1, "The mean must be a vector."
        self.d = mu.shape[0]

    def load_data(self, data: np.ndarray):
        """
        This method loads the data.

        :param data: The data to load.
        """
        self.data = data

    @abstractmethod
    def _estimate(self) -> np.ndarray:
        """
        This method estimates the mean of the data.

        :param epsilon: The privacy parameter.
        :return: The estimated mean.
        """
        pass

    def estimate_mean(self) -> np.ndarray:
        self.estimation = self._estimate()
        assert self.estimation.shape == self.mu.shape, (
            "The estimated mean must be a vector: "
            + str(self.estimation.shape)
            + " vs "
            + str(self.mu.shape)
        )
        return self.estimation

    def loss(self) -> float:
        """
        This method computes the loss of the estimator.

        :param mu: The mean to compute the loss at.
        :return: The loss.
        """
        return np.mean((self.estimation - self.mu)) ** 2
