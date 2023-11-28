"""
This file defines an adversary that can be used to attack a mean estimation algorithm.
"""

import numpy as np
from abc import ABC, abstractmethod


class Adversary(ABC):
    """
    This class defines an adversary that can be used to attack a mean estimation algorithm.
    """

    def __init__(self, true_mu: float, true_sigma: float, epsilon: float):
        """
        This method initializes the adversary.

        :param mean: The mean of the adversary's distribution.
        :param std: The standard deviation of the adversary's distribution.
        """

        self.true_mu = true_mu
        self.true_sigma = true_sigma
        self.epsilon = epsilon

    @abstractmethod
    def _corrupt_sample(self, sample: np.ndarray) -> np.ndarray:
        """
        This method returns a sample from the adversary's distribution.

        :return: A sample from the adversary's distribution.
        """
        pass

    def corrupt_sample(self, sample: np.ndarray) -> np.ndarray:
        n, _d = sample.shape
        corrupted_samples = self._corrupt_sample(sample)

        assert (
            np.all(corrupted_samples == sample, axis=0).sum()
            >= (1.0 - self.epsilon) * n
        ), "Adversary corrupted too many samples."

        return corrupted_samples
