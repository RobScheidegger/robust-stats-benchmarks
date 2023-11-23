"""
This file defines an adversary that can be used to attack a mean estimation algorithm.
"""

import numpy as np
from abc import abstractmethod


class Adversary:
    """
    This class defines an adversary that can be used to attack a mean estimation algorithm.
    """

    def __init__(self, true_mu: float, true_stddev: float, epsilon: float):
        """
        This method initializes the adversary.

        :param mean: The mean of the adversary's distribution.
        :param std: The standard deviation of the adversary's distribution.
        """

        self.true_mu = true_mu
        self.true_stddev = true_stddev
        self.epsilon = epsilon

    @abstractmethod
    def corrupt_sample(self, sample: np.ndarray) -> np.ndarray:
        """
        This method returns a sample from the adversary's distribution.

        :return: A sample from the adversary's distribution.
        """
        pass
