from abc import abstractmethod
import numpy as np


class Distribution:
    """
    Class representing some probability distribution.
    """

    def __init__(self, mu: np.ndarray[float], stddev: np.ndarray[float]) -> None:
        """
        This method initializes the distribution.

        :param mu: The mean of the distribution.
        :param stddev: The standard deviation of the distribution.
        """

        self.mu = mu
        self.stddev = stddev

    @abstractmethod
    def sample(self, n: int, d: int) -> np.ndarray:
        """
        This method returns a sample from the distribution.

        :param n: The number of samples to return.
        :param d: The dimension of each sample.
        :return: A sample from the distribution.
        """
        raise NotImplementedError()
