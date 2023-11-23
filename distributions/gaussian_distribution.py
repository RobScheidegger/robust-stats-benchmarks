import numpy as np
from distribution import Distribution


class GaussianDistribution(Distribution):
    def __init__(self, mu: np.ndarray[float], stddev: np.ndarray[float]) -> None:
        """
        This method initializes the distribution.

        :param mu: The mean of the distribution.
        :param stddev: The standard deviation of the distribution.
        """

        super().__init__(mu, stddev)

    def sample(self, n: int, d: int) -> np.ndarray:
        # Sample from a Gaussian distribution
        return np.random.normal(self.mu, self.stddev, (n, d))
