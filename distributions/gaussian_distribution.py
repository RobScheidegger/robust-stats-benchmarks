import numpy as np
from distribution import Distribution


class GaussianDistribution(Distribution):
    def __init__(self, mu: np.ndarray[float], sigma: np.ndarray[float]) -> None:
        """
        This method initializes the distribution.

        :param mu: The mean of the distribution.
        :param stddev: The standard deviation of the distribution.
        """

        super().__init__(mu, sigma)

    def sample(self, n: int, d: int) -> np.ndarray:
        # Sample from a Gaussian distribution
        assert self.mu.shape == (d,), "Actual:" + str(self.mu.shape)
        assert self.sigma.shape == (d,)
        return np.random.normal(self.mu, self.sigma, (n, d))
