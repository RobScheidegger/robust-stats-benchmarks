from adversary import Adversary
import numpy as np


class ZeroAdversary(Adversary):
    def __init__(self, true_mu: float, true_sigma: float, epsilon: float):
        """
        This method initializes the adversary.

        :param mean: The mean of the adversary's distribution.
        :param std: The standard deviation of the adversary's distribution.
        """

        super().__init__(true_mu, true_sigma, epsilon)

    def corrupt_sample(self, sample: np.ndarray) -> np.ndarray:
        # Select a random direction in R^d, and select the \epsilon-th quantile of the distribution along that direction
        # Then, set those to zero
        n, d = sample.shape

        # Pick a random direction
        direction = np.random.normal(size=(d))

        # Normalize the direction
        direction /= np.linalg.norm(direction)

        # Compute the quantile
        projections = sample @ direction
        percentile = np.percentile(projections, self.epsilon * 100)

        sample[projections < percentile] = 0
        return sample
