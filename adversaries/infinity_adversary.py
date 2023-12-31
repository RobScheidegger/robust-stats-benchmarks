from adversary import Adversary
import numpy as np

INFINITY = 10**10


class InfinityAdversary(Adversary):
    def __init__(self, true_mu: float, true_sigma: float, epsilon: float):
        """
        This method initializes the adversary.

        :param mean: The mean of the adversary's distribution.
        :param std: The standard deviation of the adversary's distribution.
        """

        super().__init__(true_mu, true_sigma, epsilon)

    def _corrupt_sample(self, sample: np.ndarray) -> np.ndarray:
        # Select a random direction in R^d, and select the \epsilon-th quantile of the distribution along that direction
        # Then, set those to "infinity"
        n, d = sample.shape

        # Pick a random direction
        direction = np.random.normal(size=(d))

        # Normalize the direction
        direction /= np.linalg.norm(direction)

        # Compute the quantile
        projections = sample @ direction
        percentile = np.percentile(projections, self.epsilon * 100)

        sample[projections < percentile] = INFINITY
        return sample
