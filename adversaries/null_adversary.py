from adversary import Adversary
import numpy as np


class NullAdversary(Adversary):
    def __init__(self, true_mu: float, true_sigma: float, epsilon: float):
        """
        This method initializes the adversary.

        :param mean: The mean of the adversary's distribution.
        :param std: The standard deviation of the adversary's distribution.
        """

        super().__init__(true_mu, true_sigma, epsilon)

    def corrupt_sample(self, sample: np.ndarray) -> np.ndarray:
        # For the null adversary, don't corrupt the sample at all (ground truth)
        return sample
