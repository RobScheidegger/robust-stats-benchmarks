import numpy as np
from robust_mean_estimator import RobustMeanEstimator


class MedianEstimator(RobustMeanEstimator):
    """
    Trivial base emulator that performs the traditional mean computation (no robustness).
    """

    def _estimate(self, sample: np.ndarray, epsilon: float) -> np.ndarray:
        return np.median(sample, axis=0)
