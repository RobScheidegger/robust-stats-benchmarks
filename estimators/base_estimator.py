import numpy as np
from robust_mean_estimator import RobustMeanEstimator


class BaseEstimator(RobustMeanEstimator):
    """
    Trivial base emulator that performs the traditional mean computation (no robustness).
    """

    def _estimate(self, samples: np.ndarray, epsilon: float) -> np.ndarray:
        return np.mean(samples, axis=0)
