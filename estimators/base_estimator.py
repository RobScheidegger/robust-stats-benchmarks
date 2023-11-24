import numpy as np
from robust_mean_estimator import RobustMeanEstimator


class BaseEstimator(RobustMeanEstimator):
    """
    Trivial base emulator that performs the traditional mean computation (no robustness).
    """

    def _estimate(self) -> np.ndarray:
        data = self.data
        return np.mean(data, axis=0)
