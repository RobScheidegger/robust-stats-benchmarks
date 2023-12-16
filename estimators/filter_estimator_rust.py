import numpy as np
from robust_mean_estimator import RobustMeanEstimator
from robust_stats.robust_stats import robust_mean_filter


class FilterRustEstimator(RobustMeanEstimator):
    """
    Trivial base emulator that performs the traditional mean computation (no robustness).
    """

    def _estimate(self, samples: np.ndarray, epsilon: float) -> np.ndarray:
        return robust_mean_filter(samples, epsilon)
