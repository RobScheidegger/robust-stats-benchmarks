import numpy as np
from robust_mean_estimator import RobustMeanEstimator


class QueFilterEstimator(RobustMeanEstimator):
    """
    Filtering estimator based on the code from the following paper: TODO
    """

    def _estimate(self, sample: np.ndarray, epsilon: float) -> np.ndarray:
        return np.median(sample, axis=0)
