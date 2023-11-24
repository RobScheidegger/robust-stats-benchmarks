import numpy as np
from robust_mean_estimator import RobustMeanEstimator


class QueFilterEstimator(RobustMeanEstimator):
    """
    Filtering estimator based on the code from the following paper: TODO
    """

    def _estimate(self) -> np.ndarray:
        data = self.data
        return np.median(data, axis=0)
