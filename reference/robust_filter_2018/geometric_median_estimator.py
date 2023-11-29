"""
This file contains python implementations of the geometric median estimator, as adapted from the 
referenced comparison MATLAB code.

https://github.com/hoonose/robust-filter/blob/master/comparisonCode/ransacGaussianMean.m
"""

import numpy as np

from robust_mean_estimator import RobustMeanEstimator


class GeoMedianEstimator(RobustMeanEstimator):
    """
    Estimator for the mean using the geometric median algorithm.
    """

    def _estimate(self, sample: np.ndarray, epsilon: float) -> np.ndarray:
        n, d = sample.shape

        num_iters = 100
        current_estimate = np.mean(sample, axis=0)
        for _ in range(num_iters):
            num = 0
            den = 0
            for j in range(n):
                dist_to_estimate = np.linalg.norm(current_estimate - sample[j])
                num += sample[j] / dist_to_estimate
                den += 1 / dist_to_estimate
            current_estimate = num / den

        return current_estimate
