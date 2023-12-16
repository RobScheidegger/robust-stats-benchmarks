"""
This file contains python implementations of the RANSAC algorithm for mean estimation, as adapted from the
comparison code here:

https://github.com/hoonose/robust-filter/blob/master/comparisonCode/ransacGaussianMean.m
"""

import numpy as np

from robust_mean_estimator import RobustMeanEstimator


class RANSACEstimator(RobustMeanEstimator):
    """
    Estimator for the mean using the RANSAC algorithm.
    """

    tau = 0.05
    """
    Tau parameter for the RANSAC algorithm determining fit to data.
    """

    def _estimate(self, sample: np.ndarray, epsilon: float) -> np.ndarray:
        n, d = sample.shape

        empirical_mean = np.mean(sample, axis=0)
        assert empirical_mean.shape == (d,)
        ransacN = np.ceil(2 * (d * np.log(4) + np.log(2 / self.tau)) / epsilon**2)

        if ransacN > n:
            return empirical_mean

        num_iters = 100
        thresh = (
            d
            + 2 * (np.sqrt(d * np.log(n / self.tau)) + np.log(n / self.tau))
            + epsilon**2 * (np.log(1 / epsilon)) ** 2
        )

        best_inliers = 0
        best_mean = empirical_mean
        for j in range(n):
            if np.linalg.norm(empirical_mean - sample[j]) ** 2 <= thresh:
                best_inliers += 1

        for i in range(num_iters):
            ransac_data = sample[np.random.choice(n, int(ransacN), replace=False)]
            ransac_mean = np.mean(ransac_data, axis=0)
            cur_inliers = 0
            for j in range(n):
                if np.linalg.norm(ransac_mean - sample[j]) ** 2 <= thresh:
                    cur_inliers += 1
            if cur_inliers > best_inliers:
                best_mean = ransac_mean
                best_inliers = cur_inliers

        return best_mean
