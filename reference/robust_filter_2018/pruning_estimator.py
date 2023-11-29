import numpy as np

from robust_mean_estimator import RobustMeanEstimator


class PruningEstimator(RobustMeanEstimator):
    """
    Estimator for the mean by pruning points that are far away.
    """

    def _estimate(self, sample: np.ndarray, epsilon: float) -> np.ndarray:
        n, d = sample.shape

        coord_med = np.zeros(d)
        for i in range(d):
            coord_med[i] = np.median(sample[:, i])

        filtered_points = np.zeros(sample.shape)
        num_filtered_points = 0
        K = 5
        threshold = K * np.sqrt(float(d) * np.log(float(n) / epsilon))

        for i in range(n):
            if np.linalg.norm(sample[i] - coord_med) < threshold:
                num_filtered_points += 1
                filtered_points[num_filtered_points] = sample[i]

        filtered_points = filtered_points[:num_filtered_points]

        return np.mean(filtered_points, axis=0)
