import numpy as np
from scipy.linalg import svd
from scipy import special

from robust_mean_estimator import RobustMeanEstimator


class Filter2018Estimator(RobustMeanEstimator):
    """
    Estimator for the mean by pruning points that are far away.
    """

    def _estimate(self, sample: np.ndarray, epsilon: float) -> np.ndarray:
        n, d = sample.shape

        empirical_mean = np.mean(sample, axis=0)
        threshold = epsilon * np.log(1.0 / epsilon)  # TODO: Check
        centered_data = (sample - empirical_mean) / np.sqrt(n)
        cher = 2

        U, S, _ = svd(centered_data.T, full_matrices=False)

        lambda_ = S[0] ** 2
        v = U[:, 0]

        # If the largest eigenvalue is about right, just return
        if lambda_ < 1 + 3 * threshold:
            return empirical_mean

        # Otherwise, project in direction of v and filter
        delta = 2 * epsilon
        projected_data1 = (
            sample * v
        )  # TODO: Is this coordinate-wise or matrix multiplication?
        med = np.median(projected_data1)

        print(projected_data1.shape, med.shape, sample.shape, v.shape)
        projected_data = np.concatenate([np.abs(projected_data1 - med), sample], axis=1)

        sorted_projected_data = projected_data[
            projected_data[:, 0].argsort()
        ]  # TODO: Questionable
        for i in range(n):
            T = sorted_projected_data[i, 0] - delta
            if (n - i) > 0.5 * n * (1 - special.erf(T / np.sqrt(2)) + threshold):
                break

        if i == 0 or i == n:
            return empirical_mean

        return self._estimate(sorted_projected_data[:i, 1:], epsilon)
