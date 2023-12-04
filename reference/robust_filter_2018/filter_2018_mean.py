import numpy as np
from scipy import special, sparse
import matlab.engine


from robust_mean_estimator import RobustMeanEstimator


class Filter2018MATLABEstimator(RobustMeanEstimator):
    """
    Estimator for the mean by pruning points that are far away.
    """

    def __init__(self):
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath("reference/robust_filter_2018/", nargout=0)

    def _estimate(self, sample: np.ndarray, epsilon: float) -> np.ndarray:
        # Define a sample array
        input_array = matlab.double(sample.tolist())

        tau = 0.1
        cher = 2.5
        eps = epsilon
        data = input_array
        # Call the MATLAB function from the script
        result = self.eng.filterGaussianMean(data, eps, tau, cher, nargout=1)

        return np.array(result).reshape(-1)

    def cleanup(self):
        # Stop the MATLAB engine
        self.eng.quit()


class Filter2018PythonEstimator(RobustMeanEstimator):
    """
    Estimator for the mean by pruning points that are far away.
    """

    def _estimate(self, sample: np.ndarray, epsilon: float) -> np.ndarray:
        n, d = sample.shape
        empirical_mean = np.mean(sample, axis=0)
        threshold = epsilon * np.log(1.0 / epsilon)  # TODO: Check
        centered_data = (empirical_mean - sample) / np.sqrt(n)
        cher = 2.5
        tau = 0.1

        U, S, _ = sparse.linalg.svds(centered_data.T, k=1)

        lambda_ = S[0] ** 2
        v = U[:, 0]

        # If the largest eigenvalue is about right, just return
        if lambda_ < 1 + 3 * threshold:
            return empirical_mean

        # Otherwise, project in direction of v and filter
        delta = 2 * epsilon
        projected_data1 = sample @ v
        med = np.median(projected_data1)

        projected_data = np.abs(sample @ v - med)
        sort_order = projected_data.argsort()
        sorted_projected_samples = sample[sort_order]
        projected_data = projected_data[sort_order]

        I = 0
        for i in range(n):
            T = projected_data[i] - delta
            if (n - i) > (
                0.5 * cher * float(n) * special.erfc(T / np.sqrt(2))
                + epsilon / (float(d) * np.log(float(d) * epsilon / tau))
            ):
                break

            I += 1

        if I == 0 or I >= n - 1:
            return empirical_mean

        # print("Doing a thing!")
        return self._estimate(sorted_projected_samples[:I], epsilon)
