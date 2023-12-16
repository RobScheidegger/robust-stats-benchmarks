import numpy as np
import matlab.engine


from robust_mean_estimator import RobustMeanEstimator


class CDGS20_PGDEstimator(RobustMeanEstimator):
    """
    Estimator for the mean by pruning points that are far away.
    """

    def __init__(self):
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath("reference/robust_bn_faster/", nargout=0)

    def _estimate(self, sample: np.ndarray, epsilon: float) -> np.ndarray:
        # Define a sample array
        data = matlab.double(sample.tolist())

        # Call the MATLAB function from the script
        result = self.eng.robust_mean_pgd(data, epsilon, nargout=1)

        return np.array(result).reshape(-1)

    def cleanup(self):
        # Stop the MATLAB engine
        self.eng.quit()


class CDGS20_PGDEstimatorPython(RobustMeanEstimator):
    def project_onto_capped_simplex_simple(
        self, w: np.ndarray, cap: float
    ) -> np.ndarray:
        tL = w.min() - 1
        tR = w.max()

        for _ in range(50):
            t = (tL + tR) / 2
            if np.sum(np.minimum(np.maximum(w - t, 0), cap)) < 1:
                tR = t
            else:
                tL = t

        t = (tL + tR) / 2
        return np.minimum(np.maximum(w - t, 0), cap)

    def _estimate(self, sample: np.ndarray, epsilon: float, n_itr=10) -> np.ndarray:
        n, d = sample.shape
        epsN = round(epsilon * n)
        step_size = 1 / n
        w = np.ones((n, 1)) / n

        for _ in range(n_itr):
            Xw = sample.T @ w
            Sigma_w = sample.T @ np.diag(w.reshape(-1)) @ sample - Xw @ Xw.T
            # return the largest eigenvalue and eigenvector of sigma_w
            v, u = np.linalg.eigh(Sigma_w)
            v = v[-1]
            u = u[:, -1].reshape(-1, 1)

            Xu = sample @ u
            nabla_f_w = Xu * Xu - 2 * (w.T @ Xu) @ Xu
            old_w = w
            w = w - step_size * nabla_f_w / np.linalg.norm(nabla_f_w)
            w = self.project_onto_capped_simplex_simple(w, 1 / (n - epsN))

            Sigma_w = sample.T @ np.diag(w) @ sample - Xw @ Xw.T
            _, new_v = np.linalg.eigh(Sigma_w)

            if new_v < v:
                step_size = step_size * 2
            else:
                step_size = step_size / 4
                w = old_w

        return sample.T @ w


class DKKLMS16_FilterEstimator(RobustMeanEstimator):
    """
    Estimator for the mean by pruning points that are far away.
    """

    def __init__(self):
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath("reference/robust_bn_faster/", nargout=0)

    def _estimate(self, sample: np.ndarray, epsilon: float) -> np.ndarray:
        # Define a sample array
        data = matlab.double(sample.tolist())

        # Call the MATLAB function from the script
        result = self.eng.robust_mean_heuristic(data, epsilon, nargout=1)

        return np.array(result).reshape(-1)

    def cleanup(self):
        # Stop the MATLAB engine
        self.eng.quit()
