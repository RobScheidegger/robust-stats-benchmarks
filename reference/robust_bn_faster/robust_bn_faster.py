import numpy as np
import matlab.engine

from tqdm import tqdm
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

    def _estimate(self, sample: np.ndarray, epsilon: float, **kwargs) -> np.ndarray:
        (n, d), k = sample.shape, 1
        n_itr = kwargs.get("n_itr", 20)
        epsN = round(epsilon * n)
        step_size = 1 / n
        w = np.ones(n) / n

        for _ in range(n_itr):
            Xw = sample.T @ w
            Sigma_w = (sample.T @ np.diag(w) @ sample) - np.outer(Xw, Xw)

            u_val, u = np.linalg.eigh(Sigma_w)
            u_val = u_val[-1]
            u = u[:, -1]
            Xu = sample @ u
            nabla_f_u = Xu * Xu - (2 * np.inner(u, Xw)) * Xu
            w = w - step_size * nabla_f_u / np.linalg.norm(nabla_f_u)
            w = self.project_onto_capped_simplex_simple(w, 1 / (n - epsN))

        return np.sum(w * sample.T, axis=1)


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
