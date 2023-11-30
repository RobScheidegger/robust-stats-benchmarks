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
