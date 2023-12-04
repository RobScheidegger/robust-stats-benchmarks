from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np


class RobustMeanEstimator(ABC):
    """
    Abstract class for a robust mean estimator.
    """

    @abstractmethod
    def _estimate(self, data: np.ndarray, epsilon: float) -> np.ndarray:
        """
        This method estimates the mean of the data.

        :param epsilon: The corruption parameter.
        :return: The estimated mean.
        """
        pass

    def estimate_mean(self, data: np.ndarray, epsilon: float) -> np.ndarray:
        assert len(data.shape) == 2, "The data must be a matrix."
        assert 0 <= epsilon <= 1, "Epsilon must be in [0, 1]."

        n, d = data.shape
        estimation = self._estimate(data, epsilon)
        assert estimation.shape == (d,), (
            "The estimated mean must be a vector: "
            + str(estimation.shape)
            + " vs "
            + str(d)
        )
        return estimation

    def _restructure_covariance_estimate_rows(self, samples):
        assert len(samples.shape) == 3, "The samples must be a 3D tensor."
        n, d1, d2 = samples.shape
        assert d1 == d2, "The samples must be square matrices."

        return samples.transpose(1, 0, 2).reshape(d1, n, d1)


    def estimate_covariance(self, data: np.ndarray, epsilon: float) -> np.ndarray:
        assert len(data.shape) == 2, "The data must be a matrix."
        assert 0 <= epsilon <= 1, "Epsilon must be in [0, 1]."

        n, d = data.shape

        centered_data = data - self._estimate(data, epsilon)
        cov_samples = np.array([np.outer(centered_data[i], centered_data[i]) for i in tqdm(range(n))])

        print("transposing")
        restructured_cov_samples = self._restructure_covariance_estimate_rows(cov_samples)

        print("final computation")
        return np.array([self._estimate(sample, epsilon) for sample in tqdm(restructured_cov_samples)])


    def cleanup(self):
        """
        Cleans up any state that the estimator may have used or created.
        """
        pass
