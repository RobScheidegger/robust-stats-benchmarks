from abc import ABC, abstractmethod
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

        self.n, self.d = data.shape
        self.estimation = self._estimate(data, epsilon)
        assert self.estimation.shape == (self.d,), (
            "The estimated mean must be a vector: "
            + str(self.estimation.shape)
            + " vs "
            + str(self.d)
        )
        return self.estimation

    def cleanup(self):
        """
        Cleans up any state that the estimator may have used or created.
        """
        pass
