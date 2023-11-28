from dataclasses import dataclass
import numpy as np
from adversary import Adversary
from distribution import Distribution
from robust_mean_estimator import RobustMeanEstimator

SAMPLE_COUNT = 100


@dataclass
class BenchmarkResult:
    """
    This class represents the result of a benchmark.
    """

    n: int
    d: int
    adversary: str
    estimator: str
    distribution: str
    true_mu: float
    true_sigma: float
    epsilon: float
    loss: float
    loss_stddev: float


class RobustMeanEvaluator:
    def __init__(
        self,
        epsilon: float,
        n: int,
        d: int,
        adversary_type: type[Adversary],
        estimator_type: type[RobustMeanEstimator],
        distribution: Distribution,
    ) -> None:
        self.epsilon = epsilon
        self.mu = distribution.mu
        self.sigma = distribution.sigma
        self.n = n
        self.d = d
        self.adversary_type = adversary_type

        self.estimator_type = estimator_type
        self.distribution = distribution

        self._prepare()

    def _prepare(self):
        self.samples = [
            self.distribution.sample(self.n, self.d) for _ in range(SAMPLE_COUNT)
        ]
        self.adversary = self.adversary_type(self.mu, self.sigma, self.epsilon)
        self.corrupted_samples = [
            self.adversary.corrupt_sample(sample.copy()) for sample in self.samples
        ]

    def evaluate(
        self,
        estimate_loss: bool = False,
    ) -> BenchmarkResult:
        losses = []
        for corrupted_sample in self.corrupted_samples:
            estimator: RobustMeanEstimator = self.estimator_type()

            estimate = estimator.estimate_mean(corrupted_sample, self.epsilon)

            if estimate_loss:
                loss = self.loss(estimate)
                losses.append(loss)

        return BenchmarkResult(
            self.n,
            self.d,
            self.adversary_type.__name__,
            self.estimator_type.__name__,
            type(self.distribution).__name__,
            self.mu,
            self.sigma,
            self.epsilon,
            np.mean(losses),
            np.std(losses),
        )

    def loss(self, estimate: np.ndarray) -> float:
        """
        This method computes the loss of the estimator.

        :param mu: The mean to compute the loss at.
        :return: The loss.
        """
        return np.mean((estimate - self.mu)) ** 2