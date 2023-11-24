"""
This file contains custom benchmarks for the accuracy and timing of robust mean estimators.
"""

from dataclasses import dataclass
from adversaries import *
from distribution import Distribution
from estimators import *
from adversary import Adversary
from distributions import *
import time


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
    true_stddev: float
    epsilon: float
    loss: float
    loss_stddev: float
    time: float
    time_stddev: float


class RobustMeanEvaluator:
    def __init__(
        self,
        adversaries: list[Adversary],
        estimators: list[RobustMeanEstimator],
        distributions: list[Distribution],
        repeat: int = 10,
    ) -> None:
        self.adversaries = adversaries
        self.estimators = estimators
        self.distributions = distributions
        self.results = []
        self.repeat = repeat

    def benchmark(
        self, mu: np.ndarray, stddev: np.ndarray, n: int, d: int, epsilon: float
    ) -> BenchmarkResult:
        for distribution in self.distributions:
            # For each distribution, make the data ahead of time, so that every time it is fair
            distribution_instance: Distribution = distribution(mu, stddev)
            data: np.ndarray = distribution_instance.sample(n, d)
            for adversary in self.adversaries:
                for estimator in self.estimators:
                    self.results.append(
                        self.benchmark_single(
                            mu,
                            stddev,
                            n,
                            d,
                            epsilon,
                            adversary,
                            estimator,
                            distribution,
                            data,
                        )
                    )

        pass

    def benchmark_single(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        n: int,
        d: int,
        epsilon: float,
        adversary_type: Adversary,
        estimator_type: RobustMeanEstimator,
        distribution_type: Distribution,
        data: np.ndarray,
    ) -> BenchmarkResult:
        adversary_instance: Adversary = adversary_type(mu, sigma, epsilon)

        corrupted_data: np.ndarray = adversary_instance.corrupt_sample(data.copy())

        times = []
        losses = []
        for _ in range(self.repeat):
            estimator_instance: RobustMeanEstimator = estimator_type(
                corrupted_data, mu, sigma, epsilon
            )

            # Start a timer to compute elapsed time
            start_time = time.time()
            estimate = estimator_instance.estimate_mean()
            end_time = time.time()
            time_elapsed = end_time - start_time
            times.append(time_elapsed)

            loss = estimator_instance.loss()
            losses.append(loss)

        return BenchmarkResult(
            n,
            d,
            adversary_type.__name__,
            estimator_type.__name__,
            distribution_type.__name__,
            mu,
            sigma,
            epsilon,
            np.mean(losses),
            np.std(losses),
            np.mean(times),
            np.std(times),
        )


def main():
    adversaries: list[Adversary] = [
        # NullAdversary,
        # ZeroAdversary,
        ConstantAdversary,
        InfinityAdversary,
        SwapAdversary,
    ]

    estimators: list[RobustMeanEstimator] = [
        BaseEstimator,
        MedianEstimator,
    ]

    distributions: list[Distribution] = [
        GaussianDistribution,
    ]

    evaluator = RobustMeanEvaluator(adversaries, estimators, distributions)

    mu = np.ones((2000)) * 1000
    sigma = np.ones((2000)) * 10

    evaluator.benchmark(
        mu=mu,
        stddev=sigma,
        n=1000,
        d=2000,
        epsilon=0.10,
    )

    # Print a header column
    print(
        f"{'n':6s} {'d':6s} {'adversary':20s} {'estimator':20s} {'epsilon':10s} {'loss(x1000)':10s} {'loss_stddev':10s} {'time':10s} {'time_stddev':10s}"
    )
    LOSS_FACTOR = 1
    for result in evaluator.results:
        # Pretty print the results as different columns
        print(
            f"{result.n:6d} {result.d:6d} {result.adversary:20s} {result.estimator:20s} {result.epsilon:10.4f} {LOSS_FACTOR * result.loss:10.4f} {result.loss_stddev:10.4f} {result.time:10.4f} {result.time_stddev:10.4f}"
        )


if __name__ == "__main__":
    main()
