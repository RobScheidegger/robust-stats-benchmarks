"""
This file contains custom benchmarks for the accuracy and timing of robust mean estimators.
"""

from dataclasses import dataclass
from adversaries import *
from estimators import *
from adversary import Adversary


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
    ) -> None:
        self.adversaries = adversaries
        self.estimators = estimators

    def benchmark(self, n: int, d: int, dist) -> BenchmarkResult:
        pass


def main():
    adversaries: list[Adversary] = [
        NullAdversary,
        ZeroAdversary,
        ConstantAdversary,
        InfinityAdversary,
    ]

    estimators: list[RobustMeanEstimator] = [
        BaseEstimator,
        MedianEstimator,
    ]

    evaluator = RobustMeanEvaluator(adversaries, estimators)

    pass


if __name__ == "__main__":
    main()
