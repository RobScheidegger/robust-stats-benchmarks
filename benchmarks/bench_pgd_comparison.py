import numpy as np
from benchmarker import Benchmarker
from estimators import (
    MedianEstimator,
    BaseEstimator,
    CDGS20_PGDEstimator,
    CDGS20_PGDEstimatorPython,
    PGDRustEstimator
)
from distributions import GaussianDistribution
from adversaries import SwapAdversary

D = 100
EPSILON = 0.1
K = 5
N = int(K * D / (EPSILON**2))
NUM_REPETITIONS = 1


distribution = GaussianDistribution(mu=np.zeros(D), sigma=np.ones(D))
np.random.seed(0)
samples = [distribution.sample(N, D) for _ in range(NUM_REPETITIONS)]
adversary = SwapAdversary(true_mu=0, true_sigma=1, epsilon=EPSILON)
samples = [
    adversary.corrupt_sample(sample.copy()).astype(np.float32) for sample in samples
]

python_estimator = CDGS20_PGDEstimatorPython()
matlab_estimator = CDGS20_PGDEstimator()
rust_estimator = PGDRustEstimator()
median_estimator = MedianEstimator()
mean_estimator = BaseEstimator()


def mean():
    for sample in samples:
        mean_estimator.estimate_mean(sample, EPSILON)


def python():
    for sample in samples:
        python_estimator.estimate_mean(sample, EPSILON)


def matlab():
    for sample in samples:
        matlab_estimator.estimate_mean(sample, EPSILON)


def median():
    for sample in samples:
        median_estimator.estimate_mean(sample, EPSILON)


def rust():
    for sample in samples:
        rust_estimator.estimate_mean(sample, EPSILON)


if __name__ == "__main__":
    benchmarker = Benchmarker()
    benchmarker.add_benchmark(mean, "Mean (Baseline)", True)
    # benchmarker.add_benchmark(median, "Median")
    # benchmarker.add_benchmark(python, "Python PGD")
    # benchmarker.add_benchmark(matlab, "MATLAB PGD")
    benchmarker.add_benchmark(rust, "Rust PGD")

    benchmarker.run()

    benchmarker.print_results()
    benchmarker.to_csv("bench_pgd_comparison.csv")
