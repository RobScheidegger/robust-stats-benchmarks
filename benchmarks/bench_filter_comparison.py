import numpy as np
from benchmarker import Benchmarker
from estimators import (
    Filter2018MATLABEstimator,
    Filter2018PythonEstimator,
    MedianEstimator,
    DKKLMS16_FilterEstimator,
    BaseEstimator,
    HeuristicRustEstimator,
    FilterRustEstimator,
)
from distributions import GaussianDistribution
from adversaries import SwapAdversary

D = 500
EPSILON = 0.1
K = 5
N = int(K * D / (EPSILON**2))
NUM_REPETITIONS = 1


distribution = GaussianDistribution(mu=np.zeros(D), sigma=np.ones(D))
np.random.seed(0)
samples = [distribution.sample(N, D) for _ in range(NUM_REPETITIONS)]
adversary = SwapAdversary(true_mu=np.zeros(D), true_sigma=np.ones(D), epsilon=EPSILON)
samples = [
    adversary.corrupt_sample(sample.copy()).astype(np.float32) for sample in samples
]

python_estimator = Filter2018PythonEstimator()
matlab_estimator = Filter2018MATLABEstimator()
median_estimator = MedianEstimator()
heuristic_estimator = DKKLMS16_FilterEstimator()
mean_estimator = BaseEstimator()
heuristic_rust_esimtator = HeuristicRustEstimator()
filter_rust_estimator = FilterRustEstimator()


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


def heuristic():
    for sample in samples:
        heuristic_estimator.estimate_mean(sample, EPSILON)


def heuristic_rust():
    for sample in samples:
        heuristic_rust_esimtator.estimate_mean(sample, EPSILON)


def filter_rust():
    for sample in samples:
        filter_rust_estimator.estimate_mean(sample, EPSILON)


if __name__ == "__main__":
    benchmarker = Benchmarker()
    benchmarker.add_benchmark(mean, "Mean (Baseline)", True)
    benchmarker.add_benchmark(median, "Median")
    benchmarker.add_benchmark(python, "Python Filter 2018")
    benchmarker.add_benchmark(matlab, "MATLAB Filter 2018")
    benchmarker.add_benchmark(heuristic, "Heuristic Filter 2016")
    benchmarker.add_benchmark(heuristic_rust, "Heuristic Filter 2016 (Rust)")
    benchmarker.add_benchmark(filter_rust, "Filter 2018 (Rust)")

    benchmarker.run()

    benchmarker.print_results()
    benchmarker.to_csv("bench_filter_comparison.csv")
