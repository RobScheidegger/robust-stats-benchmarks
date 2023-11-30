import numpy as np
from benchmarker import Benchmarker
from estimators import (
    Filter2018MATLABEstimator,
    Filter2018PythonEstimator,
    MedianEstimator,
    DKKLMS16_FilterEstimator,
    BaseEstimator,
)
from distributions import GaussianDistribution

D = 20
EPSILON = 0.2
K = 5
N = int(K * D / (EPSILON**2))
NUM_SAMPLES = 10


distribution = GaussianDistribution(mu=np.zeros(D), sigma=np.ones(D))
np.random.seed(0)
samples = [distribution.sample(N, D) for _ in range(10)]

python_estimator = Filter2018PythonEstimator()
matlab_estimator = Filter2018MATLABEstimator()
median_estimator = MedianEstimator()
heuristic_estimator = DKKLMS16_FilterEstimator()
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


def heuristic():
    for sample in samples:
        heuristic_estimator.estimate_mean(sample, EPSILON)


if __name__ == "__main__":
    benchmarker = Benchmarker()
    benchmarker.add_benchmark(mean, "Mean (Baseline)", True)
    benchmarker.add_benchmark(median, "Median")
    benchmarker.add_benchmark(python, "Python Filter 2018")
    benchmarker.add_benchmark(matlab, "MATLAB Filter 2018")
    benchmarker.add_benchmark(heuristic, "Heuristic Filter 2016")

    benchmarker.run()

    benchmarker.print_results()
