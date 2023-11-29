from adversaries import *
from adversary import Adversary
from distribution import Distribution
from distributions import *
from estimators import *
from robust_mean_estimator import RobustMeanEstimator
from robust_mean_evaluator import RobustMeanEvaluator

import numpy as np


def main():
    d_options: list[int] = [10, 100, 1000, 10000]
    n_options: list[int] = [100]
    epsilon_options: list[int] = [0.05, 0.1, 0.15, 0.2, 0.25]

    adversaries: list[Adversary] = [
        # NullAdversary,
        # ZeroAdversary,
        # ConstantAdversary,
        # InfinityAdversary,
        SwapAdversary,
    ]

    estimators: list[RobustMeanEstimator] = [
        # BaseEstimator,
        MedianEstimator,
        # RANSACEstimator,
        # GeoMedianEstimator,
        PruningEstimator,
    ]

    distributions: list[Distribution] = [
        GaussianDistribution,
    ]

    MU = 100
    SIGMA = 10

    results = []
    for d in d_options:
        for n in n_options:
            for epsilon in epsilon_options:
                for adversary_type in adversaries:
                    for estimator_type in estimators:
                        for distribution_type in distributions:
                            mu = np.ones((d)) * MU
                            sigma = np.ones((d)) * SIGMA

                            evaluator = RobustMeanEvaluator(
                                epsilon=epsilon,
                                n=n,
                                d=d,
                                adversary_type=adversary_type,
                                estimator_type=estimator_type,
                                distribution=GaussianDistribution(mu, sigma),
                            )

                            result = evaluator.evaluate(estimate_loss=True)
                            results.append(result)

    # Print a header column
    print(
        f"{'n':6s} {'d':6s} {'adversary':20s} {'estimator':20s} {'epsilon':10s} {'loss':10s} {'loss_stddev':10s} {'loss/epsilon':10s} {'loss/(eps * sqrt(d))':20s}"
    )
    for result in results:
        print_outputs = [
            f"{result.n:6d}",
            f"{result.d:6d}",
            f"{result.adversary:20s}",
            f"{result.estimator:20s}",
            f"{result.epsilon:10.4f}",
            f"{result.loss:10e}",
            f"{result.loss_stddev:10e}",
            f"{result.loss/result.epsilon:10e}",
            f"{result.loss/(result.epsilon * np.sqrt(result.d)):10e}",
        ]
        print(" ".join(print_outputs))


if __name__ == "__main__":
    main()
