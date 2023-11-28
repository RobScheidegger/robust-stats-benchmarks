from adversaries import *
from adversary import Adversary
from distribution import Distribution
from distributions import *
from estimators import *
from robust_mean_estimator import RobustMeanEstimator
from robust_mean_evaluator import RobustMeanEvaluator

import numpy as np


def main():
    d_options: list[int] = [10, 10000]
    n_options: list[int] = [100]
    epsilon_options: list[int] = [0.05]

    adversaries: list[Adversary] = [
        NullAdversary,
        # ZeroAdversary,
        ConstantAdversary,
        InfinityAdversary,
        SwapAdversary,
    ]

    estimators: list[RobustMeanEstimator] = [
        base_estimator,
        MedianEstimator,
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
                            evaluator = (
                                RobustMeanEvaluator(
                                    epsilon=epsilon,
                                    n=n,
                                    d=d,
                                    adversary_type=adversary_type,
                                    estimator_type=estimator_type,
                                    distribution=GaussianDistribution(mu, sigma),
                                ),
                            )

                            result = evaluator.evaluate(estimate_loss=True)
                            results.append(result)

    # Print a header column
    print(
        f"{'n':6s} {'d':6s} {'adversary':20s} {'estimator':20s} {'epsilon':10s} {'loss(x1000)':10s} {'loss_stddev':10s}"
    )
    LOSS_FACTOR = 1
    for result in evaluator.results:
        # Pretty print the results as different columns
        print(
            f"{result.n:6d} {result.d:6d} {result.adversary:20s} {result.estimator:20s} {result.epsilon:10.4f} {LOSS_FACTOR * result.loss:10.4f} {result.loss_stddev:10.4f}"
        )


if __name__ == "__main__":
    main()
