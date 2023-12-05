from adversaries import *
from adversary import Adversary
from distribution import Distribution
from distributions import *
from estimators import *
from robust_mean_estimator import RobustMeanEstimator
from robust_mean_evaluator import RobustMeanEvaluator

import numpy as np


def main():
    d_options: list[int] = [5, 10, 15, 20, 25, 30, 35, 40]
    # n_options: list[int] = [100]
    epsilon_options: list[int] = [0.05, 0.1, 0.2]

    adversaries: list[Adversary] = [
        # NullAdversary,
        # InfinityAdversary,
        SwapAdversary,
    ]

    estimators: list[RobustMeanEstimator] = [
        # BaseEstimator,
        MedianEstimator,
        # RANSACEstimator,
        # GeoMedianEstimator,
        # PruningEstimator,
        Filter2018PythonEstimator,
        # Filter2018MATLABEstimator,
        CDGS20_PGDEstimator,
        # DKKLMS16_FilterEstimator,
        HeuristicRustEstimator,
        # FilterRustEstimator,
    ]

    distributions: list[Distribution] = [
        GaussianDistribution,
    ]

    MU = 0
    SIGMA = 1

    results = []
    for estimator_type in estimators:
        for d in d_options:
            for epsilon in epsilon_options:
                n = int(5 * d / (epsilon**2))
                for adversary_type in adversaries:
                    for distribution_type in distributions:
                        mu = np.ones((d)) * MU
                        sigma = np.ones((d)) * SIGMA

                        evaluator = RobustMeanEvaluator(
                            epsilon=epsilon,
                            n=n,
                            d=d,
                            adversary_type=adversary_type,
                            estimator_type=estimator_type,
                            distribution=distribution_type(mu, sigma),
                        )

                        result = evaluator.evaluate(estimate_loss=True)
                        results.append(result)

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
            f"{result.loss/(result.epsilon * np.sqrt(np.log(1.0 / result.epsilon))):10e}",
        ]
        print(" ".join(print_outputs))

    # Also write the results to a file as CSV
    import csv

    with open("results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "n",
                "d",
                "adversary",
                "estimator",
                "epsilon",
                "loss",
                "loss_stddev",
                "loss/epsilon",
                "loss_reg1",
                "loss_reg2",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.n,
                    result.d,
                    result.adversary,
                    result.estimator,
                    result.epsilon,
                    result.loss,
                    result.loss_stddev,
                    result.loss / result.epsilon,
                    result.loss / (result.epsilon * np.sqrt(result.d)),
                    result.loss
                    / (result.epsilon * np.sqrt(np.log(1.0 / result.epsilon))),
                ]
            )


if __name__ == "__main__":
    main()
