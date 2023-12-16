import time
from typing import Callable
import numpy as np

WARMUP_EXECUTIONS = 0
REPETITIONS = 1
PRINT_COLUMN_WIDTH = 25


class Benchmarker:
    """
    A simple benchmarking class to time the execution of a function.
    """

    def __init__(self):
        self.benchmarks: list[tuple[Callable, str]] = []
        self.results: list[list[float]] = []
        self.baseline = None

    def add_benchmark(self, benchmark, name: str, baseline: bool = False):
        self.benchmarks.append((benchmark, name))
        self.baseline = name if baseline else self.baseline

    def run(self):
        for benchmark, name in self.benchmarks:
            times = []
            for i in range(WARMUP_EXECUTIONS):
                benchmark()

            for _ in range(REPETITIONS):
                start_time = time.time()
                benchmark()
                times.append(time.time() - start_time)

            self.results.append(times)

    def print_results(self):
        # Make a header with the names of the benchmarks padded with space so it prints nicely
        header_columns = ["Name", "Mean", "Sigma", "Min", "Max", "Baseline"]
        header_columns = [column.ljust(PRINT_COLUMN_WIDTH) for column in header_columns]

        print("".join(header_columns))

        # Find the baseline index
        baseline_index = None
        baseline_mean = None
        for i, (_, name) in enumerate(self.benchmarks):
            if name == self.baseline:
                baseline_index = i
                baseline_mean = np.mean(self.results[i])
                break

        assert baseline_mean is not None, "The baseline must be set."

        for i in range(len(self.benchmarks)):
            _, name = self.benchmarks[i]
            result = self.results[i]
            mean = np.mean(result)
            sigma = np.std(result)
            min = np.min(result)
            max = np.max(result)
            columns = [
                self.benchmarks[i][1].ljust(PRINT_COLUMN_WIDTH),
                f"{mean:10.8}".ljust(PRINT_COLUMN_WIDTH),
                f"{sigma:10.8}".ljust(PRINT_COLUMN_WIDTH),
                f"{min:10.8}".ljust(PRINT_COLUMN_WIDTH),
                f"{max:10.8}".ljust(PRINT_COLUMN_WIDTH),
                f"{mean / baseline_mean:10.8}".ljust(PRINT_COLUMN_WIDTH)
                if baseline_index is not None
                else "--".ljust(PRINT_COLUMN_WIDTH),
            ]
            print("".join(columns))

    def to_csv(self, filename: str):
        # Print all of the results out to a CSV

        # Find the baseline index
        baseline_mean = None
        for i, (_, name) in enumerate(self.benchmarks):
            if name == self.baseline:
                baseline_mean = np.mean(self.results[i])
                break

        assert baseline_mean is not None, "The baseline must be set."

        with open(filename, "w") as f:
            f.write("Name,Mean,Sigma,Min,Max,Baseline\n")
            for i in range(len(self.benchmarks)):
                _, name = self.benchmarks[i]
                result = self.results[i]
                mean = np.mean(result)
                sigma = np.std(result)
                min = np.min(result)
                max = np.max(result)
                f.write(f"{name},{mean},{sigma},{min},{max},{mean / baseline_mean}\n")

    def get_results(self):
        return self.results
