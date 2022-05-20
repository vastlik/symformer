import csv
import datetime
import os
import re
import typing as t

import numpy as np
from aclick import command

from symformer.model.runner import Runner
from symformer.model.utils.const_improver import OptimizationType


def process_benchmark(expression):
    pow_regexp = r"pow\((.*?),(.*?)\)"
    pow_replace = r"((\1) ^ (\2))"
    processed = re.sub(pow_regexp, pow_replace, expression)

    div_regexp = r"div\((.*?),(.*?)\)"
    div_replace = r"((\1) / (\2))"
    processed = re.sub(div_regexp, div_replace, processed)
    processed = processed.replace("x1", "x")
    processed = processed.replace("x2", "y")
    return processed


def open_csv(file_name: str):
    equations = []
    with open(file_name) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",")
        for row in reader:
            equations.append(
                (
                    row["name"],
                    process_benchmark(row["expression"]),
                    int(row["variables"]),
                )
            )
    return equations


@command("evaluate-benchmark")
def evaluate(
    univariate_model: str,
    bivariate_model: str,
    benchmark_path: t.Optional[str] = None,
    optimization_type: OptimizationType = "gradient",
):
    if benchmark_path is None:
        benchmark_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "assets", "benchmarks.csv"
        )
    runner = Runner.from_checkpoint(
        univariate_model,
        optimization_type=optimization_type,
        use_pool=False,
        early_stopping=True,
    )

    runner_2d = Runner.from_checkpoint(
        bivariate_model,
        optimization_type=optimization_type,
        use_pool=False,
        early_stopping=True,
    )

    r2s = []
    maes = []
    starting_time = datetime.datetime.now()
    for i, (name, eq, num_variables) in enumerate(open_csv(benchmark_path)):
        prev = datetime.datetime.now()
        if num_variables == 1:
            prediction, r2, relative_error = runner.predict(eq)
        else:
            prediction, r2, relative_error = runner_2d.predict(eq)

        try:
            r2s.append(r2)
            maes.append(relative_error)
        except Exception:
            continue

        print()
        print(name)
        print(eq)
        print(prediction)
        print(r2s[-1], maes[-1])
        print(datetime.datetime.now() - prev)

    print(datetime.datetime.now() - starting_time)
    print("\nResults:")
    print(np.mean(r2s), np.median(r2s))
    print(np.mean(maes), np.median(maes))


if __name__ == "__main__":
    evaluate()
