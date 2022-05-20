import os

import aclick

import numpy as np

import tensorflow as tf
from matplotlib import pyplot as plt

from symformer.dataset.tokenizers import GeneralTermTokenizer
from symformer.dataset.utils.sympy_functions import evaluate_points, expr_to_func
from symformer.dataset.utils.tree import prefix_to_infix


def load_dataset(path):
    def filter_inf(x):
        return (
            tf.reduce_all(tf.math.is_finite(x["points"]))
            and tf.reduce_all(tf.math.is_finite(x["constants"]))
            and (
                tf.reduce_all(x["constants"] < 10_000)
                and tf.reduce_all(x["constants"] > -10_000)
            )
            and tf.reduce_all(x["points"] > -1e6)
        )

    datasets_path = [
        os.path.abspath(os.path.join(path, name)) for name in os.listdir(path)
    ]

    dataset = [tf.data.experimental.load(path) for path in datasets_path]

    full_dataset = dataset[0]
    for i in range(1, len(dataset)):
        full_dataset = full_dataset.concatenate(dataset[i])

    full_dataset = full_dataset.filter(filter_inf)
    return full_dataset.shuffle(100_000)


def visualize(func, points, func_str, i):
    points_ranges = [(-5, 5), (0, 5), (-5, 0)]
    for a, b in points_ranges:
        lin_space = np.reshape(np.linspace(a, b, 100_000), (-1, 1))
        try:
            y = evaluate_points(func, lin_space)
            if np.any(np.isnan(y)):
                continue
            break
        except RuntimeWarning:
            continue

    plt.title(func_str)
    if np.any(np.isnan(y)):
        return False

    plt.plot(lin_space, y, color="blue", linewidth=1)
    plt.scatter(points[:, 0], points[:, 1], s=7, color="red")
    plt.savefig(f"../visualisations/{i}.png")
    plt.clf()
    plt.close()
    return True


def vizualize_dataset(dataset):
    tokenizer = GeneralTermTokenizer()

    j = 0
    for data in dataset.as_numpy_iterator():
        tokens = data["symbolic_expression"].decode("utf8").split()
        func_str = prefix_to_infix(tokens, data["constants"], tokenizer)[1]
        func = expr_to_func(func_str, ["x"])
        success = visualize(func, data["points"], func_str, j)
        if success:
            j += 1

        if j == 200:
            break


@aclick.command("visualize-dataset")
def main(path: str, /, num: int = 200):
    """
    Visualize dataset.

    :param num: number of visualizations
    """
    dataset = load_dataset(path).take(num * 10)
    vizualize_dataset(dataset)


if __name__ == "__main__":
    main()
