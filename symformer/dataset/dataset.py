import copy
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import tensorflow as tf
from aclick import command, FlattenParameterRenamer


@dataclass
class SymbolicExpressionDataset:
    def build(self, size):
        raise NotImplementedError()

    def build_and_save(self, path, size):
        dataset = self.build(size).take(size)
        tf.data.experimental.save(dataset, path)


@dataclass
class GeneralDataset(SymbolicExpressionDataset):
    points_range: Tuple[float, float] = (-5, 5)
    num_sample_points: int = 500
    max_num_ops: int = 10
    num_variables: int = 1
    seed: int = 0

    @classmethod
    def _get_class_name(cls):
        return "general"

    def build(self, size):
        from symformer.dataset.utils.expression import generate_random_expression

        rng = np.random.default_rng(seed=self.seed)
        variables = ["x"]
        if self.num_variables == 2:
            variables = ["x", "y"]

        gen = generate_random_expression(
            rng,
            n_points=self.num_sample_points,
            sampled_points_range=self.points_range,
            max_num_ops=self.max_num_ops,
            variables=variables,
        )

        exprs = []
        for i in range(size):
            if i % 100 == 0:
                print(f"{self.seed}: {i}")
            exprs.append(gen())

        def py_generator():
            yield from exprs

        dataset = tf.data.Dataset.from_generator(
            py_generator,
            output_signature=dict(
                constants=tf.TensorSpec((None,), dtype=tf.float32),
                points=tf.TensorSpec((self.num_sample_points, self.num_variables + 1)),
                symbolic_expression=tf.TensorSpec(tuple(), dtype=tf.string),
            ),
        )
        return dataset


supported_dataset_type = GeneralDataset


@command("generate-dataset", map_parameter_name=FlattenParameterRenamer(1))
def generate_dataset_command(
    dataset: supported_dataset_type,
    dataset_size: int,
    output_dir: str,
    n_processes: int = 1,
    seed: int = 0,
):
    """
    Generates dataset for training and evaluation.

    :param output_dir: Directory containing the resulting dataset.
    """
    if n_processes == 1:
        task_id = int(os.getenv("TASK_ID", 0))
        dataset.seed = seed * 100_000_000 + task_id
        dataset.build_and_save(
            f"{output_dir}/dataset_{seed}_{task_id}_{dataset.seed}", dataset_size
        )
    else:
        import multiprocessing as mp

        processes = []
        for i in range(n_processes):
            task_id = int(os.getenv("TASK_ID", 0))
            test_dataset = copy.deepcopy(dataset)
            test_dataset.seed = seed * 10_000_000 + i + 100000 * task_id
            size = dataset_size // n_processes
            if i == n_processes - 1:
                size += dataset_size % n_processes

            processes.append(
                mp.Process(
                    target=test_dataset.build_and_save,
                    args=(
                        f"{output_dir}/dataset_{seed}_{i}_{task_id}_{test_dataset.seed}",
                        size,
                    ),
                )
            )

        for p in processes:
            p.start()

        for p in processes:
            p.join()


if __name__ == "__main__":
    generate_dataset_command()
