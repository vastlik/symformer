import os
import typing as t
from collections import Counter

import tensorflow as tf
from tensorflow.python.data.ops.options import AutoShardPolicy
from tensorflow.python.framework.errors_impl import DataLossError

from ..dataset.tokenizers import Tokenizer
from ..dataset.utils.tree import prefix_to_infix
from ..model.config import Config
from ..model.model import TransformerType


def print_all(dataset, tokenizer):
    j = 0
    with open("outputs.out", "w") as f:
        for i in dataset.as_numpy_iterator():
            j += 1
            if j % 1000 == 0:
                print(j)
            expr = i["symbolic_expression"].decode("utf8").split()
            constants = i["constants"]
            _, infix = prefix_to_infix(expr, constants, tokenizer)
            print(infix, file=f)


def filter_not_seen(not_seen):
    def is_in_not_seen(x):
        return tf.math.reduce_any(tf.equal(not_seen, x["symbolic_expression"]))

    return is_in_not_seen


def filter_unknown_ops(x):
    return tf.reduce_all(x["symbolic_expr_input"] != -1)


def filter_inf(x):
    return (
        tf.reduce_all(tf.math.is_finite(x["points"]))
        and tf.reduce_all(tf.math.is_finite(x["constants"]))
        and (
            tf.reduce_all(x["constants"] < 100_000)
            and tf.reduce_all(x["constants"] > -100_000)
        )
        and tf.reduce_all(tf.math.abs(x["constants"])[x["constants"] != 0] > 1e-10)
    )


def convert_constant(constants):
    non_zero_mask = constants != 0
    values = tf.math.ceil(tf.math.log(tf.math.abs(constants)) / tf.math.log(10.0))
    exponents = tf.where(non_zero_mask, values, 0)
    mantissa = tf.where(non_zero_mask, constants / tf.math.pow(10.0, exponents), 0)
    return mantissa, exponents


def convert_symbolic_expr(x):
    exp, sym = x
    if sym == "C":
        return exp, tf.strings.join(["C", tf.strings.as_string(exp)], separator="")
    return exp, sym


def get_tokenizer(
    tokenizer: Tokenizer,
    transformer_type: TransformerType,
    sample_points: bool,
    num_points,
    extended_repre,
):
    def tokenize(data):
        if extended_repre:
            mantissa, exponents = convert_constant(data["constants"][:-1])
            exponents = tf.cast(exponents, tf.int32)

            sym = tf.strings.split(data["symbolic_expression"], sep=" ")
            symbolic_expression = tf.map_fn(convert_symbolic_expr, (exponents, sym))[1]
            symbolic_expression = tf.strings.reduce_join(
                symbolic_expression, separator=" "
            )
        else:
            mantissa = data["constants"][:-1]
            symbolic_expression = data["symbolic_expression"]

        res = {
            "points": data["points"],
            "constants_target": tf.concat([mantissa, [0.0]], axis=-1),
            "symbolic_expr_input": tokenizer.encode(
                tf.strings.reduce_join(
                    [tokenizer.start, symbolic_expression], separator=" "
                )
            ),
            "symbolic_expr_target": tokenizer.encode(
                tf.strings.reduce_join(
                    [symbolic_expression, tokenizer.end], separator=" "
                )
            ),
        }

        if transformer_type in [TransformerType.REG_AS_SEQ]:
            res["constants_input"] = tf.concat([[0.0], mantissa], axis=-1)

        if sample_points:
            indicies = tf.range(0, tf.shape(res["points"])[0])
            indicies = tf.random.shuffle(indicies)[:num_points]
            res["points"] = tf.gather(res["points"], indicies, axis=0)
        else:
            res["points"] = res["points"][:num_points]
        return res

    return tokenize


def merge_dataset_files(path: str):
    dataset_paths = [
        os.path.abspath(os.path.join(path, name)) for name in os.listdir(path)
    ]
    dataset = [tf.data.experimental.load(path) for path in dataset_paths]
    full_dataset = dataset[0]
    for i in range(1, len(dataset)):
        full_dataset = full_dataset.concatenate(dataset[i])

    return full_dataset


def get_not_seen(dataset, num_of_not_seen: int):
    not_seen = set()
    for i in dataset.as_numpy_iterator():
        not_seen.add(i["symbolic_expression"])
        if len(not_seen) == num_of_not_seen:
            break

    not_seen = list(not_seen)
    filtered_not_seen = []
    for i in not_seen:
        res = tf.strings.regex_replace(i, "^-?[0-9] ", "C ")
        res = tf.strings.regex_replace(res, " -?[0-9]$", " C")
        res = tf.strings.regex_replace(res, " -?[0-9] ", " C ")
        filtered_not_seen.append(res)
    return filtered_not_seen


def count_dataset_size(dataset):
    j = 0
    for _ in dataset.as_numpy_iterator():
        j += 1
        if j % 10000 == 0:
            print(j)

    print(j)
    exit()


def get_vocab(dataset):
    cnt = Counter()
    for i, data in enumerate(dataset):
        sym = data["symbolic_expression"].numpy().decode("utf8").split()
        cnt.update(sym)
        if i % 1000 == 0:
            print(i)

    print(cnt)
    exit()


def get_options():
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
    return options


def sharding_policy(
    path: str,
    tokenizer_func: t.Callable,
    batch_size: int,
    dataset_size: int,
    training: bool,
):
    dataset_paths = [
        os.path.abspath(os.path.join(path, name)) for name in os.listdir(path)
    ]

    def policy(input_context: tf.distribute.InputContext):
        pipeline_id = input_context.input_pipeline_id
        num_pipelines = input_context.num_input_pipelines
        datasets = dataset_paths[pipeline_id::num_pipelines]
        datasets = [
            tf.data.experimental.load(dataset_path) for dataset_path in datasets
        ]
        datasets = tf.data.Dataset.from_tensor_slices(datasets)
        interleaved = datasets.interleave(
            lambda x: x, cycle_length=1, num_parallel_calls=tf.data.AUTOTUNE
        ).filter(filter_inf)

        if dataset_size != 0:
            interleaved = interleaved.take(dataset_size)

        tokenized = interleaved.map(
            tokenizer_func, num_parallel_calls=tf.data.AUTOTUNE
        ).filter(filter_unknown_ops)
        if training:
            tokenized = tokenized.shuffle(50_000)
        else:
            tokenized = tokenized.cache()

        return tokenized.padded_batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)

    return policy


def load_func(path):
    try:
        dataset = tf.data.experimental.load(path)
        j = 0
        for i in dataset:
            j += 1
        return j
    except DataLossError:
        tf.print(path)


def get_datasets(
    config: Config,
    strategy: tf.distribute.Strategy,
    tokenizer: Tokenizer,
    transformer_type,
):
    dataset_config = config.dataset_config

    extended_repre = dataset_config.extended_representation
    tokenizer_func = get_tokenizer(
        tokenizer,
        transformer_type,
        dataset_config.sample_points,
        dataset_config.num_points,
        extended_repre,
    )

    training_policy = sharding_policy(
        dataset_config.path,
        tokenizer_func,
        dataset_config.batch_size,
        dataset_config.dataset_size,
        True,
    )
    training_set = strategy.distribute_datasets_from_function(training_policy)

    validation_policy = sharding_policy(
        dataset_config.valid_path,
        tokenizer_func,
        dataset_config.batch_size,
        dataset_config.test_size,
        False,
    )
    validation_set = strategy.distribute_datasets_from_function(validation_policy)

    return training_set, validation_set
