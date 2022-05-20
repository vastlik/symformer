import datetime

import numpy as np
import tensorflow as tf
from aclick import command

from symformer.dataset.tokenizers import GeneralTermTokenizer
from symformer.dataset.utils.tree import prefix_to_infix
from symformer.model.model import TransformerType
from symformer.model.runner import Runner
from symformer.model.utils.const_improver import OptimizationType
from symformer.model.utils.convertor import clean_expr
from symformer.training.datasets import get_datasets


@command("evaluate")
def evaluate(
    model: str,
    test_dataset_path: str,
    num_equations: int = 256,
    optimization_type: OptimizationType = "gradient",
):
    tokenizer = GeneralTermTokenizer()
    runner = Runner.from_checkpoint(
        model, num_equations=num_equations, optimization_type=optimization_type
    )
    config = runner.config
    config.dataset_config.path = config.dataset_config.valid_path = test_dataset_path
    _, train = get_datasets(
        config,
        tf.distribute.get_strategy(),
        tokenizer,
        TransformerType.REG_AS_SEQ,
    )

    current = datetime.datetime.now()
    r2s = []
    relative_errors = []
    for equation in train:
        tokens = tokenizer.decode(
            tf.convert_to_tensor(equation["symbolic_expr_target"])
        ).numpy()
        constants = equation["constants_target"]
        infixes = []
        for token, const in zip(tokens, constants):
            cleaned = clean_expr(token, tokenizer)
            succ, infix = prefix_to_infix(cleaned, const.numpy(), tokenizer)
            infixes.append(infix)
        for j in range(config.dataset_config.batch_size):
            eq = infixes[j]
            print()
            print(f"{j}:")
            print(eq)
            try:
                pred, r2, re = runner.predict(eq)
                r2s.append(r2)
                relative_errors.append(re)
                print(pred, r2, re)
            except Exception as e:
                r2s.append(-np.inf)
                relative_errors.append(np.inf)
                print(str(e))

    print("Equations", num_equations)
    print(np.mean(r2s), np.nanmedian(r2s))
    print(np.mean(relative_errors), np.nanmedian(relative_errors))
    print(datetime.datetime.now() - current)


if __name__ == "__main__":
    evaluate()
