from math import ceil

import aclick

import tensorflow as tf

from .dataset.tokenizers import GeneralTermTokenizer
from .model.config import Config
from .model.model import (
    Transformer,
    TransformerNoRegression,
    TransformerType,
    TransformerWithRegressionAsSeq,
)
from .model.schedules import TransformerSchedule
from .training.callbacks import get_callbacks
from .training.datasets import get_datasets


@aclick.command(
    "train", map_parameter_name=aclick.RegexParameterRenamer([(r"config(?:\.|)", "")])
)
@aclick.configuration_option(
    "--config",
    parse_configuration=lambda f: dict(config=aclick.utils.parse_json_configuration(f)),
)
def train(config: Config):
    """
    Runs the training
    """
    multi_gpu = True

    if multi_gpu:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()

    transformer_type = TransformerType.REG_AS_SEQ
    tokenizer = GeneralTermTokenizer(
        config.dataset_config.variables,
        extended_repre=config.dataset_config.extended_representation,
    )

    with strategy.scope():
        optimizer_schedule = TransformerSchedule(config.encoder_config.d_model)
        optimizer = tf.optimizers.Adam(optimizer_schedule)
        if transformer_type == TransformerType.CLASSIC:
            transformer = Transformer(config, tokenizer, strategy)
            transformer.build(
                input_shape=[
                    (
                        None,
                        config.dataset_config.num_points,
                        len(config.dataset_config.variables) + 1,
                    ),
                    (None, None),
                ]
            )
        elif transformer_type == TransformerType.REG_AS_SEQ:
            transformer = TransformerWithRegressionAsSeq(
                config, tokenizer, config.input_regularizer, strategy
            )
            transformer.build(
                input_shape=[
                    (
                        None,
                        config.dataset_config.num_points,
                        len(config.dataset_config.variables) + 1,
                    ),
                    (None, None),
                    (None, None),
                ]
            )
        elif transformer_type == TransformerType.NO_REG:
            transformer = TransformerNoRegression(config, tokenizer, strategy)
            transformer.build(
                input_shape=[
                    (
                        None,
                        config.dataset_config.num_points,
                        len(config.dataset_config.variables) + 1,
                    ),
                    (None, None),
                ]
            )
        else:
            raise ValueError("no transformer found")

        train_set, val_set = get_datasets(config, strategy, tokenizer, transformer_type)
        callbacks = get_callbacks(
            config, tokenizer, val_set, transformer, transformer_type
        )
        transformer.compile(optimizer=optimizer, run_eagerly=False)

        transformer.summary(expand_nested=True)

        dataset_config = config.dataset_config
        steps_per_epoch = ceil(
            int(1e6) / (dataset_config.batch_size * strategy.num_replicas_in_sync)
        )
        steps_per_val_epoch = ceil(
            dataset_config.test_size
            / (dataset_config.batch_size * strategy.num_replicas_in_sync)
        )
        transformer.fit(
            x=train_set,
            epochs=config.max_epoch,
            validation_data=val_set,
            callbacks=callbacks,
            use_multiprocessing=True,
            steps_per_epoch=steps_per_epoch,
            validation_steps=steps_per_val_epoch,
        )


if __name__ == "__main__":
    train()
    # parser = argparse.ArgumentParser(description='Runs the training')
    # parser.add_argument('--config_module', type=str, help='Config module')
    # args = parser.parse_args()

    # config = importlib.import_module(args.config_module).get_config()
    # main(config)
