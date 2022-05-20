import os
from dataclasses import asdict

import tensorflow as tf

import wandb

from ..dataset.tokenizers import Tokenizer
from ..model.callback_metrics import (
    FuncVisualizer,
    IntegralDifferenceMetric,
    PointMetrics,
    TableCallbackMetric,
)
from ..model.callbacks import (
    EvalDatasetWithoutTeacherForcing,
    ExtendedWandbCallback,
    InputRegularizerScheduler,
    LRCallback,
    RegressionLambdaScheduler,
)
from ..model.config import Config
from ..model.model import TransformerType
from ..model.schedules import DelayedSchedule, InverseCosineScheduler

from ..model.utils.convertor import (
    BestFittingFilter,
    FitConstantsConvertor,
    SimpleConvertor,
)
from ..model.utils.decoding import GreedySearch, RandomSampler, TopK


def get_callbacks(
    config: Config, tokenizer: Tokenizer, val_set, transformer, transformer_type
):
    callbacks = []

    inverse_cosine_scheduler = InverseCosineScheduler(1.0, config.max_epoch)
    scheduler = DelayedSchedule(
        inverse_cosine_scheduler, config.callback_config.regression_delay, 0.0
    )

    wandb_config = {
        "dataset_size": config.dataset_config.dataset_size,
        "num_sampled_points": config.dataset_config.num_points,
        "num_variables": len(config.dataset_config.variables),
        "model_config": asdict(config),
        "regression_lambda_scheduler": scheduler.get_config(),
        "num_epochs": config.max_epoch,
        "transformer_type": transformer_type.value,
    }

    if transformer_type in [TransformerType.REG_AS_SEQ]:
        regularizer_scheduler = tf.keras.optimizers.schedules.CosineDecay(
            0.1, config.max_epoch
        )
        wandb_config["input_regularization"] = config.input_regularizer.config()
        input_regularizer_callback = InputRegularizerScheduler(
            regularizer_scheduler, config.input_regularizer
        )
        callbacks.append(input_regularizer_callback)

    wandb.init(project="symbolic-regression", entity="r4i", config=wandb_config)
    wandb_callback = ExtendedWandbCallback(save_model=False)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=f'{os.getenv("OUTPUT")}/logs/{os.getenv("JOB_ID")}/',
        profile_batch=(300, 305),
    )

    variables = config.dataset_config.variables

    metrics = [
        TableCallbackMetric(variables),
        PointMetrics([10, 1, 1e-1, 1e-2, 1e-3], 50, variables),
        IntegralDifferenceMetric(variables, 5),
        FuncVisualizer(config.dataset_config.variables),
    ]

    greedy_search_config = config.callback_config.greedy_search
    convertor = (
        SimpleConvertor(tokenizer)
        if config.decoder_config.mode != "no_reg"
        else FitConstantsConvertor(
            tokenizer, config.dataset_config.extended_representation
        )
    )
    optimization_type = (
        BestFittingFilter(
            tokenizer, extended_repre=config.dataset_config.extended_representation
        )
        if config.decoder_config.mode != "no_reg"
        else BestFittingFilter(
            tokenizer,
            "bfgs",
            extended_repre=config.dataset_config.extended_representation,
        )
    )

    no_teacher_callback = EvalDatasetWithoutTeacherForcing(
        {
            "validation": val_set,
        },
        tokenizer,
        metrics,
        GreedySearch(
            None,
            50,
            tokenizer,
            transformer,
            config.decoder_config.mode != "no_reg",
            convertor,
        ),
        "no_teacher",
        evaluate_each=greedy_search_config.evaluate_each,
        max_num=greedy_search_config.max_num,
    )

    beam_search_config = config.callback_config.beam_search
    beam_search = EvalDatasetWithoutTeacherForcing(
        {
            "validation": val_set,
        },
        tokenizer,
        metrics,
        RandomSampler(
            TopK(16),
            50,
            tokenizer,
            transformer,
            config.decoder_config.mode != "no_reg",
            optimization_type,
            beam_search_config.beam_width,
        ),
        "beam_search",
        evaluate_each=beam_search_config.evaluate_each,
        max_num=beam_search_config.max_num,
    )

    regression_callback = RegressionLambdaScheduler(scheduler)

    callbacks += [
        wandb_callback,
        tensorboard_callback,
        no_teacher_callback,
        regression_callback,
        beam_search,
        LRCallback(),
    ]
    return callbacks
