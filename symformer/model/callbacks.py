import numpy as np
import tensorflow as tf
import wandb
from sympy import sympify

from wandb.integration.keras import WandbCallback

from ..dataset.utils.tree import prefix_to_infix
from .callback_metrics import MetricType
from .utils.convertor import clean_expr
from .utils.decoding import Search


def convert_to_ndarray(arr):
    if not isinstance(arr, np.ndarray):
        return np.array(arr)
    return arr


class EvalDatasetCallback(tf.keras.callbacks.Callback):
    def __init__(self, datasets, tokenizer):
        self.datasets = datasets
        self.dataset_sizes = {}
        for name, dataset in self.datasets.items():
            size = 0
            for _ in dataset:
                size += 1
            self.dataset_sizes[name] = size

    def on_epoch_end(self, epoch, logs):
        for dataset_name, dataset in self.datasets.items():
            wandb_dict = {"epoch": epoch}
            # use batch size
            for key, value in self.model.evaluate(
                dataset, return_dict=True, steps=self.dataset_sizes[dataset_name]
            ).items():
                wandb_dict[f"{dataset_name}/{key}"] = value
            wandb.log(wandb_dict)


class EvalDatasetWithoutTeacherForcing(tf.keras.callbacks.Callback):
    def __init__(
        self,
        datasets,
        tokenizer,
        metrics,
        search: Search,
        name: str,
        evaluate_each: int = 0,
        max_num: int = 0,
    ):
        self.datasets = datasets
        self.tokenizer = tokenizer
        self.golden_syms = {}
        self.strategy: tf.distribute.Strategy = tf.distribute.get_strategy()
        self.metrics = metrics
        self.convert_to_symbolic = False
        for metric in metrics:
            self.convert_to_symbolic |= metric.TYPE == MetricType.SYMBOLIC
        self.search = search
        self.name = name
        self.evaluate_each = evaluate_each
        self.max_num = max_num

    def convert_golden(self, data):
        equations = []
        golden_preorder = self.tokenizer.decode(
            tf.cast(data["symbolic_expr_target"], tf.int64)
        ).numpy()
        golden_consts = convert_to_ndarray(data["constants_target"])
        for golden, golden_const in zip(golden_preorder, golden_consts):
            golden = convert_to_ndarray(clean_expr(golden, self.tokenizer))
            _, golden_inorder = prefix_to_infix(
                golden, golden_const.flatten(), self.tokenizer
            )
            equations.append(golden_inorder)

        return equations

    def reset_metric_states(self, epoch):
        for m in self.model.metrics:
            m.reset_states()

        for m in self.metrics:
            m.reset_states(epoch)

    def on_epoch_end(self, epoch, logs):
        if self.evaluate_each != 0 and (epoch == 0 or epoch % self.evaluate_each != 0):
            return

        for dataset_name, dataset in self.datasets.items():
            wandb_dict = {"epoch": epoch}
            self.reset_metric_states(epoch)
            all_results = []
            for data in dataset:
                outs = self.run(data)
                all_results.append(outs)
                break  # one batch

            per_replica_results = self.strategy.experimental_local_results(
                all_results
            )  # this it tuple
            for results in per_replica_results:
                for result in results:
                    out = result[0]
                    if self.search.reg_input:
                        self.model.update_metrics_from_search(
                            out["preds"],
                            out["reg_preds"],
                            out["logits"],
                            out["sym_target"],
                            out["const_target"],
                        )
                    else:
                        self.model.update_metrics_from_search(
                            out["preds"], out["logits"], out["sym_target"]
                        )
                    for out, metric in zip(result[1:], self.metrics):
                        metric.log(out)

            for key, metric in self.model.h_metrics.items():
                wandb_dict[f"{dataset_name}/{self.name}/{key}"] = metric.result()

            for metric in self.metrics:
                metric.commit(dataset_name, self.name, self.model)
            wandb.log(wandb_dict, commit=True)

    def run(self, data):
        return self.strategy.run(self.update_metrics, (data,))

    def update_metrics(self, data):
        if self.max_num != 0:
            points = data["points"][: self.max_num]
            sym_target = data["symbolic_expr_target"][: self.max_num]
            const_target = data["constants_target"][: self.max_num]
        else:
            points = data["points"]
            sym_target = data["symbolic_expr_target"]
            const_target = data["constants_target"]

        (
            preds,
            reg_preds,
            logits,
            preds_inorder,
            preds_symbolic,
        ) = self.search.batch_decode(points)
        golden_inorders = self.convert_golden(data)
        metric_results = [
            {
                "preds": preds,
                "reg_preds": reg_preds,
                "logits": logits,
                "sym_target": sym_target,
                "const_target": const_target,
            }
        ]

        for _ in self.metrics:
            metric_results.append([])

        for pred, pred_sym, golden_inorder in zip(
            preds_inorder, preds_symbolic, golden_inorders
        ):
            if golden_inorder not in self.golden_syms:
                self.golden_syms[str(golden_inorder)] = sympify(golden_inorder)

            golden_sym = self.golden_syms[str(golden_inorder)]

            for idx, m in enumerate(self.metrics):
                if m.TYPE == MetricType.INORDER:
                    metric_results[idx + 1].append(m.update(golden_inorder, pred))
                elif m.TYPE == MetricType.SYMBOLIC:
                    metric_results[idx + 1].append(m.update(golden_sym, pred_sym))
        return metric_results


class RegressionLambdaScheduler(tf.keras.callbacks.Callback):
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_epoch_begin(self, epoch, logs):
        tf.keras.backend.set_value(
            self.model.regression_loss_lambda, self.scheduler(epoch)
        )

        wandb.log(
            {
                "epoch": epoch,
                "regression_lambda": self.model.regression_loss_lambda,
            }
        )


class InputRegularizerScheduler(tf.keras.callbacks.Callback):
    def __init__(self, scheduler, regularizer):
        self.scheduler = scheduler
        self.regularizer = regularizer

    def on_epoch_begin(self, epoch, logs):
        self.regularizer.update_params(self.scheduler(epoch))
        name, value = self.regularizer.wandb_log()

        wandb.log(
            {
                "epoch": epoch,
                f"regularizer_{name}": value,
            }
        )


class ExtendedWandbCallback(WandbCallback):
    def on_epoch_end(self, epoch, logs={}):
        new_logs = {}
        for metric_name, value in logs.items():
            if metric_name[:4] == "val_":
                name = "validation/" + metric_name[4:]
            else:
                name = "train/" + metric_name
            new_logs[name] = value
        super().on_epoch_end(epoch, new_logs)


class LRCallback(tf.keras.callbacks.Callback):
    def on_train_batch_begin(self, batch, logs):
        if isinstance(
            self.model.optimizer.lr, tf.keras.optimizers.schedules.LearningRateSchedule
        ):
            current_lr = self.model.optimizer.lr(self.model.optimizer.iterations)
        else:
            current_lr = self.model.optimizer.lr

        if batch % 100 == 0:
            tf.print(current_lr)
