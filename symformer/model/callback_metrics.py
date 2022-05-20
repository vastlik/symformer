import enum
import io
import os
from typing import List

import numpy as np

import PIL
import tensorflow as tf
import wandb
from matplotlib import pyplot as plt
from scipy.integrate import trapz
from sklearn import metrics

from ..dataset.utils import (
    generate_all_possible_extrapolation_ranges,
    generate_all_possible_ranges,
)
from ..dataset.utils.sympy_functions import evaluate_points, expr_to_func


class MetricType(enum.Enum):
    INORDER = "inorder"
    SYMBOLIC = "symbolic"


class CallbackMetric:
    TYPE = None

    def __init__(self, variables: List[str]):
        self.variables = variables

    def reset_states(self, epoch):
        raise NotImplementedError()

    def update(self, golden, prediction):
        raise NotImplementedError()

    def log(self, logs):
        raise NotImplementedError()

    def commit(self, dataset_name: str, search_name: str, model: tf.keras.Model):
        raise NotImplementedError()


class TableCallbackMetric(CallbackMetric):
    TYPE = MetricType.INORDER

    def __init__(self, variables: List[str]):
        super().__init__(variables)
        self.table = None
        self.wandb_log = None

    def reset_states(self, epoch):
        self.table = wandb.Table(columns=["golden", "prediction"])
        self.wandb_log = {
            "epoch": epoch,
        }

    def update(self, golden, prediction):
        if golden is None or prediction is None:
            return {"golden": None, "pred": None}

        return {"golden": str(golden), "pred": str(prediction)}

    def log(self, logs):
        for log in logs:
            self.table.add_data(log["golden"], log["pred"])

    def commit(self, dataset_name: str, search_name: str, model: tf.keras.Model):
        self.wandb_log[f"{dataset_name}/{search_name}/examples"] = self.table
        wandb.log(self.wandb_log, commit=False)


class IntegralDifferenceMetric(CallbackMetric):
    TYPE = MetricType.SYMBOLIC

    def __init__(self, variables: List[str], per_epoch=0):
        super().__init__(variables)
        self.wandb_log = None
        self.function_diffs = None
        self.integral_goldens = None
        self.epoch = None
        self.per_epoch = per_epoch
        self.golden_integrals = {}
        self.integral_goldens_results = []

        self.points_ranges = [(-5, 5), (0, 5), (-5, 0)]

    def reset_states(self, epoch):
        self.epoch = epoch
        self.wandb_log = {
            "epoch": epoch,
        }
        self.function_diffs = []
        self.integral_goldens_results = []

    def update(self, golden, prediction):
        results = {"diff": None, "golden": None}
        if (
            not (self.per_epoch == 0 or self.epoch % self.per_epoch == 0)
            or len(self.variables) > 1
        ):
            return results

        if golden is None or prediction is None:
            return results

        np.seterr(all="raise")
        integral, integral_golden = self.calculate_diff_integrals(golden, prediction)
        if integral is not None:
            results["diff"] = integral
            results["golden"] = integral_golden

        return results

    def log(self, logs):
        for log in logs:
            if log["diff"] is not None:
                self.function_diffs.append(log["diff"])
                self.integral_goldens_results.append(log["golden"])

    def commit(self, dataset_name: str, search_name: str, model: tf.keras.Model):
        if len(self.integral_goldens_results) != 0:
            self.wandb_log[f"{dataset_name}/{search_name}/integral_diff"] = np.median(
                self.function_diffs
            )
            self.wandb_log[f"{dataset_name}/{search_name}/integral_golden"] = np.median(
                self.integral_goldens_results
            )
            self.wandb_log[f"{dataset_name}/{search_name}/number_of_integrals"] = len(
                self.integral_goldens_results
            )
            wandb.log(self.wandb_log, commit=False)

    def calculate_diff_integrals(self, golden_sympy, pred_sympy):
        try:
            diff = expr_to_func(abs(golden_sympy - pred_sympy), self.variables)
            if str(golden_sympy) not in self.golden_integrals:
                golden_lambda = expr_to_func(
                    abs(golden_sympy), self.variables
                )  # this stays same and can be cached
                integral_golden = self.integrate(golden_lambda)
                self.golden_integrals[str(golden_sympy)] = integral_golden
            else:
                integral_golden = self.golden_integrals[str(golden_sympy)]

            integral = self.integrate(diff)
            if np.isfinite(integral) and np.isreal(integral):
                return integral, integral_golden

        except (
            TypeError,
            ZeroDivisionError,
            OverflowError,
            RuntimeWarning,
            FloatingPointError,
            KeyError,
        ):
            return None, None

    def integrate(self, func):
        num_points = 1000
        if len(self.variables) == 2:
            x = np.linspace(-1, 1, num_points)
            y = np.linspace(-1, 1, num_points)
            z = func(x.reshape((-1, 1)), y.reshape((1, -1)))
            z = np.broadcast_to(z, (num_points, num_points))
            return trapz([trapz(z_x, x) for z_x in z], y)
        elif len(self.variables) == 1:
            for left, right in self.points_ranges:
                try:
                    x = np.linspace(left, right, num_points)
                    y = evaluate_points(func, np.reshape(x, (-1, 1)))
                    y = y.reshape(-1)
                    if np.any(np.isnan(y)):
                        raise RuntimeWarning("Is nan!")

                    return trapz(y, x)
                except (
                    TypeError,
                    ZeroDivisionError,
                    OverflowError,
                    RuntimeWarning,
                    FloatingPointError,
                ):
                    raise RuntimeWarning(e)
            raise RuntimeWarning("No range for integration found")


class PointMetrics(CallbackMetric):
    TYPE = MetricType.SYMBOLIC

    def __init__(
        self, eps: List[float], num_points: int, variables: List[str], save_model=True
    ):
        super().__init__(variables)
        self.golden_points = {}
        self.eps = eps
        self.save_model = save_model
        self.best_r2 = -np.inf

        self.num_points = num_points
        self.point_dim = len(variables)
        self.golden_interpolation = []
        self.golden_extrapolation = []
        self.prediction_interpolation = []
        self.prediction_extrapolation = []
        self.wandb_log = {}

        self.interpolation_barriers = (-5, 5)

    def reset_states(self, epoch):
        self.golden_interpolation = []
        self.golden_extrapolation = []
        self.prediction_interpolation = []
        self.prediction_extrapolation = []
        self.wandb_log = {"epoch": epoch}

    def update(self, golden, pred):
        results = {
            "interpolation": {
                "valid_length": 0,
                "golden": [],
                "prediction": [],
            },
            "extrapolation": {
                "valid_length": 0,
                "golden": [],
                "prediction": [],
            },
        }
        if golden is None or pred is None:
            return self.pad(results)

        try:
            str_golden = str(golden)
            if str_golden not in self.golden_points:
                golden_lambda = expr_to_func(golden, self.variables)
                points, res, ranges = self.calculate_interpolation(golden_lambda)
                self.golden_points[str_golden] = {
                    "interpolation": (points, res),
                    "extrapolation": self.calculate_extrapolation(
                        golden_lambda, ranges
                    ),
                }

            golden_points = self.golden_points[str_golden]

            pred = expr_to_func(pred, self.variables)
            if golden_points["interpolation"][0] is not None:
                results["interpolation"]["golden"] = golden_points["interpolation"][1]
                results["interpolation"]["prediction"] = evaluate_points(
                    pred, golden_points["interpolation"][0]
                )
                results["interpolation"]["valid_length"] = len(
                    golden_points["interpolation"][1]
                )

            if golden_points["extrapolation"][0] is not None:
                results["extrapolation"]["golden"] = golden_points["extrapolation"][1]
                results["extrapolation"]["prediction"] = evaluate_points(
                    pred, golden_points["extrapolation"][0]
                )
                results["extrapolation"]["valid_length"] = len(
                    golden_points["extrapolation"][1]
                )

        except (
            AttributeError,
            KeyError,
            RuntimeWarning,
            NameError,
            TypeError,
            RuntimeError,
            OverflowError,
            ZeroDivisionError,
            MemoryError,
            ValueError,
            FloatingPointError,
        ):
            pass
        return self.pad(results)

    def pad(self, results):
        inter = results["interpolation"]
        inter["golden"] = np.pad(
            inter["golden"], (0, self.num_points - len(inter["golden"]))
        )
        inter["prediction"] = np.pad(
            inter["prediction"], (0, self.num_points - len(inter["prediction"]))
        )

        extr = results["extrapolation"]
        extr["golden"] = np.pad(
            extr["golden"], (0, self.num_points - len(extr["golden"]))
        )
        extr["prediction"] = np.pad(
            extr["prediction"], (0, self.num_points - len(extr["prediction"]))
        )

        results["interpolation"] = inter
        results["extrapolation"] = extr
        return results

    def log(self, logs):
        for log in logs:
            name = "interpolation"
            length = log[name]["valid_length"]
            if length != 0:
                self.golden_interpolation.append(log[name]["golden"][:length])
                self.prediction_interpolation.append(log[name]["prediction"][:length])

            name = "extrapolation"
            length = log[name]["valid_length"]
            if length != 0:
                self.golden_extrapolation.append(log[name]["golden"][:length])
                self.prediction_extrapolation.append(log[name]["prediction"][:length])

    def commit(self, dataset_name: str, search_name: str, model: tf.keras.Model):
        maes = []
        relative_errors = []
        r2s = []
        for pred, golden in zip(
            self.prediction_interpolation, self.golden_interpolation
        ):
            pred = tf.reshape(pred, [-1])
            golden = tf.reshape(golden, [-1])
            mask = np.isfinite(pred)
            pred = pred[mask]
            golden = golden[mask]
            if len(pred) == 0 or len(golden) == 0:
                continue
            mae = self.mae(golden, pred)
            relative_error = mae / (np.abs(golden) + 1e-8)
            r2 = metrics.r2_score(golden, pred)
            maes.append(np.nanmean(mae))
            relative_errors.append(np.nanmean(relative_error))
            r2s.append(r2)

        if len(maes) != 0:
            r2median = np.nanmedian(r2s)
            if r2median > self.best_r2 and search_name == "no_teacher":
                model.save_weights(
                    f'{os.getenv("OUTPUT")}/checkpoints/{os.getenv("JOB_ID")}/weights',
                    overwrite=True,
                )
                tf.print(
                    f"New best median MAE {r2median} found. Previously {self.best_r2}. Overwriting."
                )
                self.best_r2 = r2median

            model.save_weights(
                f'{os.getenv("OUTPUT")}/checkpoints/{os.getenv("JOB_ID")}/last',
                overwrite=True,
            )

            self._log(
                maes, relative_errors, r2s, dataset_name, search_name, "interpolation"
            )

        maes = []
        relative_errors = []
        r2s = []
        for pred, golden in zip(
            self.prediction_extrapolation, self.golden_extrapolation
        ):
            pred = tf.reshape(pred, [-1])
            golden = tf.reshape(golden, [-1])
            mask = np.isfinite(pred)
            pred = pred[mask]
            golden = golden[mask]
            if len(pred) == 0 or len(golden) == 0:
                continue
            mae = self.mae(golden, pred)
            relative_error = mae / (np.abs(golden) + 1e-8)
            r2 = metrics.r2_score(golden, pred)
            maes.append(np.nanmean(mae))
            relative_errors.append(np.nanmean(relative_error))
            r2s.append(r2)

        if len(maes) != 0:
            self._log(
                maes, relative_errors, r2s, dataset_name, search_name, "extrapolation"
            )

        wandb.log(self.wandb_log, commit=False)

    def _log(self, maes, relative_error, r2, dataset_name, search_name, metric_type):
        if len(maes) != 0:
            self.wandb_log[
                f"{dataset_name}/{search_name}/points_mae_{metric_type}"
            ] = np.nanmedian(maes)
            self.wandb_log[
                f"{dataset_name}/{search_name}/points_mae_{metric_type}_mean"
            ] = np.nanmean(maes)

            self.wandb_log[
                f"{dataset_name}/{search_name}/points_relative_error_{metric_type}"
            ] = np.nanmedian(relative_error)
            self.wandb_log[
                f"{dataset_name}/{search_name}/points_relative_error_{metric_type}_mean"
            ] = np.nanmean(relative_error)

            self.wandb_log[
                f"{dataset_name}/{search_name}/points_accuracy_{metric_type}_image"
            ] = self.plot_image(maes)
            self.wandb_log[
                f"{dataset_name}/{search_name}/points_relative_error_{metric_type}_image"
            ] = self.plot_image(relative_error)

            self.wandb_log[
                f"{dataset_name}/{search_name}/points_accuracy_{metric_type}_auc"
            ] = self.auc(maes)
            self.wandb_log[
                f"{dataset_name}/{search_name}/points_relative_error_{metric_type}_auc"
            ] = self.auc(relative_error)

            self.wandb_log[
                f"{dataset_name}/{search_name}/points_r2_{metric_type}"
            ] = np.nanmedian(r2)
            self.wandb_log[
                f"{dataset_name}/{search_name}/points_r2_{metric_type}_mean"
            ] = np.nanmean(r2)

            for eps in self.eps:
                self.wandb_log[
                    f"{dataset_name}/{search_name}/points_accuracy_{metric_type}_{eps}"
                ] = np.nanmean(np.array(maes) < eps)

                self.wandb_log[
                    f"{dataset_name}/{search_name}/points_relative_error_{metric_type}_{eps}"
                ] = np.nanmean(np.array(relative_error) < eps)

    def auc(self, errors):
        eps = np.logspace(-4, 1, 500)
        values = []
        for epsilon in eps:
            values.append(np.nanmean(np.array(errors) < epsilon))
        return metrics.auc(eps, values)

    def plot_image(self, errors):
        eps = np.logspace(-4, 1, 100)
        values = []
        for epsilon in eps:
            values.append(np.nanmean(np.array(errors) < epsilon))

        fig, ax = plt.subplots()
        ax.plot(eps, values)
        return wandb.Image(get_image(fig))

    def calculate_interpolation(self, golden_lambda):
        for left, right in generate_all_possible_ranges(
            self.point_dim,
            self.interpolation_barriers[0],
            self.interpolation_barriers[1],
        ):
            try:
                points = np.random.uniform(
                    left, right, (self.num_points, self.point_dim)
                )
                results = evaluate_points(golden_lambda, points)

                if np.any(np.isnan(results)) or np.any(np.isinf(results)):
                    continue
                return points, results, (left, right)
            except (
                AttributeError,
                KeyError,
                RuntimeWarning,
                NameError,
                TypeError,
                RuntimeError,
                OverflowError,
                ZeroDivisionError,
                MemoryError,
                ValueError,
                FloatingPointError,
            ):
                continue
        return None, None, None

    def calculate_extrapolation(self, golden_lambda, ranges):
        left = generate_all_possible_extrapolation_ranges(
            ranges[0],
            self.interpolation_barriers[0],
            self.interpolation_barriers[0] - 1,
        )
        right = generate_all_possible_extrapolation_ranges(
            ranges[1],
            self.interpolation_barriers[1],
            self.interpolation_barriers[1] + 1,
        )
        try:
            sample = 1000
            while True:
                points = np.random.uniform(left, right, (sample, self.point_dim))
                mask = np.any(points <= ranges[0], axis=-1) | np.any(
                    points >= ranges[1], axis=-1
                )
                points = points[mask]
                if len(points) >= self.num_points:
                    points = points[: self.num_points]
                    break
                sample += 5000
            results = evaluate_points(golden_lambda, points)

            if np.any(np.isnan(results)) or np.any(np.isinf(results)):
                return None, None

            return points, results
        except (
            AttributeError,
            KeyError,
            RuntimeWarning,
            NameError,
            TypeError,
            RuntimeError,
            OverflowError,
            ZeroDivisionError,
            MemoryError,
            ValueError,
            FloatingPointError,
        ):
            return None, None

    def mae(self, a, b):
        if tf.is_tensor(a):
            a = a.numpy()

        if tf.is_tensor(b):
            b = b.numpy()
        return np.abs(a.astype(float) - b.astype(float))


def get_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return PIL.Image.open(buf)


class FuncVisualizer(CallbackMetric):
    TYPE = MetricType.SYMBOLIC

    def __init__(self, variables: List[str], num_sampled_points=1000):
        super().__init__(variables)
        self.wandb_log = {}
        self.golden_points = {}
        self.images = []
        self.num_sampled_points = num_sampled_points
        self.points_ranges = [(-5, 5), (0, 5), (-5, 0)]
        self.max_num_of_calls = 20

    def reset_states(self, epoch):
        self.wandb_log = {"epoch": epoch}
        self.images = []
        self.max_num_of_calls = 20

    def update(self, golden, prediction):
        results = {"image": None}
        if (
            self.max_num_of_calls <= 0
            or golden is None
            or prediction is None
            or len(self.variables) > 1
        ):
            return results

        try:
            str_golden = str(golden)
            if str_golden not in self.golden_points:
                golden_lambda = expr_to_func(golden, self.variables)
                x, y = self.calc_points(golden_lambda)
                self.golden_points[str_golden] = (x, y)

            golden_x, golden_y = self.golden_points[str_golden]
            if golden_x is None or golden_y is None:
                return results

            pred = expr_to_func(prediction, self.variables)
            pred_y = evaluate_points(pred, golden_x)
            fig, ax = plt.subplots()
            ax.scatter(
                golden_x, golden_y, color="blue", s=2, label=f"Golden: {str_golden}"
            )
            ax.scatter(
                golden_x, pred_y, color="orange", s=2, label=f"Pred: {str(prediction)}"
            )
            ax.legend()
            pil_image = get_image(fig)

            plt.close("all")
            plt.cla()
            plt.clf()
            self.max_num_of_calls -= 1
            return {"image": pil_image}

        except (
            TypeError,
            ZeroDivisionError,
            OverflowError,
            RuntimeWarning,
            FloatingPointError,
            ValueError,
            KeyError,
        ):
            return {"image": None}

    def calc_points(self, golden):
        for left, right in self.points_ranges:
            try:
                points = np.random.uniform(left, right, self.num_sampled_points)
                points = points.reshape((-1, len(self.variables)))
                results = evaluate_points(golden, points)

                if np.any(np.isnan(results)) or np.any(np.isinf(results)):
                    continue
                return points, results
            except (
                AttributeError,
                KeyError,
                RuntimeWarning,
                NameError,
                TypeError,
                RuntimeError,
                OverflowError,
                ZeroDivisionError,
                MemoryError,
                ValueError,
                FloatingPointError,
            ):
                continue
        return None, None

    def log(self, logs):
        for log in logs:
            if log["image"] is not None and len(self.images) < 40:
                self.images.append(log["image"])

    def commit(self, dataset_name: str, search_name: str, model: tf.keras.Model):
        if len(self.images) != 0:
            self.wandb_log[f"{dataset_name}/{search_name}/images"] = [
                wandb.Image(image) for image in self.images
            ]
            wandb.log(self.wandb_log, commit=False)
