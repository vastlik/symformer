import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import tensorflow as tf
from sympy import sympify, SympifyError

from symformer import dataset
from symformer.dataset.tokenizers import Tokenizer
from symformer.dataset.utils.tree import prefix_to_infix
from symformer.model.utils.const_improver import ConstImprover, OptimizationType


def find_first_occurrence(arr, tokenizer):
    indexes = (arr == tokenizer.end) | (arr == tokenizer.start) | (arr == tokenizer.pad)
    indexes = np.reshape(indexes, (1, -1))
    if indexes.any():
        return np.argmax(indexes)

    return arr.shape[0]


def clean_expr(expr, tokenizer):
    expr = np.array([c.decode("utf-8") for c in expr])
    return expr[: find_first_occurrence(expr, tokenizer)]


def get_pred_symbolic(pred_inorder):
    try:
        return True, sympify(pred_inorder, evaluate=False)
    except (SympifyError, MemoryError, OverflowError):
        return False, None


class Convertor:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def convert(self, preds, reg_preds, points, logits, return_all: bool = False):
        raise NotImplementedError()


class BestFittingFilter(Convertor):
    def __init__(
        self,
        tokenizer: Tokenizer,
        optimization_type: OptimizationType = "no_opt",
        use_pool=False,
        early_stopping=False,
        extended_repre=True,
    ):
        super().__init__(tokenizer)
        self.convertor = SimpleConvertor(self.tokenizer)
        self.optimization_type = optimization_type
        self.use_pool = use_pool
        self.early_stopping = early_stopping
        self.extended_repre = extended_repre

    def convert_representation(self, coeffs, expression):
        int_map = [c in self.tokenizer.SPECIAL_INTEGERS for c in expression]
        padded_int_map = np.pad(
            int_map, (0, len(coeffs) - len(int_map)), constant_values=False
        )

        coeffs[padded_int_map] = expression[int_map]
        if self.extended_repre:
            expression[int_map] = "C0"
            for i in range(len(expression)):
                if expression[i][0] == "C":
                    coeffs[i] *= 10 ** int(expression[i][1:])
                    expression[i] = "C0"
        else:
            expression[int_map] = "C"

        return coeffs, expression

    def get_unique_results(self, preds, reg_preds, logits):
        unique_preds = []
        unique_reg_preds = []
        unique_logits = []
        if reg_preds is not None:
            _, indexes = np.unique(
                np.concatenate([preds, reg_preds], axis=1), axis=0, return_index=True
            )
        else:
            _, indexes = np.unique(preds, axis=0, return_index=True)

        for idx in indexes:
            unique_preds.append(preds[idx])
            unique_logits.append(logits[idx])
            if reg_preds is not None:
                unique_reg_preds.append(reg_preds[idx])

        unique_preds = tf.convert_to_tensor(unique_preds)
        tokenized_preds = self.tokenizer.decode(tf.cast(unique_preds, tf.int64)).numpy()
        unique_logits = tf.convert_to_tensor(unique_logits)
        if reg_preds is not None:
            unique_reg_preds = tf.convert_to_tensor(unique_reg_preds, dtype=tf.float32)
        else:
            unique_reg_preds = None

        return unique_preds, tokenized_preds, unique_logits, unique_reg_preds

    def convert(self, preds, reg_preds, points, logits, return_all: bool = False):
        points = points.numpy()
        (
            unique_preds,
            tokenized_preds,
            unique_logits,
            unique_reg_preds,
        ) = self.get_unique_results(preds, reg_preds, logits)
        points = tf.cast(points, tf.float32)

        optimizer = ConstImprover(
            points[:, :-1],
            points[:, -1],
            self.tokenizer,
            self.optimization_type,
            self.extended_repre,
        )
        cleaned_expressions = []
        arguments = []
        losses = []
        coeffs = []
        expressions = []

        for i, pred in enumerate(tokenized_preds):
            cleaned = clean_expr(pred, self.tokenizer)
            if unique_reg_preds is not None:
                coeff, expr = self.convert_representation(
                    unique_reg_preds[i].numpy(), cleaned
                )
                if not isinstance(coeff, np.ndarray):
                    coeff = coeff.numpy()
            else:
                expr = cleaned
                coeff = None
            if not self.use_pool:
                loss, coeff, expr = optimizer.improve(
                    coeff, expr, return_partial=return_all
                )
                losses.append(loss)
                coeffs.append(coeff)
                expressions.append(expr)
            else:
                arguments.append((coeff, expr))

            cleaned_expressions.append(cleaned)

            if not return_all and (
                self.early_stopping and optimizer.calc_r2(coeff, expr) > 0.9999
            ):
                break
        if self.use_pool:
            if self.early_stopping:
                with Pool(os.cpu_count()) as p:
                    losses = p.starmap(optimizer.calc_r2, arguments)

                mini = np.nanargmax(losses)
                if losses[mini] < 0.9999:
                    with Pool(os.cpu_count()) as p:
                        res = p.starmap(
                            partial(optimizer.improve, return_partial=return_all),
                            arguments,
                        )
                    losses, coeffs, expressions = tuple(zip(*res))
                else:
                    losses = -1 * np.array(losses)
            else:
                with Pool(32) as p:
                    res = p.starmap(
                        partial(optimizer.improve, return_partial=return_all), arguments
                    )
                losses, coeffs, expressions = tuple(zip(*res))

        if not return_all:
            best = np.nanargmin(losses)

            best_coeff = coeffs[best]
            best_pred = unique_preds[best]
            best_logits = unique_logits[best]
            best_preorder = expressions[best]

            _, inorder = prefix_to_infix(best_preorder, best_coeff, self.tokenizer)
            _, symbolic = get_pred_symbolic(inorder)

            return best_pred, best_coeff, best_logits, inorder, symbolic
        else:
            best = np.argsort([lseries[-1] for lseries in losses])
            best_coeff = [coeffs[i] for i in best]
            best_pred = [unique_preds[i] for i in best]
            best_logits = [unique_logits[i] for i in best]
            best_preorder = [expressions[i] for i in best]
            inorder = []
            symbolic = []

            for bp_series, bc_series in zip(best_preorder, best_coeff):
                symbolic_series, inorder_series = [], []
                inorder.append(inorder_series)
                symbolic.append(symbolic_series)
                for bp, bc in zip(bp_series, bc_series):
                    _, local_inorder = prefix_to_infix(bp, bc, self.tokenizer)
                    _, local_symbolic = get_pred_symbolic(local_inorder)
                    inorder_series.append(local_inorder)
                    symbolic_series.append(local_symbolic)

            return best_pred, best_coeff, best_logits, inorder, symbolic


class SimpleConvertor(Convertor):
    def convert(self, preds, reg_preds, points, logits, return_all: bool = False):
        assert not return_all
        inorder = []
        symbolic = []

        tokenized_preds = self.tokenizer.decode(tf.cast(preds, tf.int64)).numpy()
        for pred, reg_pred in zip(tokenized_preds, reg_preds.numpy()):
            try:
                prediction = clean_expr(pred, self.tokenizer)
                success, pred_inorder = prefix_to_infix(
                    prediction, reg_pred, self.tokenizer
                )
                pred_sym = None
                if success:
                    success, pred_sym = get_pred_symbolic(pred_inorder)

                inorder.append(pred_inorder)
                symbolic.append(pred_sym)
            except dataset.utils._common.TimeoutError:
                print("TIMEOUT")
                pass

        return preds, reg_preds, logits, inorder, symbolic


class FitConstantsConvertor(Convertor):
    def __init__(self, tokenizer: Tokenizer, extended_repre):
        super().__init__(tokenizer)
        self.optimization_type = "bfgs"
        self.extended_repre = extended_repre

    def convert(self, preds, reg_preds, points, logits, return_all: bool = False):
        assert not return_all
        inorder = []
        symbolic = []

        tokenized_preds = self.tokenizer.decode(tf.cast(preds, tf.int64)).numpy()
        for i, pred in enumerate(tokenized_preds):
            optimizer = ConstImprover(
                points[i, :, :-1],
                points[i, :, -1],
                self.tokenizer,
                self.optimization_type,
                self.extended_repre,
            )
            try:
                prediction = clean_expr(pred, self.tokenizer)
                _, coeff, expr = optimizer.improve(None, prediction)
                success, pred_inorder = prefix_to_infix(
                    prediction, coeff, self.tokenizer
                )
                pred_sym = None
                if success:
                    success, pred_sym = get_pred_symbolic(pred_inorder)

                inorder.append(pred_inorder)
                symbolic.append(pred_sym)
            except dataset.utils._common.TimeoutError:
                print("TIMEOUT")
                pass

        return preds, reg_preds, logits, inorder, symbolic
