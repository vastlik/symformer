import traceback
from functools import reduce
from typing import List, Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
from sklearn.metrics import r2_score

from symformer.dataset.tokenizers import GeneralTermTokenizer


def stack_with_pad(values, axis=0, *, pad_value=0):
    if not values:
        return values
    max_shape = reduce(
        lambda agg, x: tuple(map(max, agg, x.shape)), values, values[0].shape
    )
    padded_values = [
        np.pad(
            x,
            tuple(
                (0, max_l - cur_l)
                for i, (max_l, cur_l) in enumerate(zip(max_shape, x.shape))
            ),
            mode="constant",
            constant_values=pad_value,
        )
        for x in values
    ]
    return np.stack(padded_values, axis=axis)


def eval_binary(left, right, operation):
    if operation == "^":
        res = tf.math.pow(left, right)
        if tf.reduce_all(tf.math.is_finite(res)):
            return res, False
        else:
            return tf.math.pow(tf.math.abs(left), right), True

    if operation == "+":
        return left + right, False

    if operation == "*":
        return left * right, False


def eval_unary(x, operation):
    if operation == "abs":
        return tf.math.abs(x), False

    if operation == "sqrt":
        res = tf.math.sqrt(x)

        if tf.reduce_all(tf.math.is_finite(res)):
            return res, False
        else:
            return tf.math.sqrt(tf.math.abs(x)), True

    if operation == "inv":
        return 1.0 / x, False

    if operation == "pow2":
        return x ** 2, False

    if operation == "pow3":
        return x ** 3, False

    if operation == "ln":
        res = tf.math.log(x)
        if tf.reduce_all(tf.math.is_finite(res)):
            return res, False
        else:
            return tf.math.log(tf.math.abs(x)), True

    if operation == "exp":
        return tf.math.exp(x), False

    if operation == "sin":
        return tf.math.sin(x), False

    if operation == "cos":
        return tf.math.cos(x), False

    if operation == "tan":
        return tf.math.tan(x), False

    if operation == "cot":
        return 1.0 / tf.math.tan(x), False

    if operation == "asin":
        return tf.math.asin(x), False

    if operation == "acos":
        return tf.math.acos(x), False

    if operation == "atan":
        return tf.math.atan(x), False

    if operation == "acot":
        return 1.0 / tf.math.atan(x), False

    if operation == "neg":
        return -1.0 * x, False

    raise ValueError(f"No operation found for {operation}.")


def eval_leaf(symbol, constant, values, tokenizer: GeneralTermTokenizer):
    if symbol == "C":
        return constant

    if "C" in symbol:
        exponent = int(symbol[1:])
        return tf.broadcast_to(constant * (10 ** exponent), [values.shape[0]])

    if symbol in tokenizer.variables:
        for i, var in enumerate(tokenizer.variables):
            if var == symbol:
                return values[:, i]

    if symbol in tokenizer.SPECIAL_INTEGERS:
        return tf.broadcast_to(float(symbol), [values.shape[0]])

    raise ValueError(f"Symbol {symbol} not found.")


def is_finite(value):
    return tf.reduce_all(tf.math.is_finite(value))


def evaluate(
    constants: Optional[List[float]],
    expr: Union[List[str], np.ndarray],
    values,
    tokenizer: GeneralTermTokenizer,
):
    constants = tf.cast(constants, tf.float32)
    values = tf.cast(values, tf.float32)

    abs_positions = []
    stack = []
    error = False
    for i in tf.range(tf.shape(expr)[0] - 1, -1, -1):
        symbol = expr[i]
        if tokenizer.is_binary(symbol):
            if len(stack) < 2:
                error = True
                break
            tmp_str, was_safe = eval_binary(stack.pop(), stack.pop(), symbol)
            if was_safe:
                abs_positions.append(i)
            stack.append(tmp_str)
        elif tokenizer.is_unary(symbol) or symbol == "abs":
            if len(stack) < 1:
                error = True
                break
            try:
                tmp_value, was_safe = eval_unary(stack.pop(), symbol)
                if was_safe:
                    abs_positions.append(i)
            except ZeroDivisionError:
                error = True
                break

            stack.append(tmp_value)
        elif tokenizer.is_leaf(symbol):
            tmp_value = eval_leaf(symbol, constants[i], values, tokenizer)
            stack.append(tmp_value)

    if len(stack) != 1 or error:
        return False, None, abs_positions

    return True, stack.pop(), abs_positions


OptimizationType = Literal["gradient", "bfgs_init", "bfgs", "no_opt"]


class ConstImprover:
    def __init__(self, x_s, y_s, tokenizer, type: OptimizationType, extended_encoding):
        self.x_s = tf.cast(x_s, tf.float32)
        self.y_s = tf.cast(y_s, tf.float32)
        self.variables = tokenizer.variables
        self.type = type
        self.tokenizer = None
        self.extended_encoding = extended_encoding

    def init_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = GeneralTermTokenizer(
                self.variables, extended_repre=self.extended_encoding
            )

    def improve(self, coeff, expr, return_partial: bool = False):
        self.init_tokenizer()
        if self.type == "gradient":
            tf_loss, tf_coeff = self.optimize_gradient(
                coeff, expr, return_partial=return_partial
            )
            loss, coeff = tf_loss.numpy(), tf_coeff.numpy()
        elif self.type == "bfgs":
            loss, coeff = self.optimize_bfgs(coeff, expr, False)
            loss, coeff = loss[np.newaxis, ...], coeff[np.newaxis, ...]
        elif self.type == "bfgs_init":
            loss, coeff = self.optimize_bfgs(coeff, expr, True)
            loss, coeff = loss[np.newaxis, ...], coeff[np.newaxis, ...]
        elif self.type == "no_opt":
            tf_loss = self.calc_loss(coeff, expr)
            loss, coeff = tf_loss.numpy()[np.newaxis, ...], coeff[np.newaxis, ...]
        else:
            raise RuntimeError(f"Optimization type {self.type} not supported")

        if not return_partial:
            loss, coeff = loss[-1:], coeff[-1:]

        exprs = []
        improved_coeffs = []
        for local_coeff in coeff:
            succ, _, safe_operations = evaluate(
                local_coeff, expr, self.x_s, self.tokenizer
            )
            if len(safe_operations) > 0:
                improved_expr = []
                improved_coeff = []
                for idx, (symbol, c) in enumerate(zip(expr, local_coeff)):
                    improved_expr.append(symbol)
                    improved_coeff.append(c)
                    if idx in safe_operations:
                        improved_expr.append("abs")
                        improved_coeff.append(0.0)

                exprs.append(improved_expr)
                improved_coeffs.append(np.array(improved_coeff, np.float32))
            else:
                exprs.append(expr)
                improved_coeffs.append(local_coeff)

        improved_coeffs = stack_with_pad(improved_coeffs, 0)
        if not return_partial:
            return loss[-1], improved_coeffs[-1], exprs[-1]
        return loss, improved_coeffs, exprs

    def optimize_bfgs(self, coef, expression, init):
        if not init:
            tmp_coef = np.random.uniform(size=len(expression))
            tmp_coef[expression != "C"] = 0  # we need to change this
            coef = tmp_coef

        results = minimize(
            lambda x_coef: self.calc_loss(x_coef, expression).numpy(),
            coef,
            method="BFGS",
        )
        if not results.success:
            return np.array(np.inf, np.float32), coef
        return np.array(results.fun, np.float32), results.x

    def calc_loss(self, coeff, expression):
        self.init_tokenizer()
        success, predicted, _ = evaluate(coeff, expression, self.x_s, self.tokenizer)
        if not success:
            return tf.convert_to_tensor(np.inf)
        predicted = tf.reshape(predicted, [-1])
        diff = tf.abs(self.y_s - predicted) ** 2
        return tf.reduce_mean(diff)

    def calc_r2(self, coeff, expression):
        self.init_tokenizer()
        success, predicted, _ = evaluate(coeff, expression, self.x_s, self.tokenizer)
        if not success or not tf.reduce_all(tf.math.is_finite(predicted)):
            return tf.convert_to_tensor(-np.inf)
        predicted = tf.broadcast_to(predicted, self.y_s.shape)
        return r2_score(self.y_s, predicted)

    def optimize(self, coef, expression, opt):
        with tf.GradientTape() as tape:
            tape.watch(coef)
            loss = self.calc_loss(coef, expression)

        if tf.reduce_all(coef == 0.0) or not tf.math.is_finite(loss):
            return loss

        grads = tape.gradient(loss, coef)
        opt.apply_gradients(zip([grads], [coef]))
        return self.calc_loss(coef, expression)

    def optimize_gradient(self, coef, expression, *, return_partial: bool = False):
        coef = tf.Variable(coef)
        losses, coefs = [], []
        best_coef = tf.identity(coef)
        best_loss = self.calc_loss(coef, expression)
        coefs.append(best_coef)
        losses.append(best_loss)
        lr = 1e-3
        opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, clipnorm=10.0)
        not_improved = 0
        i = 0
        while True:
            try:
                loss = self.optimize(coef, expression, opt)
            except ValueError as e:
                print(traceback.print_exc())
                print(e)
                print(expression)
                break
            not_improved += 1
            if loss < best_loss:
                if 1 - loss / best_loss > 0.001:
                    # need to improve for at least 0.1 percent
                    not_improved = 0
                else:
                    lr /= 10
                    opt.lr.assign(lr)

                best_loss = loss
                best_coef = tf.identity(coef)
                if return_partial:
                    coefs.append(best_coef.numpy())
                    losses.append(best_loss)
                else:
                    coefs[-1] = coef
                    losses[-1] = best_loss

            if not_improved >= 5:
                break

            if not tf.reduce_all(tf.math.is_finite(coef)) or not tf.math.is_finite(
                loss
            ):
                break

            if tf.reduce_all(coef == 0.0):
                break
            i += 1

        losses = tf.stack(losses, 0)
        coefs = tf.stack(coefs, 0)
        return losses, coefs
