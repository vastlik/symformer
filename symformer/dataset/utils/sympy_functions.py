from typing import List

import numpy as np
from sympy import lambdify


def expr_to_func(sympy_expr, variables: List[str]):
    def cot(x):
        return 1 / np.tan(x)

    def acot(x):
        return 1 / np.arctan(x)

    def coth(x):
        return 1 / np.tanh(x)

    return lambdify(
        variables,
        sympy_expr,
        modules=["numpy", {"cot": cot, "acot": acot, "coth": coth}],
    )


def evaluate_points(func, points):
    y = func(*[points[:, i] for i in range(points.shape[1])])
    y = np.reshape(y, (-1, 1))
    if y.shape[0] != points.shape[0]:
        y = np.broadcast_to(y, (points.shape[0], 1))
    if np.any(np.iscomplex(y)):
        return np.broadcast_to(np.inf, (points.shape[0], 1))
    return y.astype(np.float64)
