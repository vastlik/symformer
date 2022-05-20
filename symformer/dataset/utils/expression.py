# algorithm based on https://arxiv.org/pdf/1912.01412.pdf
import time
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import sympy
from sympy import expand

from ..tokenizers import GeneralTermTokenizer
from . import generate_all_possible_ranges, timeout, TimeoutError
from .sympy_functions import evaluate_points, expr_to_func
from .tree import convert_to_binary_tree, prefix_to_infix, Tree


@dataclass
class Operator:
    arity: int
    weight: int


class ExpressionGenerator:
    OPERATORS = {
        # Elementary functions
        "+": Operator(2, 8),
        "-": Operator(2, 5),
        "*": Operator(2, 8),
        "/": Operator(2, 5),
        "^": Operator(2, 2),
        "pow2": Operator(1, 8),
        "pow3": Operator(1, 6),
        "pow4": Operator(1, 4),
        "pow5": Operator(1, 4),
        "pow6": Operator(1, 3),
        "inv": Operator(1, 8),
        "sqrt": Operator(1, 8),
        "exp": Operator(1, 2),
        "ln": Operator(1, 4),
        # 'abs': Operator(1, 2),
        # Trigonometric Functions
        "sin": Operator(1, 4),
        "cos": Operator(1, 4),
        "tan": Operator(1, 2),
        "cot": Operator(1, 2),
        # Inverse functions
        "asin": Operator(1, 1),
        "acos": Operator(1, 1),
        "atan": Operator(1, 1),
        "acot": Operator(1, 1),
    }

    def __init__(self, max_ops, rng: np.random.Generator, variables=None):
        self.variables = variables if variables is not None else ["x"]
        self.nl = 1  # self.n_leaves
        self.p1 = 1  # len(self.una_ops)
        self.p2 = 1  # len(self.bin_ops)
        self.max_ops = max_ops
        self.leaf_probs = [20, 10, 10, 1]
        self.leaf_probs = self.leaf_probs / np.sum(self.leaf_probs)

        self.ubi_dist = self.generate_ubi_dist()
        self.una_ops = []
        self.bin_ops = []
        for op_name, op in self.OPERATORS.items():
            if op.arity == 1:
                self.una_ops.append(op_name)
            else:
                self.bin_ops.append(op_name)
        self.una_ops_probs = self.convert_to_dist(self.una_ops)
        self.bin_ops_probs = self.convert_to_dist(self.bin_ops)
        self.rng = rng

    def convert_to_dist(self, operations: List[str]):
        array = []
        for operation in operations:
            array.append(self.OPERATORS[operation].weight)
        array = np.array(array)
        return array / array.sum()

    def generate_ubi_dist(self):
        """
        Copied from https://github.com/facebookresearch/SymbolicMathematics/blob/4596d070e1a9a1c2239c923d7d68fda577c8c007/src/envs/char_sp.py
        `max_ops`: maximum number of operators
        Enumerate the number of possible unary-binary trees that can be generated from empty nodes.
        D[e][n] represents the number of different binary trees with n nodes that
        can be generated from e empty nodes, using the following recursion:
            D(0, n) = 0
            D(e, 0) = L ** e
            D(e, n) = L * D(e - 1, n) + p_1 * D(e, n - 1) + p_2 * D(e + 1, n - 1)
        """
        # enumerate possible trees
        # first generate the tranposed version of D, then transpose it
        D = [[0] + ([self.nl ** i for i in range(1, 2 * self.max_ops + 1)])]
        for n in range(1, 2 * self.max_ops + 1):  # number of operators
            s = [0]
            for e in range(1, 2 * self.max_ops - n + 1):  # number of empty nodes
                s.append(
                    self.nl * s[e - 1]
                    + self.p1 * D[n - 1][e]
                    + self.p2 * D[n - 1][e + 1]
                )
            D.append(s)
        assert all(len(D[i]) >= len(D[i + 1]) for i in range(len(D) - 1))
        D = [
            [D[j][i] for j in range(len(D)) if i < len(D[j])]
            for i in range(max(len(x) for x in D))
        ]
        return D

    def get_leaf(self, max_int: int):
        """
        Generate a leaf.
        """
        leaf_type = self.rng.choice(len(self.leaf_probs), p=self.leaf_probs)
        if leaf_type == 0:
            return self.rng.choice(self.variables)
        elif leaf_type == 1:
            num = self.rng.integers(1, max_int + 1)
            if self.rng.uniform() <= 0.5:
                return -num
            return num
        elif leaf_type == 2:
            return np.round(self.rng.uniform(-max_int, max_int), 4)
        else:
            return 0

    def sample_next_pos_ubi(self, nb_empty, nb_ops):
        """
        Sample the position of the next node (unary-binary case).
        Sample a position in {0, ..., `nb_empty` - 1}, along with an arity.
        """
        assert nb_empty > 0
        assert nb_ops > 0
        probs = []
        for i in range(nb_empty):
            probs.append(
                (self.nl ** i) * self.p1 * self.ubi_dist[nb_empty - i][nb_ops - 1]
            )
        for i in range(nb_empty):
            probs.append(
                (self.nl ** i) * self.p2 * self.ubi_dist[nb_empty - i + 1][nb_ops - 1]
            )
        probs = [p / self.ubi_dist[nb_empty][nb_ops] for p in probs]
        probs = np.array(probs, dtype=np.float64)
        e = self.rng.choice(2 * nb_empty, p=probs)
        arity = 1 if e < nb_empty else 2
        e = e % nb_empty
        return e, arity

    def generate_expr(self, nb_total_ops, max_int):
        """
        Copied from https://github.com/facebookresearch/SymbolicMathematics/blob/4596d070e1a9a1c2239c923d7d68fda577c8c007/src/envs/char_sp.py
        Create a tree with exactly `nb_total_ops` operators.
        """
        stack = [None]
        nb_empty = 1  # number of empty nodes
        l_leaves = 0  # left leaves - None states reserved for leaves
        t_leaves = 1  # total number of leaves (just used for sanity check)

        # create tree
        last_op = None
        for nb_ops in range(nb_total_ops, 0, -1):

            # next operator, arity and position
            skipped, arity = self.sample_next_pos_ubi(nb_empty, nb_ops)
            if arity == 1:
                op = self.rng.choice(self.una_ops, p=self.una_ops_probs)  # add probs
            else:
                op = self.rng.choice(self.bin_ops, p=self.bin_ops_probs)  # add probs

            nb_empty += (
                self.OPERATORS[op].arity - 1 - skipped
            )  # created empty nodes - skipped future leaves
            t_leaves += self.OPERATORS[op].arity - 1  # update number of total leaves
            l_leaves += skipped  # update number of left leaves

            # update tree
            pos = [i for i, v in enumerate(stack) if v is None][l_leaves]
            stack = (
                stack[:pos]
                + [op]
                + [None for _ in range(self.OPERATORS[op].arity)]
                + stack[pos + 1 :]
            )

        _ = last_op
        leaves = [self.get_leaf(max_int) for _ in range(t_leaves)]
        self.rng.shuffle(leaves)

        # insert leaves into tree
        for pos in range(len(stack) - 1, -1, -1):
            if stack[pos] is None:
                stack = stack[:pos] + [leaves.pop()] + stack[pos + 1 :]
        assert len(leaves) == 0
        return [str(s) for s in stack]

    def rewrite(self, arg, x):
        if x == "pow2":
            return f"(({arg})^2)"
        elif x == "pow3":
            return f"(({arg})^3)"
        elif x == "pow4":
            return f"(({arg})^4)"
        elif x == "pow5":
            return f"(({arg})^5)"
        elif x == "pow6":
            return f"(({arg})^6)"
        elif x == "inv":
            return f"(1/({arg}))"
        else:
            return x + "(" + arg + ")"

    def infix(self, exp):
        """Returns an infix string representation giving a prefix token list."""
        stack = []
        for x in reversed(exp):
            if x not in self.OPERATORS:
                stack.append(x)
            elif self.OPERATORS[x].arity == 1:
                arg = stack.pop()
                stack.append(self.rewrite(arg, x))
            else:
                left = stack.pop()
                right = stack.pop()
                stack.append("(" + left + " " + x + " " + right + ")")
        assert len(stack) == 1
        return stack[0]


@timeout(5)
def convert(infix):
    return expand(sympy.simplify(str(infix))).evalf(8)


def is_linear(expr, vars):
    for x in vars:
        for y in vars:
            try:
                if not sympy.Eq(sympy.diff(expr, x, y), 0):
                    return False
            except TypeError:
                return False
    return True


def generate_random_expression(
    rng: np.random.Generator,
    n_points: int,
    sampled_points_range: Tuple[float, float],
    max_num_ops: int,
    variables=None,
):
    if variables is None:
        variables = ["x"]
    gen = ExpressionGenerator(max_num_ops, rng, variables)
    tokenizer = GeneralTermTokenizer()

    def sample_points(func):
        num_variables = len(variables)
        for left, right in generate_all_possible_ranges(
            num_variables, sampled_points_range[0], sampled_points_range[1]
        ):
            for _ in range(4):
                try:
                    x = gen.rng.uniform(left, right, (n_points, num_variables))
                    y = evaluate_points(func, x)

                    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                        raise RuntimeWarning("Is nan or inf!")

                    if np.any(np.abs(y) > 1e7):
                        raise RuntimeWarning("Too large or too small")

                    if np.any(np.iscomplex(y)):
                        raise RuntimeWarning("Complex")

                    if not np.all(np.isfinite(y)):
                        raise RuntimeWarning("Not finite")

                    res = np.concatenate((x, np.reshape(y, (-1, 1))), axis=1)
                    return res
                except RuntimeWarning:
                    continue

        raise RuntimeError("No range found")

    warnings.filterwarnings("error")

    def generate():
        while True:
            try:
                num_operations = gen.rng.integers(1, max_num_ops + 1)
                max_integer = 5
                prefix = gen.generate_expr(num_operations, max_integer)
                res = gen.infix(prefix)
                sympy_res = convert(res)

                if num_operations != 1 and is_linear(sympy_res, variables):
                    raise RuntimeWarning("Found linear function")

                if sympy_res.has(sympy.I):
                    raise RuntimeWarning("Contains Imaginary number")

                func = expr_to_func(sympy_res, variables)
                x_y = sample_points(func)
                if np.isclose(np.min(x_y[:, -1]), np.max(x_y[:, -1])):
                    raise RuntimeWarning("Found almost constant function.")

                tree = Tree(convert_to_binary_tree(sympy_res, tokenizer))
                if len(tree.symbolic_pre_order) >= 50:
                    raise RuntimeWarning("Too long!")

                abs_values = np.abs(tree.value_pre_order)
                if np.any(abs_values > 1e10):
                    raise RuntimeWarning("Coefficient is too large")

                if np.any(abs_values[abs_values != 0] < 1e-10):
                    raise RuntimeWarning("Coefficient is too small.")

                return {
                    "points": x_y,
                    "constants": np.array(
                        tree.value_pre_order + [0.0], dtype=np.float32
                    ),
                    "symbolic_expression": " ".join(tree.symbolic_pre_order),
                }

            except (
                AttributeError,
                KeyError,
                RuntimeWarning,
                NameError,
                TypeError,
                RuntimeError,
                TimeoutError,
                OverflowError,
                ZeroDivisionError,
                MemoryError,
                ValueError,
                SystemError,
            ):
                pass

    return generate


if __name__ == "__main__":
    generator = generate_random_expression(
        np.random.default_rng(), 500, (-5, 5), 10, ["x"]
    )
    now = time.time()
    for i in range(1000):
        generated = generator()
        # tmp.append(generated['symbolic_expression'])
        success, infix = prefix_to_infix(
            generated["symbolic_expression"].split(),
            generated["constants"][:-1],
            GeneralTermTokenizer(["x", "y"]),
        )

        print(i, infix)
        # print(infix, generated['symbolic_expression'], generated['constants'])
    print(time.time() - now)
