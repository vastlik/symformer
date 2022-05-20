from __future__ import annotations

import numpy as np
import sympy as sp
import sympy.core.numbers
from sympy import default_sort_key
from sympy.core import Expr

from ..tokenizers import Tokenizer

SYMBOL_MAP = ["x", "y", "z"] + [chr(x) for x in range(ord("a"), ord("x"))]

SYMPY_OPERATORS = {
    # Elementary functions
    sp.core.add.Add: "+",
    sp.core.mul.Mul: "*",
    sp.Pow: "^",
    sp.exp: "exp",
    sp.log: "ln",
    sp.Abs: "abs",
    # Trigonometric Functions
    sp.sin: "sin",
    sp.cos: "cos",
    sp.tan: "tan",
    sp.cot: "cot",
    # Trigonometric Inverses
    sp.asin: "asin",
    sp.acos: "acos",
    sp.atan: "atan",
    sp.acot: "acot",
    # Hyperbolic trigonometric Inverses
    sp.sinh: "sinh",
    sp.cosh: "cosh",
    sp.tanh: "tanh",
    sp.coth: "coth",
}


class Node:
    left: Node | None = None
    right: Node | None = None

    def __init__(self, symbol: str, coefficient: float, arity: int):
        self.symbol = symbol
        self.value = coefficient
        self.arity = arity


def expression_to_symbol(expr, tokenizer: Tokenizer):
    if expr in tokenizer.SPECIAL_SYMBOLS:
        return expr

    if (
        expr.is_Number
        and float(expr).is_integer()
        and str(int(expr)) in tokenizer.SPECIAL_INTEGERS
    ):
        return str(int(expr))

    if expr.is_Symbol:
        if len(expr.name) == 1:
            return expr.name
        return SYMBOL_MAP[int(expr.name[1:])]

    for sympy_class, symbol in SYMPY_OPERATORS.items():
        if isinstance(expr, sympy_class):
            return symbol

    return "C"


def expr_to_constant(value, tokenizer: Tokenizer):
    if isinstance(value, str):
        return 0

    if (
        value.is_Number
        and float(value).is_integer()
        and str(int(value)) in tokenizer.SPECIAL_INTEGERS
    ):
        return 0

    if isinstance(value, sympy.core.numbers.Float) or value.is_Number:
        return float(value)

    return 0


class Tree:
    def __init__(self, root: Node):
        self.root = root
        self.symbolic_pre_order: list[str] = []
        self.value_pre_order: list[float] = []
        self.do_preorder(root)

    def do_preorder(self, node: Node):
        if node is not None:
            self.symbolic_pre_order.append(node.symbol)
            self.value_pre_order.append(node.value)
            self.do_preorder(node.left)
            self.do_preorder(node.right)


def convert_to_binary_tree(expr: Expr, tokenizer: Tokenizer):
    """
    Currently the divison a/b is handled as a^-1 * b
    :param expr:
    :return:
    """

    symbol = expression_to_symbol(expr, tokenizer)
    constant = expr_to_constant(expr, tokenizer)
    if len(expr.args) > 2:
        node = Node(symbol, constant, 2)
        first_node = node
        args = sorted(expr.args, key=default_sort_key)
        for arg in args[:-2]:
            node.left = convert_to_binary_tree(arg, tokenizer)
            node.right = Node(symbol, constant, 2)
            node = node.right

        node.left = convert_to_binary_tree(args[-2], tokenizer)
        node.right = convert_to_binary_tree(args[-1], tokenizer)

        return first_node
    elif len(expr.args) == 2:
        args = expr.args
        if expr.is_Pow and str(args[1]) in tokenizer.SPECIAL_OPERATORS:
            converted_operator = tokenizer.SPECIAL_OPERATORS[str(args[1])]
            symbol = expression_to_symbol(converted_operator, tokenizer)
            constant = expr_to_constant(converted_operator, tokenizer)

            node = Node(symbol, constant, 1)
            node.left = convert_to_binary_tree(args[0], tokenizer)
            return node

        if (
            expr.is_Pow
            and args[1].is_Float
            and float(args[1]) in tokenizer.SPECIAL_FLOAT_SYMBOLS
        ):
            converted_operator = tokenizer.SPECIAL_FLOAT_SYMBOLS[float(args[1])]
            symbol = expression_to_symbol(converted_operator, tokenizer)
            constant = expr_to_constant(converted_operator, tokenizer)

            node = Node(symbol, constant, 1)
            node.left = convert_to_binary_tree(args[0], tokenizer)
            return node

        if (
            expr.is_Mul
            and isinstance(args[0], sympy.core.numbers.NegativeOne)
            and float(args[0]) in tokenizer.SPECIAL_FLOAT_SYMBOLS
        ):
            converted_operator = tokenizer.SPECIAL_FLOAT_SYMBOLS[float(args[0])]
            symbol = expression_to_symbol(converted_operator, tokenizer)
            constant = expr_to_constant(converted_operator, tokenizer)

            node = Node(symbol, constant, 1)
            node.left = convert_to_binary_tree(args[1], tokenizer)
            return node

        node = Node(symbol, constant, 2)
        node.left = convert_to_binary_tree(args[0], tokenizer)
        node.right = convert_to_binary_tree(args[1], tokenizer)

        return node
    elif len(expr.args) == 1:
        node = Node(symbol, constant, 1)
        node.left = convert_to_binary_tree(expr.args[0], tokenizer)
        return node
    else:
        node = Node(symbol, constant, 0)
        return node


def prefix_to_infix(
    expr: list[str] | np.ndarray, constants: list[float] | None, tokenizer: Tokenizer
):
    stack = []
    for i, symbol in reversed(list(enumerate(expr))):
        if tokenizer.is_binary(symbol):
            if len(stack) < 2:
                return False, None
            tmp_str = "(" + stack.pop() + symbol + stack.pop() + ")"
            stack.append(tmp_str)
        elif tokenizer.is_unary(symbol) or symbol == "abs":
            if len(stack) < 1:
                return False, None
            if symbol in tokenizer.SPECIAL_SYMBOLS:
                stack.append(tokenizer.SPECIAL_SYMBOLS[symbol].format(stack.pop()))
            else:
                stack.append(symbol + "(" + stack.pop() + ")")
        elif tokenizer.is_leaf(symbol):
            if symbol == "C":
                stack.append(str(constants[i]))
            elif "C" in symbol:
                exponent = int(symbol[1:])
                stack.append(str(constants[i] * 10 ** exponent))
            else:
                stack.append(symbol)

    if len(stack) != 1:
        return False, None

    return True, stack.pop()


def print_tree(node: Node):
    if node is not None:
        print(node.value)
        print_tree(node.left)
        print_tree(node.right)
