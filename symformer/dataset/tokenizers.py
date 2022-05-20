import tensorflow as tf


class Tokenizer:

    SPECIAL_SYMBOLS = {}

    SPECIAL_FLOAT_SYMBOLS = {}

    SPECIAL_OPERATORS = {}

    SPECIAL_INTEGERS = {}

    def __init__(self):
        self.start = "<START>"
        self.start_id = 1
        self.end = "<END>"
        self.end_id = 2
        self.pad = "<PAD>"
        self.pad_id = 0
        self.vocab = [self.pad, self.start, self.end]

    def encode(self, expr):
        raise NotImplementedError()

    def decode(self, expr):
        raise NotImplementedError()

    def is_unary(self, token):
        raise NotImplementedError()

    def is_binary(self, token):
        raise NotImplementedError()

    def is_leaf(self, token):
        raise NotImplementedError()

    def get_constant_ids(self):
        pass


class GeneralTermTokenizer(Tokenizer):

    SPECIAL_SYMBOLS = {
        "inv": "(({})^-1)",
        "pow2": "(({})^2)",
        "pow3": "(({})^3)",
        "sqrt": "(({})^(1/2))",
        "neg": "(-1*{})",
    }

    SPECIAL_FLOAT_SYMBOLS = {
        0.5: "sqrt",
        -1: "neg",
    }

    SPECIAL_OPERATORS = {
        "2": "pow2",
        "3": "pow3",
        "-1": "inv",
        "1/2": "sqrt",
    }

    SPECIAL_INTEGERS = [str(i) for i in range(-5, 6)]

    def __init__(self, variables=None, extended_repre=True):
        super().__init__()
        self.variables = variables if variables is not None else ["x"]
        self.binary_ops = ["^", "+", "*"]
        self.unary_ops = [
            "sqrt",
            "inv",
            "pow2",
            "pow3",
            "ln",
            "exp",
            "sin",
            "cos",
            "tan",
            "cot",
            "asin",
            "acos",
            "atan",
            "acot",
            "neg",
        ]
        if extended_repre:
            self.constants = [f"C{i}" for i in range(-10, 11)]
        else:
            self.constants = ["C"]
        self.leafs = self.constants + self.variables + self.SPECIAL_INTEGERS
        self.vocab += list(self.binary_ops) + list(self.unary_ops) + self.leafs

        init = tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(self.vocab, dtype=tf.string),
            values=tf.constant(
                tf.range(0, len(self.vocab), dtype=tf.int64), dtype=tf.int64
            ),
        )
        self.lookup_table = tf.lookup.StaticHashTable(init, -1)

        init = tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(
                tf.range(0, len(self.vocab), dtype=tf.int64), dtype=tf.int64
            ),
            values=tf.constant(self.vocab, dtype=tf.string),
            key_dtype=tf.int64,
            value_dtype=tf.string,
        )
        self.reverse_lookup_table = tf.lookup.StaticHashTable(init, "<UNK>")
        self.constant_ids = tf.cast(
            tf.reshape(self.encode(self.constants).to_tensor(), [-1]), tf.int32
        )

    def encode(self, expr):
        return self.lookup_table[tf.strings.split(expr, " ")]

    def decode(self, ids):
        return self.reverse_lookup_table[ids]

    def is_unary(self, token):
        return token in self.unary_ops

    def is_binary(self, token):
        return token in self.binary_ops

    def is_leaf(self, token):
        return token in self.leafs + ["C"]

    def get_constant_ids(self):
        return self.constant_ids
