import tensorflow as tf

from .config import EncoderConfig


# todo We should also use LayerNorm
def create_embedding_layer(d_model):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(d_model),
        ]
    )


class RowWiseFF(tf.keras.layers.Layer):
    def __init__(self, d_model: int, num_layers: int):
        super().__init__()
        self.ffs = [
            tf.keras.layers.Dense(d_model * 4, activation="relu"),
            tf.keras.layers.Dense(d_model),
        ]

    def call(self, inputs):
        out = inputs
        for ff in self.ffs:
            out = ff(out)

        return out


class MAB(tf.keras.layers.Layer):
    def __init__(self, num_heads: int, d_model: int, rwff_num_layers: int):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads, int(d_model // num_heads)
        )
        self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads, d_model)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()
        self.row_wise_ff = RowWiseFF(d_model, rwff_num_layers)

    def call(self, inputs):
        X, Y = inputs  # X shape (B, T, dim), Y shape (B, S, dim)
        H = self.layer_norm_1(X + self.att(X, Y, Y))  # (B, T, dim)
        return self.layer_norm_2(H + self.row_wise_ff(H))  # (B, T, dim)


class ISAB(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        m_inducing_points: int,
        row_wise_ff_num_layers: int,
    ):
        super().__init__()
        self.mab = MAB(num_heads, d_model, row_wise_ff_num_layers)
        self.mab_2 = MAB(num_heads, d_model, row_wise_ff_num_layers)
        self.inducing_points = self.add_weight(
            "inducing_points",
            shape=(1, m_inducing_points, d_model),
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
        )

    def call(self, inputs):
        # nahradit self attention
        inducing_points = tf.repeat(self.inducing_points, (tf.shape(inputs)[0]), axis=0)
        H = self.mab((inducing_points, inputs))
        return self.mab_2((inputs, H))


class PMA(tf.keras.layers.Layer):
    def __init__(
        self, num_heads: int, d_model: int, k_seed_vectors: int, row_wise_ff_num_layers
    ):
        super().__init__()
        self.rff = RowWiseFF(d_model, row_wise_ff_num_layers)
        self.mab = MAB(num_heads, d_model, row_wise_ff_num_layers)
        self.seed_vector = self.add_weight(
            "seed_vector",
            shape=(1, k_seed_vectors, d_model),
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
        )

    def call(self, inputs):
        b = tf.shape(inputs)[0]
        seed_vector = tf.repeat(self.seed_vector, [b], axis=0)
        return self.mab((seed_vector, self.rff(inputs)))


class Encoder(tf.keras.layers.Layer):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.isabs = []
        self.embedding = create_embedding_layer(cfg.d_model)
        self.dropout = None
        if cfg.dropout_rate is not None:
            self.dropout = tf.keras.layers.Dropout(cfg.dropout_rate)
        for _ in range(cfg.num_layers):
            self.isabs.append(
                ISAB(
                    cfg.num_heads,
                    cfg.d_model,
                    cfg.m_inducing_points,
                    cfg.row_wise_ff_num_layers,
                )
            )
        self.PMA = PMA(
            cfg.num_heads, cfg.d_model, cfg.k_seed_vectors, cfg.row_wise_ff_num_layers
        )
        # self.SAB = MAB(cfg.num_heads, cfg.d_model, cfg.row_wise_ff_num_layers)

    def call(self, inputs, training):
        out = self.embedding(inputs)
        for isab in self.isabs:
            out = isab(out)
            if self.dropout is not None:
                out = self.dropout(out, training=training)

        out = self.PMA(out)
        return out
        # return self.SAB((out, out))
