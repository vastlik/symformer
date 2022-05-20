import tensorflow as tf

from .attention import MHA
from .config import DecoderConfig, DecoderWithRegressionConfig


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential(
        [
            # nafouknout 4x d_model, a gelu
            tf.keras.layers.Dense(dff, activation="relu"),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model),  # (batch_size, seq_len, d_model)
        ]
    )


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()

        self.mha1 = MHA(num_heads, d_model)
        self.mha2 = MHA(num_heads, d_model)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1(
            x, x, x, look_ahead_mask
        )  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, None
        )  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(
            ffn_output + out2
        )  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Decoder(tf.keras.layers.Layer):
    def __init__(self, cfg: DecoderConfig, vocab_size):
        super().__init__()

        self.d_model = cfg.d_model
        self.num_layers = cfg.num_layers

        self.embedding = tf.keras.layers.Embedding(vocab_size, cfg.d_model)
        self.pos_encoding = tf.keras.layers.Embedding(cfg.max_len, cfg.d_model)

        self.dec_layers = [
            DecoderLayer(cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.dropout_rate)
            for _ in range(cfg.num_layers)
        ]  # jako GPT2, nafouknout 4x, FF, gelu, FF, residual
        self.dropout = tf.keras.layers.Dropout(cfg.dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding(tf.range(0, seq_len))

        x = self.dropout(x, training=training)
        for i, layer in enumerate(self.dec_layers):
            x, block1, block2 = layer(x, enc_output, training, look_ahead_mask)
            attention_weights[f"decoder_layer{i + 1}_block1"] = block1
            attention_weights[f"decoder_layer{i + 1}_block2"] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class DecoderWithRegressionAsSeq(tf.keras.layers.Layer):
    def __init__(self, cfg: DecoderWithRegressionConfig, vocab_len):
        super().__init__()

        if cfg.mode == "concat":
            d_model = cfg.d_model // 2
        elif cfg.mode == "sum":
            d_model = cfg.d_model
        else:
            raise ValueError("Unknown decoder mode.")

        self.mode = cfg.mode
        self.d_model = d_model
        self.num_layers = cfg.num_layers

        self.embedding = tf.keras.layers.Embedding(vocab_len, d_model)
        self.pos_encoding = tf.keras.layers.Embedding(cfg.max_len, d_model)
        self.reg_pos_encoding = tf.keras.layers.Embedding(cfg.max_len, d_model)

        self.reg_embedding = tf.keras.Sequential([tf.keras.layers.Dense(d_model)])
        self.dec_layers = [
            DecoderLayer(cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.dropout_rate)
            for _ in range(cfg.num_layers)
        ]  # jako GPT2, nafouknout 4x, FF, gelu, FF, residual
        self.dropout = tf.keras.layers.Dropout(cfg.dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, reg_tar):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)

        pos_encoding = self.pos_encoding(tf.range(0, seq_len))
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += pos_encoding

        reg_embedding = self.reg_embedding(reg_tar)
        # reg_embedding *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # reg_embedding += self.reg_pos_encoding(tf.range(0, seq_len)) # pos_encoding

        if self.mode == "concat":
            reg_embedding *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            reg_embedding += pos_encoding
            x = tf.concat([x, reg_embedding], axis=-1)
        elif self.mode == "sum":
            x += reg_embedding
        else:
            raise ValueError(
                "You forgot to specify how the regression embedding should be handled."
            )

        x = self.dropout(x, training=training)
        for i, layer in enumerate(self.dec_layers):
            x, block1, block2 = layer(x, enc_output, training, look_ahead_mask)
            attention_weights[f"decoder_layer{i + 1}_block1"] = block1
            attention_weights[f"decoder_layer{i + 1}_block2"] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights
