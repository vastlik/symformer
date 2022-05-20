import enum

import tensorflow as tf

from .base import (
    create_mask,
    pad_with_one_hot,
    pad_with_zeroes,
    TransformerBase,
    TransformerNoRegBase,
)
from .config import Config
from .decoder_layers import Decoder, DecoderWithRegressionAsSeq
from .encoder_layers import Encoder


def get_regression_head(num_layers, dim, extended_representation):
    layers = [tf.keras.layers.Dense(dim, activation="relu") for _ in range(num_layers)]
    if extended_representation:
        layers.append(tf.keras.layers.Dense(1, activation="tanh"))
    else:
        layers.append(tf.keras.layers.Dense(1))
    return tf.keras.Sequential(layers)


class TransformerType(enum.Enum):
    CLASSIC = "classic"
    REG_AS_SEQ = "reg_as_seq"
    NO_REG = "no_reg"


class Transformer(TransformerBase):
    def __init__(self, cfg: Config, tokenizer, strategy):
        super().__init__(cfg, tokenizer, strategy)

        self.encoder = Encoder(self.cfg.encoder_config)
        self.decoder = Decoder(self.cfg.decoder_config, len(tokenizer.vocab))
        self.final_layer = tf.keras.layers.Dense(len(tokenizer.vocab))
        self.regression_layer = get_regression_head(
            self.cfg.reg_head_num_layers,
            self.cfg.reg_head_dim,
            cfg.dataset_config.extended_representation,
        )

    def call(self, inputs, training):
        inp, tar = inputs

        look_ahead_mask = create_mask(tar)

        enc_output = self.encoder(inp, training)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask
        )

        final_output = self.final_layer(
            dec_output
        )  # (batch_size, tar_seq_len, target_vocab_size)
        reg_head_output = self.regression_layer(dec_output)

        return final_output, reg_head_output, attention_weights

    def call_without_encoder(self, tar, enc_output):
        look_ahead_mask = create_mask(tar)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, False, look_ahead_mask
        )

        final_output = self.final_layer(
            dec_output
        )  # (batch_size, tar_seq_len, target_vocab_size)
        reg_head_output = self.regression_layer(dec_output)

        return final_output, reg_head_output, attention_weights

    def train_step(self, data):
        points = data["points"]
        symbolic_input = data["symbolic_expr_input"]
        symbolic_tar = data["symbolic_expr_target"]
        regression_target = data["constants_target"]
        with tf.GradientTape() as tape:
            predictions, regression_predictions, _ = self(
                (points, symbolic_input), training=True
            )

            classification_loss = self.classification_loss(predictions, symbolic_tar)
            regression_loss = self.regression_loss(
                regression_target, regression_predictions, self.min_regression_loss
            )
            loss = classification_loss + self.regression_loss_lambda * regression_loss

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.update_metrics(
            classification_loss,
            loss,
            symbolic_tar,
            predictions,
            regression_loss,
            regression_target,
            regression_predictions,
        )
        return {key: value.result() for key, value in self.h_metrics.items()}

    @tf.function
    def test_step(self, data):
        points = data["points"]
        symbolic_input = data["symbolic_expr_input"]
        symbolic_tar = data["symbolic_expr_target"]
        regression_target = data["constants_target"]
        predictions, regression_predictions, _ = self(
            (points, symbolic_input), training=False
        )

        classification_loss = self.classification_loss(predictions, symbolic_tar)
        regression_loss = self.regression_loss(
            regression_target, regression_predictions, self.min_regression_loss
        )
        loss = classification_loss + self.regression_loss_lambda * regression_loss
        self.update_metrics(
            classification_loss,
            loss,
            symbolic_tar,
            predictions,
            regression_loss,
            regression_target,
            regression_predictions,
        )
        return {key: value.result() for key, value in self.h_metrics.items()}

    @tf.function
    def test_step_without_teacher_forcing(self, data):
        points = data["points"]
        symbolic_tar = data["symbolic_expr_target"]
        regression_target = data["constants_target"]
        predictions, regression_predictions, logits = self.predict_without_teacher(
            points
        )
        max_len = tf.maximum(tf.shape(predictions)[1], tf.shape(symbolic_tar)[1])

        symbolic_tar = pad_with_zeroes(symbolic_tar, max_len)
        regression_target = pad_with_zeroes(regression_target, max_len)
        regression_predictions = pad_with_zeroes(regression_predictions, max_len)
        logits = pad_with_one_hot(logits, max_len, len(self.tokenizer.vocab))
        classification_loss = self.classification_loss(logits, symbolic_tar)
        regression_predictions = tf.expand_dims(regression_predictions, axis=-1)
        regression_loss = self.regression_loss(
            regression_target, regression_predictions, self.min_regression_loss
        )
        loss = classification_loss + self.regression_loss_lambda * regression_loss

        self.update_metrics(
            classification_loss,
            loss,
            symbolic_tar,
            logits,
            regression_loss,
            regression_target,
            regression_predictions,
        )
        return predictions, regression_predictions


class TransformerWithRegressionAsSeq(TransformerBase):
    def __init__(self, cfg, tokenizer, input_regularizer, strategy):
        super().__init__(cfg, tokenizer, strategy)

        self.encoder = Encoder(self.cfg.encoder_config)
        self.decoder = DecoderWithRegressionAsSeq(
            self.cfg.decoder_config, len(self.tokenizer.vocab)
        )
        self.final_layer = tf.keras.layers.Dense(len(self.tokenizer.vocab))
        self.regression_layer = get_regression_head(
            self.cfg.reg_head_num_layers,
            self.cfg.reg_head_dim,
            cfg.dataset_config.extended_representation,
        )
        self.input_regularizer = input_regularizer

    @tf.function
    def call(self, inputs, training):
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar, reg_tar = inputs
        reg_tar = tf.expand_dims(reg_tar, -1)
        look_ahead_mask = create_mask(tar)
        enc_output = self.encoder(inp, training)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, reg_tar
        )

        final_output = self.final_layer(
            dec_output
        )  # (batch_size, tar_seq_len, target_vocab_size)
        reg_head_output = self.regression_layer(dec_output)

        return final_output, reg_head_output, attention_weights

    @tf.function(experimental_relax_shapes=True)
    def call_without_encoder(self, tar, reg_tar, enc_output):
        reg_tar = tf.expand_dims(reg_tar, -1)
        look_ahead_mask = create_mask(tar)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, False, look_ahead_mask, reg_tar
        )

        final_output = self.final_layer(
            dec_output
        )  # (batch_size, tar_seq_len, target_vocab_size)
        reg_head_output = self.regression_layer(dec_output)

        return final_output, reg_head_output, attention_weights

    def train_step(self, data):
        points = data["points"]
        symbolic_input = data["symbolic_expr_input"]
        symbolic_tar = data["symbolic_expr_target"]
        regression_target = data["constants_target"]
        regression_input = self.input_regularizer.transform(data["constants_input"])
        regression_input = tf.clip_by_value(regression_input, -1, 1)

        with tf.GradientTape() as tape:
            predictions, regression_predictions, _ = self(
                (points, symbolic_input, regression_input), training=True
            )

            classification_loss = self.classification_loss(predictions, symbolic_tar)
            regression_loss = self.regression_loss(
                regression_target, regression_predictions, self.min_regression_loss
            )
            loss = tf.nn.compute_average_loss(
                classification_loss
            ) + tf.nn.compute_average_loss(
                self.regression_loss_lambda * regression_loss
            )

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.update_metrics(
            classification_loss * 1 / self.num_replicas,
            loss,
            symbolic_tar,
            predictions,
            regression_loss * 1 / self.num_replicas,
            regression_target,
            regression_predictions,
        )
        return {key: value.result() for key, value in self.h_metrics.items()}

    def test_step(self, data):
        points = data["points"]
        symbolic_input = data["symbolic_expr_input"]
        symbolic_tar = data["symbolic_expr_target"]
        regression_input = data["constants_input"]
        regression_target = data["constants_target"]
        predictions, regression_predictions, _ = self(
            (points, symbolic_input, regression_input), training=False
        )

        classification_loss = self.classification_loss(predictions, symbolic_tar)
        regression_loss = self.regression_loss(
            regression_target, regression_predictions, self.min_regression_loss
        )
        loss = tf.nn.compute_average_loss(
            classification_loss
        ) + tf.nn.compute_average_loss(self.regression_loss_lambda * regression_loss)
        self.update_metrics(
            classification_loss,
            loss,
            symbolic_tar,
            predictions,
            regression_loss,
            regression_target,
            regression_predictions,
        )
        return {key: value.result() for key, value in self.h_metrics.items()}

    @tf.function
    def update_metrics_from_search(
        self,
        predictions,
        regression_predictions,
        logits,
        symbolic_tar,
        regression_target,
    ):
        max_len = tf.maximum(tf.shape(predictions)[1], tf.shape(symbolic_tar)[1])

        max_len_reg = tf.maximum(
            tf.shape(regression_predictions)[1], tf.shape(regression_target)[1]
        )

        symbolic_tar = pad_with_zeroes(symbolic_tar, max_len)
        regression_target = pad_with_zeroes(regression_target, max_len_reg)
        regression_predictions = pad_with_zeroes(regression_predictions, max_len_reg)
        logits = pad_with_one_hot(logits, max_len, len(self.tokenizer.vocab))
        classification_loss = self.classification_loss(logits, symbolic_tar)
        regression_predictions = tf.expand_dims(regression_predictions, axis=-1)
        regression_loss = self.regression_loss(
            regression_target, regression_predictions, self.min_regression_loss
        )
        loss = tf.reduce_mean(classification_loss) + tf.reduce_mean(
            self.regression_loss_lambda * regression_loss
        )
        self.update_metrics(
            classification_loss,
            loss,
            symbolic_tar,
            logits,
            regression_loss,
            regression_target,
            regression_predictions,
        )


class TransformerNoRegression(TransformerNoRegBase):
    def __init__(self, cfg, tokenizer, strategy):
        super().__init__(cfg, tokenizer, strategy)

        self.encoder = Encoder(self.cfg.encoder_config)
        self.decoder = Decoder(self.cfg.decoder_config, len(self.tokenizer.vocab))
        self.final_layer = tf.keras.layers.Dense(len(self.tokenizer.vocab))

    @tf.function
    def call(self, inputs, training):
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar = inputs
        look_ahead_mask = create_mask(tar)
        enc_output = self.encoder(inp, training)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask
        )

        final_output = self.final_layer(
            dec_output
        )  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

    @tf.function(experimental_relax_shapes=True)
    def call_without_encoder(self, tar, enc_output):
        look_ahead_mask = create_mask(tar)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, False, look_ahead_mask
        )

        final_output = self.final_layer(
            dec_output
        )  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

    def train_step(self, data):
        points = data["points"]
        symbolic_input = data["symbolic_expr_input"]
        symbolic_tar = data["symbolic_expr_target"]

        with tf.GradientTape() as tape:
            predictions, _ = self((points, symbolic_input), training=True)

            classification_loss = self.classification_loss(predictions, symbolic_tar)
            loss = tf.nn.compute_average_loss(classification_loss)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.update_metrics(loss, symbolic_tar, predictions)
        return {key: value.result() for key, value in self.h_metrics.items()}

    def test_step(self, data):
        points = data["points"]
        symbolic_input = data["symbolic_expr_input"]
        symbolic_tar = data["symbolic_expr_target"]
        predictions, _ = self((points, symbolic_input), training=False)

        classification_loss = self.classification_loss(predictions, symbolic_tar)
        loss = tf.nn.compute_average_loss(classification_loss)
        self.update_metrics(loss, symbolic_tar, predictions)
        return {key: value.result() for key, value in self.h_metrics.items()}

    @tf.function
    def update_metrics_from_search(self, predictions, logits, symbolic_tar):
        max_len = tf.maximum(tf.shape(predictions)[1], tf.shape(symbolic_tar)[1])

        symbolic_tar = pad_with_zeroes(symbolic_tar, max_len)
        logits = pad_with_one_hot(logits, max_len, len(self.tokenizer.vocab))
        classification_loss = self.classification_loss(logits, symbolic_tar)
        self.update_metrics(classification_loss, symbolic_tar, logits)
