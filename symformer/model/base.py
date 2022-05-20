import tensorflow as tf

from .config import Config
from .metrics import accuracy_function, hard_accuracy


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_mask(tar):
    return create_look_ahead_mask(tf.shape(tar)[1])


def get_loss(loss_name: str):
    loss_name = loss_name.lower()
    if loss_name == "mse":
        return tf.keras.losses.MeanSquaredError(reduction="none")
    elif loss_name == "mae":
        return tf.keras.losses.MeanAbsoluteError(reduction="none")
    elif loss_name == "huber":
        return tf.keras.losses.Huber(reduction="none")

    raise ValueError("No loss found")


def pad_with_zeroes(tensor, max_len):
    return tf.pad(tensor, [[0, 0], [0, max_len - tf.shape(tensor)[1]]])


def pad_with_one_hot(tensor, max_len, vocab_size):
    pad_logit = tf.one_hot(
        tf.zeros((tf.shape(tensor)[0], max_len - tf.shape(tensor)[1]), dtype=tf.int32),
        vocab_size,
    )
    return tf.concat([tensor, pad_logit], axis=1)


def sparse_categorical_cross_entropy(label_smoothing: float, n_classes: int):
    cross_entropy = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction="none", label_smoothing=label_smoothing
    )

    def loss(y_true, y_pred):
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), n_classes)
        return cross_entropy(y_true, y_pred)

    return loss


class TransformerBase(tf.keras.Model):
    def __init__(self, cfg: Config, tokenizer, strategy: tf.distribute.Strategy):
        super().__init__()
        self.cfg = cfg
        self.cross_entropy = sparse_categorical_cross_entropy(
            cfg.label_smoothing, len(tokenizer.vocab)
        )
        self.strategy = strategy
        self.min_regression_loss = get_loss(cfg.loss)
        self.mae = tf.keras.losses.MeanAbsoluteError(reduction="none")
        self.mse = tf.keras.losses.MeanSquaredError(reduction="none")

        # vážit dle exponentu
        self.tokenizer = tokenizer
        self.num_replicas = tf.distribute.get_strategy().num_replicas_in_sync

        self.regression_loss_lambda = tf.Variable(
            cfg.regression_lambda, trainable=False, dtype=tf.float32
        )
        self.h_metrics = {
            "loss": tf.keras.metrics.Mean(name="train_loss"),
            "classification_loss": tf.keras.metrics.Mean(name="classification_loss"),
            "accuracy": tf.keras.metrics.Mean(name="train_accuracy"),
            "hard_accuracy": tf.keras.metrics.Mean("hard_accuracy"),
            "mse_loss": tf.keras.metrics.Mean(name="mse_loss"),
            "mse_loss_hard": tf.keras.metrics.Mean("mse_loss_hard"),
            "mae_loss": tf.keras.metrics.Mean(name="mae_loss"),
            "mae_loss_hard": tf.keras.metrics.Mean("mae_loss_hard"),
            "regression_loss": tf.keras.metrics.Mean("regression_loss"),
            "regression_loss_hard": tf.keras.metrics.Mean("regression_loss_hard"),
        }

    def classification_loss(self, prediction, golden):
        mask = tf.math.logical_not(tf.math.equal(golden, 0))
        mask_ = tf.reshape(mask, [-1])
        loss_ = self.cross_entropy(golden, prediction)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reshape(loss_, [-1])[mask_]

    def regression_loss(self, golden, prediction, loss, output_mask=None):
        # todo, we expect that golden has 0, where we do not want to calculate loss
        mask = tf.math.logical_not(tf.math.equal(golden, 0.0))
        mask_ = tf.reshape(mask, [-1])
        golden = tf.expand_dims(golden, -1)
        loss_ = loss(golden, prediction)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        if output_mask is not None:
            output_mask = tf.broadcast_to(
                tf.expand_dims(output_mask, -1), tf.shape(mask)
            )
            loss_ *= tf.cast(output_mask, dtype=loss_.dtype)

        return tf.reshape(loss_, [-1])[mask_]
        # return tf.math.divide_no_nan(loss_sum, sum_mask)

    def normalize_loss(self, loss):
        return tf.reduce_mean(loss)

    def update_metrics(
        self,
        classification_loss,
        loss,
        symbolic_tar,
        predictions,
        regression_loss,
        regression_target,
        regression_predictions,
    ):
        self.h_metrics["classification_loss"](self.normalize_loss(classification_loss))
        self.h_metrics["loss"](loss)
        self.h_metrics["accuracy"](accuracy_function(symbolic_tar, predictions))
        self.h_metrics["regression_loss"](self.normalize_loss(regression_loss))
        hard_acc, logical_map = hard_accuracy(symbolic_tar, predictions, self.tokenizer)
        self.h_metrics["hard_accuracy"](hard_acc)
        self.h_metrics["regression_loss_hard"](
            self.normalize_loss(
                self.regression_loss(
                    regression_target,
                    regression_predictions,
                    self.min_regression_loss,
                    logical_map,
                )
            )
        )
        self.h_metrics["mse_loss"](
            self.normalize_loss(
                self.regression_loss(
                    regression_target, regression_predictions, self.mse
                )
            )
        )
        self.h_metrics["mse_loss_hard"](
            self.normalize_loss(
                self.regression_loss(
                    regression_target, regression_predictions, self.mse, logical_map
                )
            )
        )
        self.h_metrics["mae_loss"](
            self.normalize_loss(
                self.regression_loss(
                    regression_target, regression_predictions, self.mae
                )
            )
        )
        self.h_metrics["mae_loss_hard"](
            self.normalize_loss(
                self.regression_loss(
                    regression_target, regression_predictions, self.mae, logical_map
                )
            )
        )

    @property
    def metrics(self):
        return list(self.h_metrics.values())


class TransformerNoRegBase(tf.keras.Model):
    def __init__(self, cfg: Config, tokenizer, strategy: tf.distribute.Strategy):
        super().__init__()
        self.cfg = cfg
        self.cross_entropy = sparse_categorical_cross_entropy(
            cfg.label_smoothing, len(tokenizer.vocab)
        )
        self.strategy = strategy

        self.tokenizer = tokenizer
        self.num_replicas = tf.distribute.get_strategy().num_replicas_in_sync

        self.regression_loss_lambda = tf.Variable(
            cfg.regression_lambda, trainable=False, dtype=tf.float32
        )
        self.h_metrics = {
            "loss": tf.keras.metrics.Mean(name="train_loss"),
            "accuracy": tf.keras.metrics.Mean(name="train_accuracy"),
            "hard_accuracy": tf.keras.metrics.Mean("hard_accuracy"),
        }

    def classification_loss(self, prediction, golden):
        mask = tf.math.logical_not(tf.math.equal(golden, 0))
        mask_ = tf.reshape(mask, [-1])
        loss_ = self.cross_entropy(golden, prediction)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reshape(loss_, [-1])[mask_]

    def normalize_loss(self, loss):
        return tf.reduce_mean(loss)

    def update_metrics(self, loss, symbolic_tar, predictions):
        self.h_metrics["loss"](loss)
        self.h_metrics["accuracy"](accuracy_function(symbolic_tar, predictions))
        hard_acc, logical_map = hard_accuracy(symbolic_tar, predictions, self.tokenizer)
        self.h_metrics["hard_accuracy"](hard_acc)

    @property
    def metrics(self):
        return list(self.h_metrics.values())
