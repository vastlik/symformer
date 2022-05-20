import tensorflow as tf
from aclick.utils import default_from_str


@default_from_str
class Regularizer:
    def transform(self, tensor):
        raise NotImplementedError()

    def update_params(self, params):
        raise NotImplementedError()

    def config(self):
        raise NotImplementedError()

    def wandb_log(self):
        raise NotImplementedError()


class RandomNoiseRegularizer(Regularizer):
    def __init__(self, mean=0.0, stddev=1.0):
        self.mean = tf.Variable(mean, trainable=False)
        self.stddev = tf.Variable(stddev, trainable=False)

    def transform(self, tensor):
        return tensor + tf.random.normal(
            tf.shape(tensor), self.mean, self.stddev
        ) * tf.cast(tensor != 0, tf.float32)

    def update_params(self, stddev):
        tf.keras.backend.set_value(self.stddev, stddev)

    def config(self):
        return {
            "name": "random_noise",
            "mean": self.mean,
            "stddev": self.stddev,
        }

    def wandb_log(self):
        return "stddev", self.stddev

    def __str__(self):
        mean, stddev = self.mean.value(), self.stddev.value()
        return f"random_noise_regularizer(mean={mean}, stddev={stddev})"


class DropoutRegularizer(Regularizer):
    def __init__(self, dropout_rate):
        self.dropout_rate = tf.Variable(dropout_rate, trainable=False)

    def transform(self, tensor):
        return tensor * tf.cast(
            tf.random.uniform(tf.shape(tensor)) < self.dropout_rate, tf.float32
        )

    def update_params(self, prob):
        tf.keras.backend.set_value(self.dropout_rate, prob)

    def config(self):
        return {"name": "dropout", "dropout_rate": self.dropout_rate}

    def wandb_log(self):
        return "dropout_rate", self.dropout_rate

    def __str__(self):
        dropout_rate = self.dropout_rate.value()
        return f"dropout_regularizer(dropout_rate={dropout_rate})"


class NoOpRegularizer(Regularizer):
    def update_params(self, prob):
        return

    def transform(self, tensor):
        return tensor

    def config(self):
        return {"name": "no_op"}

    def wandb_log(self):
        return None, None
