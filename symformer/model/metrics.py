import numpy as np
import tensorflow as tf


def accuracy_function(real, pred):
    # issue with this simple accuracy function is that it does not take into the account that we do not care
    # about ordering of the terms in the expression
    accuracies = tf.equal(tf.cast(real, tf.int64), tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


def create_mask(ends, max_size):
    return tf.sequence_mask(ends, max_size)


def is_special_token(expr, tokenizer):
    return tf.math.reduce_any(
        [
            expr == tokenizer.start_id,
            expr == tokenizer.end_id,
            expr == tokenizer.pad_id,
        ],
        axis=0,
    )


def hard_accuracy(real, pred, tokenizer):
    log_map = is_special_token(real, tokenizer)
    ends = tf.argmax(log_map, axis=-1, output_type=tf.int32)
    real_cleared = real * tf.sequence_mask(ends, tf.shape(real)[1], dtype=real.dtype)

    pred = tf.argmax(pred, axis=-1)
    ends = tf.argmax(is_special_token(pred, tokenizer), axis=-1, output_type=tf.int32)
    pred_cleared = pred * tf.sequence_mask(ends, tf.shape(pred)[1], dtype=pred.dtype)

    hard_acc_log_map = tf.math.reduce_all(tf.equal(real_cleared, pred_cleared), axis=-1)
    return tf.reduce_mean(tf.cast(hard_acc_log_map, tf.float32)), hard_acc_log_map


def is_close_accuracy_metric(
    golden: np.ndarray, prediction: np.ndarray, rtol=0.05, atol=1e-3
):
    return np.mean(np.isclose(prediction, golden, rtol=rtol, atol=atol)) > 0.95
