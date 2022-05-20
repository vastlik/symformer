import math
from typing import Optional

import numpy as np
import tensorflow as tf

from official.nlp.modeling.ops import sampling_module

from symformer.dataset.tokenizers import Tokenizer
from symformer.model.base import TransformerBase
from symformer.model.utils.convertor import Convertor


def get_empty_tensor(dtype: tf.dtypes.DType, shape=(0, 0)):
    return tf.RaggedTensor.from_tensor(
        tf.reshape(tf.convert_to_tensor((), dtype=dtype), shape)
    )


def pad_with_one_hot(pred, max_len, vocab_size, pad_id):
    padding = tf.one_hot(pad_id, vocab_size)
    padding = tf.broadcast_to(padding, [max_len - tf.shape(pred)[0], vocab_size])
    return tf.concat([pred, padding], axis=0)


class Sampler:
    def sample(self, logits: tf.Tensor):
        raise NotImplementedError()


class TopK(Sampler):
    def __init__(self, k: int = 8):
        self.k = k

    def sample(self, logits: tf.Tensor):
        return sampling_module.sample_top_k(logits, self.k)


class TopP(Sampler):
    def __init__(self, p=0.9):
        self.p = p

    def sample(self, logits: tf.Tensor):
        return sampling_module.sample_top_p(logits, self.p)


class TemperatureSampling(Sampler):
    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def sample(self, logits: tf.Tensor):
        return sampling_module.sample_logits_with_temperature(logits, self.temperature)


class Search:
    def __init__(
        self,
        sampler: Optional[Sampler],
        max_len: int,
        tokenizer: Tokenizer,
        model: TransformerBase,
        reg_input: bool,
        convertor: Convertor,
    ):
        self.sampler = sampler
        self.max_len = max_len
        self.start_id = tokenizer.start_id
        self.pad_id = tokenizer.pad_id
        self.end_id = tokenizer.end_id
        # todo add constant id
        self.vocab_size = len(tokenizer.vocab)
        self.model = model
        self.reg_input = reg_input
        self.convertor = convertor
        self.tokenizer = tokenizer

    def decode(self, points: tf.Tensor):
        raise NotImplementedError()

    def batch_decode(self, points: tf.Tensor):
        raise NotImplementedError()


class RandomSampler(Search):
    def __init__(
        self,
        sampler: Optional[Sampler],
        max_len: int,
        tokenizer: Tokenizer,
        model: TransformerBase,
        reg_input: bool,
        convertor: Convertor,
        beam_size: int,
    ):

        super().__init__(sampler, max_len, tokenizer, model, reg_input, convertor)
        self.beam_size = beam_size
        self.max_num_beams = 1024

    def repeat_tensor(self, tensor: tf.Tensor, size: int):
        return tf.repeat(
            tensor, self.vocab_size * tf.ones(size, dtype=tf.int32), axis=0
        )

    def get_init_values(self, size: int):
        return (
            tf.convert_to_tensor([[self.start_id]] * size, dtype=tf.int64),
            tf.convert_to_tensor([[0.0]] * size),
            tf.one_hot([[self.start_id]] * size, self.vocab_size),
        )

    def predict(
        self,
        encoder_state: tf.Tensor,
        outputs: tf.Tensor,
        reg_outputs: tf.Tensor = None,
    ):
        encoder_state = tf.broadcast_to(
            encoder_state,
            shape=(
                tf.shape(outputs)[0],
                tf.shape(encoder_state)[1],
                tf.shape(encoder_state)[2],
            ),
        )

        if self.reg_input:
            logits, reg_preds, _ = self.model.call_without_encoder(
                outputs, reg_outputs, encoder_state
            )
            preds = tf.nn.softmax(logits[:, -1])
            return preds, reg_preds[:, -1], logits[:, -1:]
        else:
            logits, _ = self.model.call_without_encoder(outputs, encoder_state)
            preds = tf.nn.softmax(logits[:, -1])
            return preds, logits[:, -1:]

    def generate_indexes(self, total_size: int):
        idxs = tf.range(0, total_size) % self.vocab_size
        return tf.reshape(idxs, (-1, 1))

    def batch_decode(self, points: tf.Tensor, return_all: bool = False):
        final_preds = []
        final_reg_preds = []
        final_logits = []
        final_preds_inorder = []
        final_preds_symbolic = []
        for i in range(tf.shape(points)[0]):
            # here we could select the best one just by evaluating, no need to convert it to sympy
            if self.reg_input:
                preds, reg_preds, logits = self.decode(points[i])
            else:
                preds, logits = self.decode(points[i])
                reg_preds = None

            (
                preds,
                reg_preds,
                logits,
                preds_inorder,
                preds_symbolic,
            ) = self.convertor.convert(
                preds, reg_preds, points[i], logits, return_all=return_all
            )
            final_preds.append(preds)
            final_reg_preds.append(reg_preds)
            final_logits.append(logits)
            final_preds_inorder.append(preds_inorder)
            final_preds_symbolic.append(preds_symbolic)

        if not return_all:
            # stack
            maxi_reg_preds = np.max([tf.shape(i)[0] for i in final_reg_preds])
            maxi_preds = np.max([tf.shape(i)[0] for i in final_preds])
            final_preds = tf.stack(
                [
                    tf.pad(pred, [[0, maxi_preds - tf.shape(pred)[0]]])
                    for pred in final_preds
                ]
            )
            final_reg_preds = tf.stack(
                [
                    tf.pad(pred, [[0, maxi_reg_preds - tf.shape(pred)[0]]])
                    for pred in final_reg_preds
                ]
            )

            final_logits = tf.stack(
                [
                    pad_with_one_hot(pred, maxi_preds, self.vocab_size, self.pad_id)
                    for pred in final_logits
                ]
            )

        return (
            final_preds,
            final_reg_preds,
            final_logits,
            final_preds_inorder,
            final_preds_symbolic,
        )

    @tf.function
    def decode(self, points: tf.Tensor):
        # maybe there should be strategy.run to get the results
        encoder_state = self.model.encoder(tf.expand_dims(points, 0), False)
        finished = get_empty_tensor(tf.int64)
        reg_finished = get_empty_tensor(tf.float32)
        logits_finished = get_empty_tensor(tf.float32, shape=(0, 0, self.vocab_size))

        iterations = math.ceil(self.beam_size / self.max_num_beams)
        reg_preds = tf.zeros([self.beam_size, 1])
        for j in range(iterations):
            if j == iterations - 1 and self.beam_size % self.max_num_beams != 0:
                size = self.beam_size % self.max_num_beams
            else:
                size = self.max_num_beams

            outputs, reg_outputs, logits_outputs = self.get_init_values(size)

            for i in tf.range(self.max_len):
                tf.autograph.experimental.set_loop_options(
                    maximum_iterations=self.max_len,
                    shape_invariants=[
                        (outputs, tf.TensorShape([None, None])),
                        (reg_outputs, tf.TensorShape([None, None])),
                        (logits_outputs, tf.TensorShape([None, None, None])),
                        (finished, tf.TensorShape([None, None])),
                        (reg_finished, tf.TensorShape([None, None])),
                        (logits_finished, tf.TensorShape([None, None, None])),
                        (reg_preds, tf.TensorShape([None, None])),
                    ],
                )

                if self.reg_input:
                    probs, reg_preds, logits = self.predict(
                        encoder_state, outputs, reg_outputs
                    )
                else:
                    probs, logits = self.predict(encoder_state, outputs)

                sampled_logits = self.sampler.sample(logits[:, -1])
                idxs = tf.random.categorical(
                    tf.nn.log_softmax(sampled_logits), dtype=tf.int64, num_samples=1
                )

                outputs = tf.concat([outputs, idxs], axis=-1)

                if self.reg_input:
                    reg_preds = tf.where(
                        tf.reduce_all(
                            idxs
                            != tf.cast(self.tokenizer.get_constant_ids(), tf.int64),
                            axis=-1,
                            keepdims=True,
                        ),
                        0.0,
                        reg_preds,
                    )

                    reg_outputs = tf.concat([reg_outputs, reg_preds], axis=-1)

                logits_outputs = tf.concat([logits_outputs, logits[:, -1:]], axis=1)

                finished_mask = tf.squeeze(idxs == self.end_id, axis=-1)
                if i >= 2:
                    finished = tf.concat([finished, outputs[finished_mask]], axis=0)
                    if self.reg_input:
                        reg_finished = tf.concat(
                            [reg_finished, reg_outputs[finished_mask]], axis=0
                        )
                    logits_finished = tf.concat(
                        [logits_finished, logits_outputs[finished_mask]], axis=0
                    )
                finished_mask = finished_mask | tf.squeeze(idxs == self.pad_id, axis=-1)
                finished_mask = finished_mask | tf.squeeze(
                    idxs == self.start_id, axis=-1
                )

                outputs = outputs[~finished_mask]
                if self.reg_input:
                    reg_outputs = reg_outputs[~finished_mask]

                logits_outputs = logits_outputs[~finished_mask]

        if self.reg_input:
            return (
                finished.to_tensor(0)[:, 1:],
                reg_finished.to_tensor(0.0)[:, 1:],
                logits_finished.to_tensor(tf.one_hot(self.pad_id, self.vocab_size))[
                    :, 1:
                ],
            )
        else:
            return (
                finished.to_tensor(0)[:, 1:],
                logits_finished.to_tensor(tf.one_hot(self.pad_id, self.vocab_size))[
                    :, 1:
                ],
            )


class GreedySearch(Search):
    def batch_decode(self, points: tf.Tensor):
        preds, reg_preds, logits = self.decode(points)
        if self.reg_input:
            return self.convertor.convert(preds, reg_preds, points, logits)
        else:
            return self.convertor.convert(preds, None, points, logits)

    @tf.function
    def decode(self, points: tf.Tensor):
        start = tf.broadcast_to(self.start_id, [tf.shape(points)[0]])
        end = tf.convert_to_tensor(self.end_id, dtype=tf.int32)

        output_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        logits_array = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        regression_output_array = tf.TensorArray(
            dtype=tf.float32, size=0, dynamic_size=True
        )
        regression_output_array = regression_output_array.write(
            0, tf.zeros(tf.shape(start))
        )

        all_predicted = tf.zeros(tf.shape(start), dtype=tf.bool)

        enc_output = self.model.encoder(
            points, False
        )  # (batch_size, inp_seq_len, d_model)
        regression_predictions = tf.zeros([tf.shape(points)[0], 1])
        for i in tf.range(self.max_len):
            tf.autograph.experimental.set_loop_options(
                maximum_iterations=self.max_len,
                shape_invariants=[
                    (all_predicted, tf.TensorShape([None])),
                    (regression_predictions, tf.TensorShape([None, None])),
                ],
            )
            output = tf.transpose(output_array.stack())
            reg_output = tf.transpose(regression_output_array.stack())
            if self.reg_input:
                (
                    predictions,
                    regression_predictions,
                    _,
                ) = self.model.call_without_encoder(output, reg_output, enc_output)
                regression_predictions = regression_predictions[:, -1]
            else:
                predictions, _ = self.model.call_without_encoder(output, enc_output)

            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            logits_array = logits_array.write(i, tf.squeeze(predictions, axis=1))
            predicted_id = tf.argmax(predictions, axis=-1, output_type=tf.int32)

            if self.reg_input:
                regression_predictions = tf.where(
                    tf.reduce_all(
                        predicted_id != self.tokenizer.get_constant_ids(),
                        axis=-1,
                        keepdims=True,
                    ),
                    0.0,
                    regression_predictions,
                )
                regression_predictions = tf.reshape(regression_predictions, [-1, 1])

            output_array = output_array.write(i + 1, tf.reshape(predicted_id, [-1]))

            if self.reg_input:
                regression_output_array = regression_output_array.write(
                    i + 1, tf.reshape(regression_predictions, [-1])
                )

            all_predicted = tf.reshape(
                tf.math.logical_or(all_predicted, tf.squeeze(predicted_id == end)),
                tf.shape(start),
            )

            if tf.reduce_all(all_predicted):
                break
        return (
            tf.transpose(output_array.stack())[:, 1:],
            tf.transpose(regression_output_array.stack())[:, 1:],
            tf.transpose(logits_array.stack(), [1, 0, 2]),
        )
