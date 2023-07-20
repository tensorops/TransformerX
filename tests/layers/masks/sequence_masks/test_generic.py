import pytest
import tensorflow as tf

from transformerx.layers.masks import PaddingMask
from transformerx.layers.masks.core import BaseMask


class TestPaddingMask:
    def test_valid_lengths(self):
        scores = tf.constant([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=tf.float32)
        valid_lens = tf.constant([3, 2])
        padding_mask = PaddingMask()
        masked = padding_mask(scores=scores, valid_lens=valid_lens)
        expected = tf.constant([[1, 2, 3, -1e9], [4, 5, -1e9, -1e9]])
        assert tf.reduce_all(tf.equal(masked, expected))

    def test_padding_value_0(self):
        scores = tf.constant([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=tf.float32)
        padding_mask = PaddingMask(padding_value=0)
        masked = padding_mask(scores=scores)
        expected = tf.constant([[1, 2, 3, -1e9], [4, 5, -1e9, -1e9]])
        assert tf.reduce_all(tf.equal(masked, expected))

    def test_padding_value_1(self):
        scores = tf.constant([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=tf.float32)
        padding_mask = PaddingMask(padding_value=1)
        masked = padding_mask(scores=scores)
        expected = tf.constant([[-1e9, 2, 3, 0], [4, 5, 0, 0]])
        assert tf.reduce_all(tf.equal(masked, expected))

    def test_multihead(self):
        scores = tf.constant(
            [[[1, 2, 3, 0], [4, 5, 0, 0]], [[1, 2, 0, 0], [4, 5, 6, 0]]],
            dtype=tf.float32,
        )
        padding_mask = PaddingMask(multihead=True)
        masked = padding_mask(scores=scores, valid_lens=tf.constant([3, 2]))
        expected = tf.constant(
            [
                [[1, 2, 3, -1e9], [4, 5, 0, -1e9]],
                [[1, 2, -1e9, -1e9], [4, 5, -1e9, -1e9]],
            ]
        )
        assert tf.reduce_all(tf.equal(masked, expected))
