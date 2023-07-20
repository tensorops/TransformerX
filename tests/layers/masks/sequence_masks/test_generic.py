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