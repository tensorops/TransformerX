import tensorflow as tf
import numpy as np
import pytest

from transformerx.layers import AbsolutePositionalEncoding

class TestAbsolutePositionalEncoding:

    @pytest.fixture
    def layer(self):
        depth = 16
        max_len = 20
        return AbsolutePositionalEncoding(depth=depth, max_len=max_len)
    def test_output_shape(self):
        depth = 64
        max_len = 100
        layer = AbsolutePositionalEncoding(depth, max_len=max_len)
        input_shape = (5, max_len, depth)
        input_tensor = tf.ones(input_shape)
        output_tensor = layer(input_tensor)
        assert output_tensor.shape == input_shape

    def test_call_batch_size_1(self, layer):
        input_shape = (1, 10, layer.depth)
        X = tf.ones(input_shape)
        output = layer(X)
        assert output.shape == input_shape
        assert not np.allclose(output.numpy(), X.numpy())
        assert np.allclose(output.numpy()[:, :5, :], X.numpy()[:, :5, :] + layer.P.numpy()[:, :5, :])


