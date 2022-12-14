import numpy as np
import pytest
import tensorflow as tf

from transformerx.utils import masked_softmax
from transformerx.layers import DotProductAttention


class TestDotProductAttention:
    # Set up the test class with some test data
    @pytest.fixture(autouse=True)
    def setup(self):
        self.x = tf.cast(np.random.random([2, 3, 2]), dtype=tf.float32)
        self.dot_product_scaled = DotProductAttention(0.2)
        self.dot_product_unscaled = DotProductAttention(dropout_rate=0.1, num_heads=8, scaled=False)

    # Test that the output shape of the `call` method is the same as the input shape of the queries, keys, and values
    def test_output_shape(self):
        queries, keys, values = self.x, self.x, self.x
        output_scaled = self.dot_product_scaled(queries, keys, values)
        output_unscaled = self.dot_product_unscaled(queries, keys, values)
        assert output_scaled.shape == (2, 3, 2)
        assert output_unscaled.shape == (2, 3, 2)

    # Test that the `call` method computes the correct dot-product attention
    def test_dot_product_attention(self):
        dot_product = DotProductAttention()
        x = self.x
        # Feed the input tensor to queries, keys, and values
        queries, keys, values = x, x, x

        # Compute the dot-product attention
        attention = dot_product(queries, keys, values)

        # Check that the attention tensor has the same shape as the input tensor
        assert attention.shape == x.shape

    def test_call(self):
        head_nums = [1, 2, 4, 8]
        x = self.x
        # Feed the input tensor to queries, keys, and values
        queries, keys, values = x, x, x
        for num in head_nums:
            dot_product = DotProductAttention(num_heads=num)

            # Compute the dot-product attention
            attention = dot_product(queries, keys, values)

            # Check that the attention tensor has the same shape as the input tensor
            assert attention.shape == queries.shape

    def test_call_with_different_input_tensor_shapes(self):
        # Create an instance of the DotProductAttention class
        dot_product = DotProductAttention()

        # Test the call method with 2D input tensors
        queries = tf.random.uniform([2, 2])
        keys = tf.random.uniform([2, 2])
        values = tf.random.uniform([2, 2])
        output = dot_product(queries, keys, values)
        assert output.shape == queries.shape

        # Test the call method with 3D input tensors
        queries = tf.random.uniform([2, 3, 2])
        keys = tf.random.uniform([2, 3, 2])
        values = tf.random.uniform([2, 3, 2])
        output = dot_product(queries, keys, values)
        assert output.shape == queries.shape
