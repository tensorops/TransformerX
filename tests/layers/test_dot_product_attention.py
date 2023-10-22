import numpy as np
import pytest
import tensorflow as tf

from transformerx.layers import DotProductAttention


class TestDotProductAttention:
    "this class tests the dot-product attention class"

    # Set up the test class with some test data
    @pytest.fixture(autouse=True)
    def setup(self):
        self.x = tf.cast(np.random.random([2, 3, 2]), dtype=tf.float32)
        self.dot_product_scaled = DotProductAttention(0.2)
        self.dot_product_unscaled = DotProductAttention(dropout_rate=0.1, scaled=False)

    # Test that the output shape of the `call` method is the same as the input shape of the queries, keys, and values
    def test_output_shape(self):
        queries, keys, values = self.x, self.x, self.x
        output_scaled, _ = self.dot_product_scaled(queries, keys, values)
        output_unscaled, _ = self.dot_product_unscaled(queries, keys, values)
        assert output_scaled.shape == (2, 3, 2)
        assert output_unscaled.shape == (2, 3, 2)

        q_in = tf.cast(np.random.random([2, 8, 10, 20]), dtype=tf.float32)
        k_in = tf.cast(np.random.random([2, 8, 15, 20]), dtype=tf.float32)
        v_in = tf.cast(np.random.random([2, 8, 15, 20]), dtype=tf.float32)
        queries, keys, values = q_in, k_in, v_in
        output_scaled, _ = self.dot_product_scaled(queries, keys, values)
        assert output_scaled.shape == (2, 8, 10, 20)

    # Test that the `call` method computes the correct dot-product attention
    def test_dot_product_attention(self):
        dot_product = DotProductAttention()
        x = self.x
        # Feed the input tensor to queries, keys, and values
        queries, keys, values = x, x, x

        # Compute the dot-product attention
        attention, _ = dot_product(queries, keys, values)

        # Check that the attention tensor has the same shape as the input tensor
        assert attention.shape == x.shape

    def test_call(self):
        head_nums = [1, 2, 4, 8]
        x = self.x
        # Feed the input tensor to queries, keys, and values
        queries, keys, values = x, x, x
        for num in head_nums:
            dot_product = DotProductAttention()

            # Compute the dot-product attention
            attention, _ = dot_product(queries, keys, values)

            # Check that the attention tensor has the same shape as the input tensor
            assert attention.shape == queries.shape

    def test_call_with_different_input_tensor_shapes(self):
        # Create an instance of the DotProductAttention class
        dot_product = DotProductAttention()

        # Test the call method with 2D input tensors
        queries = tf.random.uniform([2, 2])
        keys = tf.random.uniform([2, 2])
        values = tf.random.uniform([2, 2])
        output, _ = dot_product(queries, keys, values)
        assert output.shape == queries.shape

        # Test the call method with 3D input tensors
        queries = tf.random.uniform([2, 3, 2])
        keys = tf.random.uniform([2, 3, 2])
        values = tf.random.uniform([2, 3, 2])
        output, _ = dot_product(queries, keys, values)
        assert output.shape == queries.shape

    @pytest.mark.parametrize(
        "queries, keys, values, expected_output_shape, expected_output_values",
        [
            (
                tf.zeros((2, 3, 4)),
                tf.zeros((2, 3, 4)),
                tf.zeros((2, 3, 4)),
                (2, 3, 4),
                0,
            ),
            (tf.ones((2, 3, 4)), tf.ones((2, 3, 4)), tf.ones((2, 3, 4)), (2, 3, 4), 1),
        ],
    )
    def test_call_with_different_values(
        self, queries, keys, values, expected_output_shape, expected_output_values
    ):
        attention = DotProductAttention()
        output, _ = attention(queries, keys, values)

        assert output.shape == expected_output_shape
        assert tf.reduce_all(output == expected_output_values)

    @pytest.fixture
    def attention_layer(self):
        return DotProductAttention(dropout_rate=0.2, scaled=True)

    def test_from_config(self, attention_layer):
        config = attention_layer.get_config()
        new_layer = DotProductAttention.from_config(config)
        assert attention_layer.dropout.rate == new_layer.dropout.rate
        assert attention_layer.scaled == new_layer.scaled

    def test_get_attention_weights(self, attention_layer):
        attention_layer.attention_weights = np.random.rand(5, 10)
        weights = attention_layer.get_attention_weights()
        assert weights.shape == (5, 10)

    def test_get_config(self, attention_layer):
        config = attention_layer.get_config()
        print(config)
        assert isinstance(config, dict)
        assert config["dropout_rate"] == 0.2
        assert config["scaled"] == True

    def test_causal_masking(self, attention_layer):
        # Test the attention with causal masking
        queries = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.float32)
        keys = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.float32)
        values = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.float32)
        attention_mask = tf.constant([[1, 1], [0, 1]], dtype=tf.float32)

        attention_layer = DotProductAttention(causal_mask=True)
        # Call the attention layer with causal masking
        output, attention_weights = attention_layer(queries, keys, values)

        assert output.shape == values.shape, "Output shape mismatch"
        assert attention_weights.shape == (2, 2, 2), "Attention weights shape mismatch"
