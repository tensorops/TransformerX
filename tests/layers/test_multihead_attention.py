import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras

from transformerx.layers import MultiHeadAttention


# def test_multihead_attention():
#     # Create an instance of the MultiHeadAttention class with 4 attention heads
#     multihead = MultiHeadAttention(d_model=8, num_heads=4)
#
#     # Define some input tensor with shape [batch_size, sequence_length, d_model]
#     x = tf.constant(np.random.random([2, 3, 8]), dtype=tf.float32)
#
#     # Apply the multi-head attention to the input tensor
#     output = multihead(x, x, x)
#
#     # Check that the output has the expected shape [batch_size, sequence_length, d_model]
#     assert output.shape == (2, 3, 8)
#
# # Create a test case for the MultiHeadAttention class
# @pytest.fixture
# def multihead_attention():
#     return MultiHeadAttention(d_model=8, num_heads=3, dropout_rate=0.0, bias=False)
#
# # Test the initialization of the MultiHeadAttention class
# def test_multihead_attention_init(multihead_attention):
#     assert multihead_attention.d_model == 8
#     assert multihead_attention.num_heads == 3
#     assert multihead_attention.dropout_rate == 0.0
#     assert multihead_attention.bias == False
#
# def test_split_heads():
#     # Create a random tensor with 3 dimensions
#     x = tf.random.uniform((2, 3, 24))
#
#     # Create a MultiHeadAttention layer with 4 attention heads
#     multihead = MultiHeadAttention(num_heads=8)
#
#     # Test the split_heads method with x as input
#     assert multihead.split_heads(x).shape == (2, 8, 3, 3)
#
# def test_inverse_transpose_qkv(multihead_attention):
#     # Test 1
#     x = tf.random.uniform(shape=(2, 4, 3, 6))
#     expected_output_shape = (2, 3, 24)
#     output = multihead_attention.inverse_transpose_qkv(x)
#     assert output.shape == expected_output_shape
#
#     # Test 2
#     x = tf.random.uniform(shape=(2, 1, 3, 6))
#     expected_output_shape = (2, 3, 6)
#     output = multihead_attention.inverse_transpose_qkv(x)
#     assert output.shape == expected_output_shape
#
#     # Test 3
#     x = tf.random.uniform(shape=(2, 4, 1, 6))
#     expected_output_shape = (2, 1, 24)
#     output = multihead_attention.inverse_transpose_qkv(x)
#     assert output.shape == expected_output_shape
#
# # Set the random seed for reproducibility
# np.random.seed(42)
#
# # Define the dimensions of the input tensor and the number of heads
# d_model = 8
# num_heads = 4
#
# # Create a random tensor as input
# x = tf.constant(np.random.random([2, 3, d_model]), dtype=tf.float32)
#
# # Create a MultiHeadAttention object with the specified dimensions
# multihead = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
#
# # Test the call method
# def test_call():
#     # The call method should return a tensor with the concatenated attention heads
#     output = multihead(x, x, x)
#     assert output.shape == (2, 3, d_model)
#
#     # Test that the output has the expected number of attention heads
#     num_heads_output = output.shape[-1] / d_model
#     assert num_heads_output == num_heads
#
#

class TestMultiHeadAttention:
    @pytest.fixture
    def attention(self):
        return MultiHeadAttention(d_model=32, num_heads=4, dropout=0.1)

    @pytest.fixture
    def inputs(self):
        # Create some dummy input tensors for testing
        x = tf.constant(np.random.random([2, 3, 2]), dtype=tf.float32)
        y = tf.constant(np.random.random([2, 3, 2]), dtype=tf.float32)
        z = tf.constant(np.random.random([2, 3, 2]), dtype=tf.float32)
        return x, y, z

    def test_split_heads(self, attention):
        queries = tf.random.normal((3, 10, 16))
        queries_split = attention.split_heads(queries)
        assert queries_split.shape == (3, 4, 10, 4)

    def test_multihead_attention_init(self):
        # Test the initialization of the MultiHeadAttention class
        multihead = MultiHeadAttention(d_model=8)
        assert isinstance(multihead, MultiHeadAttention)

    def test_multihead_attention_call(self, inputs):
        # Test the call method of the MultiHeadAttention class
        x, y, z = inputs
        multihead = MultiHeadAttention(d_model=8)
        output = multihead(x, y, z)
        assert output.shape == (2, 3, 8)

    def test_inverse_transpose_qkv(self, attention):
        queries_split = tf.random.normal((3, 4, 10, 4))
        queries = attention.inverse_transpose_qkv(queries_split)
        assert queries.shape == (3, 10, 16)

    def test_multihead_attention_with_mask(self):
        attention = MultiHeadAttention(d_model=64, num_heads=8)
        # create a batch of inputs and a mask
        inputs = tf.random.normal((32, 64, 64), dtype=tf.float32)
        mask = tf.random.uniform((32, 1, 64), dtype=tf.float32)

        # call the layer with the inputs and mask
        outputs = attention(inputs, inputs, inputs, attention_mask=mask)

        # check that the output shape is correct
        assert outputs.shape == (32, 64, 64)

        # check that the output values are not all zero
        assert not tf.reduce_all(tf.math.equal(outputs, 0.0))

        # x = tf.constant(np.random.random([2, 3, 20]), dtype=tf.float32)
        # mask = tf.constant([[1, 1, 0], [1, 0, 0]], dtype=tf.float32)
        # weights = attention(x, x, x, attention_mask=mask)
        # assert weights[0, 0, 2].numpy() == 0
        # assert weights[0, 1, 2] == 0
        # assert weights[1, 0, 1] == 0

    def test_call_with_causal_mask(self, attention):
        queries = tf.random.normal((4, 10, 32))
        values = tf.random.normal((4, 20, 32))
        keys = tf.random.normal((4, 20, 32))

        output = attention(queries, values, keys, causal_mask=True)

        assert output.shape == (4, 10, 32)

    def test_call_with_both_masks(self, attention):
        queries = tf.random.normal((4, 10, 32))
        values = tf.random.normal((4, 10, 64))
        keys = tf.random.normal((4, 10, 32))
        attention_mask = tf.random.uniform((4, 10), maxval=2, dtype=tf.int32)

        output = attention(queries, keys, values, attention_mask=attention_mask, causal_mask=True)

        assert output.shape == (4, 10, 32)

    def test_call_with_zero_mask(self, attention):
        queries = tf.random.normal((4, 10, 32))
        values = tf.random.normal((4, 10, 64))
        keys = tf.random.normal((4, 10, 32))
        attention_mask = tf.zeros((4, 10), dtype=tf.int32)

        output = attention(queries, keys, values, attention_mask=attention_mask)

        assert output.shape == (4, 10, 32)

    def test_call_with_ones_mask(self, attention):
        queries = tf.random.normal((4, 10, 32))
        values = tf.random.normal((4, 10, 64))
        keys = tf.random.normal((4, 10, 32))
        attention_mask = tf.ones((4, 10), dtype=tf.int32)

        output = attention(queries, keys, values, attention_mask=attention_mask)

        assert output.shape == (4, 10, 32)