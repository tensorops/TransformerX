import pytest
import numpy as np
import tensorflow as tf

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
        return MultiHeadAttention(d_model=16, num_heads=4, dropout=0.1)

    def test_split_heads(self, attention):
        queries = tf.random.normal((3, 10, 16))
        queries_split = attention.split_heads(queries)
        assert queries_split.shape == (3, 4, 10, 4)

    def test_inverse_transpose_qkv(self, attention):
        queries_split = tf.random.normal((3, 4, 10, 4))
        queries = attention.inverse_transpose_qkv(queries_split)
        assert queries.shape == (3, 10, 16)

    def test_call_without_window_mask(self, attention):
        queries = tf.random.normal((3, 10, 16))
        keys = tf.random.normal((3, 20, 16))
        values = tf.random.normal((3, 20, 16))
        valid_lens = tf.constant([10, 15, 20])
        output = attention(queries, keys, values, valid_lens)
        assert output.shape == (3, 10, 16)

    def test_call_with_window_mask(self, attention):
        queries = tf.random.normal((3, 10, 16))
        keys = tf.random.normal((3, 20, 16))
        values = tf.random.normal((3, 20, 16))
        valid_lens = tf.constant([10, 15, 20])
        window_mask = tf.ones((3, 10, 20))
        output = attention(queries, keys, values, valid_lens, window_mask)
        assert output.shape == (3, 10, 16)