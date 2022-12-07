import pytest
import numpy as np
import tensorflow as tf

from transformerx.layers import MultiHeadAttention


def test_multihead_attention():
    # Create an instance of the MultiHeadAttention class with 4 attention heads
    multihead = MultiHeadAttention(d_model=8, num_heads=4)

    # Define some input tensor with shape [batch_size, sequence_length, d_model]
    x = tf.constant(np.random.random([2, 3, 8]), dtype=tf.float32)

    # Apply the multi-head attention to the input tensor
    output = multihead(x, x, x)

    # Check that the output has the expected shape [batch_size, sequence_length, d_model]
    assert output.shape == (2, 3, 8)

# Create a test case for the MultiHeadAttention class
@pytest.fixture
def multihead_attention():
    return MultiHeadAttention(d_model=8, num_heads=3, dropout_rate=0.0, bias=False)

# Test the initialization of the MultiHeadAttention class
def test_multihead_attention_init(multihead_attention):
    assert multihead_attention.d_model == 8
    assert multihead_attention.num_heads == 3
    assert multihead_attention.dropout_rate == 0.0
    assert multihead_attention.bias == False

# Test the forward pass of the MultiHeadAttention class
def test_multihead_attention_forward(multihead_attention):
    x = tf.constant(np.random.random([2, 3, 2]), dtype=tf.float32)
    output = multihead_attention(x, x, x)

    assert output.shape == (2, 3, 8)

def test_split_heads():
    # Create a random tensor with 3 dimensions
    x = tf.random.uniform((2, 3, 10))

    # Create a MultiHeadAttention layer with 4 attention heads
    multihead = MultiHeadAttention(num_heads=4)

    # Test the split_heads method with x as input
    assert multihead.split_heads(x).shape == (2, 4, 3, 5)

def test_inverse_transpose_qkv(self):
    # Test 1
    x = tf.random.uniform(shape=(2, 4, 3, 6))
    expected_output_shape = (2, 3, 24)
    output = self.multihead_attention.inverse_transpose_qkv(x)
    self.assertEqual(output.shape, expected_output_shape)

    # Test 2
    x = tf.random.uniform(shape=(2, 1, 3, 6))
    expected_output_shape = (2, 3, 6)
    output = self.multihead_attention.inverse_transpose_qkv(x)
    self.assertEqual(output.shape, expected_output_shape)

    # Test 3
    x = tf.random.uniform(shape=(2, 4, 1, 6))
    expected_output_shape = (2, 1, 24)
    output = self.multihead_attention.inverse_transpose_qkv(x)
    self.assertEqual(output.shape, expected_output_shape)