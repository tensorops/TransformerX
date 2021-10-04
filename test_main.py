import pytest
from main import MultiHeadAttention, PositionalEncoding, Plot, PositionWiseFFN, AddNorm
import numpy as np
import tensorflow as tf


@pytest.fixture()
def test_transpose_qkv():
    x = np.random.random([100, 10, 5])
    assert MultiHeadAttention.transpose_qkv(x, x)


encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
X = pos_encoding(tf.zeros((2, num_steps, encoding_dim)), training=False)
P = pos_encoding.P[:, : X.shape[1], :]
plotter = Plot()
plotter.plot_pe(np.arange(7, 11), P, num_steps)
# plotter.plot_pe(np.arange(7, 11), P, num_steps, position=0)


ffn = PositionWiseFFN(4, 8)
print(ffn(tf.ones((2, 3, 4))).shape)

add_norm = AddNorm([1, 2], 0.5)
print(add_norm(tf.ones((2, 3, 4)), tf.ones((2, 3, 4)), training=False).shape)

ffn = PositionWiseFFN(4, 8)
print(ffn(tf.ones((2, 3, 4)))[0])
