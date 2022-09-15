import os

import numpy as np
import pytest
import tensorflow as tf

from transformerx.layers import (
    MultiHeadAttention, PositionalEncoding, PositionWiseFFN, AddNorm,
    TransformerEncoderBlock, TransformerEncoder, TransformerDecoderBlock, DotProductAttention,
)
from transformerx.txplot import Plot


@pytest.fixture()
def test_transpose_qkv():
    x = np.random.random([100, 10, 5])
    assert MultiHeadAttention.split_heads(x, x)

x = tf.constant(np.random.random([2, 3, 2]), dtype=tf.float32)
multihead = MultiHeadAttention(d_model=8)
print(multihead)
output = multihead(x, x, x)
print(output)
# ====================================


depth, num_steps = 32, 50
pos_encoding = PositionalEncoding(depth, 0)
X = pos_encoding(tf.zeros((2, num_steps, depth)), training=False)
P = pos_encoding.P[:, : X.shape[1], :]
plotter = Plot()
plotter.plot_pe(np.arange(7, 11), P, num_steps)
# plotter.plot_pe(np.arange(7, 11), P, num_steps, position=0)
# ====================================


ffn = PositionWiseFFN(4, 8)
print(ffn(tf.ones((2, 3, 4))).shape)
# ====================================

add_norm = AddNorm([1, 2], 0.5)
print(add_norm(tf.ones((2, 3, 4)), tf.ones((2, 3, 4)), training=False).shape)
# Also:
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if __name__ == "__main__":
    X = tf.constant(np.arange(10).reshape(5, 2) * 10, dtype=tf.float32)
    Y = tf.constant(np.arange(10).reshape(5, 2) * 10, dtype=tf.float32)
    norm_shape = [0, 1]

    dropout = tf.keras.layers.Dropout(0)
    output = dropout(Y) + X
    print("dropout_rate: ", output)

    addnorm = AddNorm(norm_shape, .2)
    output = addnorm(X, Y)
    print(output)
# ====================================

ffn = PositionWiseFFN(4, 8)
print(ffn(tf.ones((2, 3, 4)))[0])
# ====================================

x = tf.cast(np.random.random([2, 3, 2]), dtype=tf.float32)
print(x)
dot = DotProductAttention(0.2)
dot2 = DotProductAttention(dropout_rate=0.1, num_heads=8, scaled=False)
queries, keys, values = x, x, x
output = dot(queries, keys, values)
output2 = dot2(queries, keys, values)
print(x.shape)
print(output)
print(output2)
print(output == output2)
# ====================================

X = tf.ones((2, 100, 24))
valid_lens = tf.constant([3, 2])
norm_shape = [i for i in range(len(X.shape))][1:]
encoder_block = TransformerEncoderBlock(24, norm_shape, 48, 8, 0.5)
print(encoder_block(X, valid_lens, training=False))
# ====================================

encoder = TransformerEncoder(200, 24, [1, 2], 48, 8, 2, 0.5)
print(encoder(tf.ones((2, 100)), valid_lens, training=False).shape, (2, 100, 24))
# ====================================

decoder_block = TransformerDecoderBlock(24, [1, 2], 48, 8, 0.5, 0)
state = [encoder_block(X, valid_lens), valid_lens, [None]]
print(decoder_block(X, state, training=False)[0].shape, X.shape)
# ====================================
