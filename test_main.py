import pytest
from layers.transformer_decoder_block import TransformerDecoderBlock
from layers.transformer_encoder import TransformerEncoder
from layers.transformer_encoder_block import TransformerEncoderBlock
from layers.addnorm import AddNorm
from layers.positionwise_ffn import PositionWiseFFN
from txplot.plot_pe import Plot
from layers.positional_encoding import PositionalEncoding
from layers.multihead_attention import MultiHeadAttention
import numpy as np
import tensorflow as tf


@pytest.fixture()
def test_transpose_qkv():
    x = np.random.random([100, 10, 5])
    assert MultiHeadAttention.split_heads(x, x)


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

X = tf.ones((2, 100, 24))
valid_lens = tf.constant([3, 2])
norm_shape = [i for i in range(len(X.shape))][1:]
encoder_blk = TransformerEncoderBlock(24, 24, 24, 24, norm_shape, 48, 8, 0.5)
print(encoder_blk(X, valid_lens, training=False))

encoder = TransformerEncoder(200, 24, 24, 24, 24, [1, 2], 48, 8, 2, 0.5)
print(encoder(tf.ones((2, 100)), valid_lens, training=False).shape, (2, 100, 24))

decoder_blk = TransformerDecoderBlock(24, 24, 24, 24, [1, 2], 48, 8, 0.5, 0)
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
print(decoder_blk(X, state, training=False)[0].shape, X.shape)
