import tensorflow as tf

from transformerx.layers.addnorm import AddNorm
from transformerx.layers.multihead_attention import MultiHeadAttention
from transformerx.layers.positionwise_ffn import PositionWiseFFN


class TransformerEncoderBlock(tf.keras.layers.Layer):
    """Transformer encoder block."""

    def __init__(
            self,
            num_hiddens,
            norm_shape,
            ffn_num_hiddens,
            num_heads,
            dropout,
            bias=False,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout, bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def call(self, X, valid_lens, **kwargs):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens, **kwargs), **kwargs)
        return self.addnorm2(Y, self.ffn(Y), **kwargs)
