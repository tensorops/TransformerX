import tensorflow as tf

from transformerx.layers.addnorm import AddNorm
from transformerx.layers.multihead_attention import MultiHeadAttention
from transformerx.layers.positionwise_ffn import PositionWiseFFN


class TransformerEncoderBlock(tf.keras.layers.Layer):
    """Transformer encoder block [1]_. 
    
    Include a stack of layers used in the transformer encoder block.
    
    Parameters
    ----------
    d_model : int
        Dimensions of the queries, keys, and values
    norm_shape :
        Arbitrary. Shape of the input.
    ffn_num_hiddens :
        Number of input hidden units
    num_heads : int
        Number of the heads in the multi-head attention
    dropout_rate :
        Float between 0 and 1. Fraction of the input units to drop.
    bias : bool - default = False
        Indicates the usage of bias in the dense layers (i.e. W_q, W_k, W_v, and W_o)
    """

    def __init__(
            self,
            d_model,
            norm_shape,
            ffn_num_hiddens,
            num_heads,
            dropout_rate,
            bias=False,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout_rate, bias)
        self.addnorm1 = AddNorm(norm_shape, dropout_rate)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, d_model)
        self.addnorm2 = AddNorm(norm_shape, dropout_rate)

    def call(self, X, valid_lens, **kwargs):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens, **kwargs), **kwargs)
        return self.addnorm2(Y, self.ffn(Y), **kwargs)
