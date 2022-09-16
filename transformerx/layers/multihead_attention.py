import numpy as np
import tensorflow as tf
from einops import rearrange

from transformerx.layers.dot_product_attention import DotProductAttention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head attention [1]_

    Multi-head attention class that runs through an attention mechanism (i.e. most commonly scaled dot-product) several
    times in parallel. The independent attention outputs are then concatenated and linearly transformed into the
    expected dimension.

    See Also
    --------
    layers.dot_product_attention : (Scaled) Dot-Product attention.

    Notes
    -----
    Intuitively, multiple attention heads allows for attending to parts of the sequence differently
    (e.g. longer-term dependencies versus shorter-term dependencies).

    It can be formulated as:

    .. math::
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where

    .. math::
        head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)

    Above :math:`W` are all learnable parameter matrices.
    For more please see [2]

    Parameters
    ----------
    d_model : int
        Dimensions of the queries, keys, and values
    num_heads : int
        Number of the heads in the multi-head attention
    dropout_rate : float
        Float between 0 and 1. Fraction of the input units to drop.
    bias : bool - default = False
        Indicates the usage of bias in the dense layers (i.e. W_q, W_k, W_v, and W_o)

    Returns
    -------
    output:
        Concatenated tensors

    Examples
    --------
    >>> x = tf.constant(np.random.random([2, 3, 2]), dtype=tf.float32)
    >>> multihead = MultiHeadAttention(d_model=8)
    >>> print(multihead)
    <__main__.MultiHeadAttention object at 0x7ff83c16bb80>

    >>> output = multihead(x, x, x)
    >>> print(output)
    tf.Tensor(
    [[[ 0.2051548   0.32050014  0.2915167  -0.04056092  0.12072253
        0.06477361  0.18725544  0.02056682]
      [ 0.19823116  0.2983173   0.27711272 -0.04071879  0.11172265
        0.06080601  0.18654731  0.00577436]
      [ 0.19831955  0.30106473  0.27666807 -0.03963682  0.11234044
        0.0615251   0.18657821  0.00680977]]
     [[ 0.14630345  0.21267754  0.26289055 -0.10759152  0.03963668
        0.04118761  0.11257525  0.05869889]
      [ 0.14556082  0.21070784  0.26139364 -0.10755821  0.03894955
        0.04060047  0.11260018  0.05745776]
      [ 0.14547291  0.21081978  0.26109838 -0.10745162  0.03889
        0.04069766  0.11251941  0.05741404]]], shape=(2, 3, 8), dtype=float32)

    Process finished with exit code 0


    References
    ----------
    .. [1] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, I. Polosukhin, Attention
        is all you need, in: NIPS, pp. 5998–6008.

    .. [2] Transformers in Action: Attention Is All You Need
        https://towardsdatascience.com/transformers-in-action-attention-is-all-you-need-ac10338a023a#d417
    """

    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 8,
            dropout_rate: float = 0,
            bias: bool = False,
            **kwargs,
    ):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout_rate, num_heads)
        self.W_q = tf.keras.layers.Dense(d_model, use_bias=bias)
        self.W_k = tf.keras.layers.Dense(d_model, use_bias=bias)
        self.W_v = tf.keras.layers.Dense(d_model, use_bias=bias)
        self.W_o = tf.keras.layers.Dense(d_model, use_bias=bias)

    def split_heads(self, X: tf.Tensor) -> tf.Tensor:
        """Transpose tensors for parallel computation of attention heads.

        First transposition produces a tensor of shape x: (batch_size, num_heads, no. of queries or key-value pairs,
        depth / num_heads).
        Next it is rearranged to a new order (batch_size * num_heads, no. of queries or key-value pairs,
        depth / num_heads) which is then passed to the last rearrangement and returned.

        Parameters
        ----------
        X : Shape (batch_size, no. of queries or key-value pairs, depth).
            The tensor to be transposed and prepared for the multi-head attention layer (i.e. queries, keys, and values)
        Returns
        -------
        x : tf.Tensor
            Transposed tensor of shape ((batch_size * num_heads, no. of queries or key-value pairs, depth / num_heads)
        """

        # x = tf.reshape(x, shape=(x.shape[0], x.shape[1], self.num_heads, -1))
        X = rearrange(X, "b h (heads hidden) -> b h heads hidden", heads=self.num_heads)
        # print("x reshaped: ", x.shape)
        # x = tf.transpose(x, perm=(0, 2, 1, 3))
        X = rearrange(X, "b d1 d2 d3 -> b d2 d1 d3")
        # print("x transposed: ", x.shape)
        # return tf.reshape(x, shape=(-1, x.shape[2], x.shape[3]))
        X = rearrange(X, "b d1 d2 d3 -> (b d1) d2 d3")
        # print("x reshaped2: ", x.shape)
        return X

    def inverse_transpose_qkv(self, X):
        """Reverse the operation of split_heads."""
        X = tf.reshape(X, shape=(-1, self.num_heads, X.shape[1], X.shape[2]))
        X = tf.transpose(X, perm=(0, 2, 1, 3))
        return tf.reshape(X, shape=(X.shape[0], X.shape[1], -1))

    def call(self, queries: tf.Tensor, values: tf.Tensor, keys: tf.Tensor, valid_lens: tf.Tensor = None,
             window_mask: bool = None, **kwargs) -> tf.Tensor:
        # Shape of queries, keys, or values:
        # (batch_size, no. of queries or key-value pairs, depth)
        # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
        # After transposing, shape of output queries, keys, or values:
        # (batch_size * num_heads, no. of queries or key-value pairs,
        # depth / num_heads)

        # print("wq(queries): ", self.W_q(queries).shape)
        # print("queries: ", queries.shape)
        queries = self.split_heads(self.W_q(queries))
        # print("keys: ", keys.shape)
        keys = self.split_heads(self.W_k(keys))
        # print("values: ", values.shape)
        values = self.split_heads(self.W_v(values))

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for num_heads
            # times, then copy the next item, and so on
            valid_lens = tf.repeat(valid_lens, repeats=self.num_heads, axis=0)

        # Shape of output: (batch_size * num_heads, no. of queries,
        # depth / num_heads)
        output = self.attention(
            queries, keys, values, valid_lens, window_mask, **kwargs
        )

        # Shape of output_concat: (batch_size, no. of queries, depth)
        output_concat = self.inverse_transpose_qkv(output)
        return self.W_o(output_concat)