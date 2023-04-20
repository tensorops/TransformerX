import numpy as np
import tensorflow as tf

from transformerx.layers.masks.global_attention_mask import GlobalAttentionMask


class DotProductAttention(tf.keras.layers.Layer):
    """Compute (scaled) dot-product attention [1]_

    Implement multiplicative (dot-product) and scaled multiplicative attention for the input queries, keyes, and values.

    Parameters
    ----------
    dropout_rate : float
        Fraction of the input units to drop. A float between 0 and 1.
    scaled : bool
        Indicate whether to scale the dot-product

    Returns
    -------
    output : tf.Tensor with the same shape of Query, Key, and value
        (Scaled) dot-product of the keys, queries, and values

    Notes
    -----
    Dot-product attention formulation is as following:
    .. math:: Attention(Q, K, V) = softmax(Q K^T) V

    And scaled dot-product attention [1]_ is formulated as:

    ..math:: Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V


    Examples
    --------
    Scaled dot-product (scaled multiplicative) self-attention of tensor `x` (we feed `x` to queries, keys, and
    values).

    >>> x = tf.cast(np.random.random([2, 3, 2]), dtype=tf.float32)
    >>> print(x)
    tf.Tensor(
    [[[0.5418388  0.23626359]
      [0.4220487  0.394948  ]
      [0.6125364  0.12296485]]

     [[0.17872103 0.5700011 ]
      [0.28264287 0.02290592]
      [0.24536102 0.39220297]]], shape=(2, 3, 2), dtype=float32)  #random

    >>> dot_product = DotProductAttention(0.2)
    >>> queries, keys, values = x, x, x
    >>> output = dot_product(queries, keys, values)
    >>> print(output)
    tf.Tensor(
    [[[0.45955482 0.63378114]
      [0.48054144 0.62751293]
      [0.43684354 0.64026886]]

     [[0.82063836 0.2958246 ]
      [0.8300792  0.30486548]
      [0.83300924 0.30762452]]], shape=(2, 3, 2), dtype=float32)

    The next example shows the dot-product (multiplicative) self-attention of tensor `x`.

    >>> dot_product = DotProductAttention(dropout_rate=0.1, scaled=False)
    >>> output = dot_product(queries, keys, values)
    >>> print(output)
    tf.Tensor(
    [[[0.5195807  0.6383675 ]
      [0.49765232 0.6440835 ]
      [0.5132934  0.64001364]]

     [[0.6074392  0.80120546]
      [0.6098373  0.80074203]
      [0.5967663  0.7891044 ]]], shape=(2, 3, 2), dtype=float32)

    References
    ----------
    .. [1] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, I. Polosukhin, Attention
    is all you need, in: NIPS, pp. 5998–6008.
    """

    def __init__(
        self,
        dropout_rate: float = 0,
        scaled: bool = True,
        kernel_initializer: str = "ones",
        kernel_regularizer: str = None,
        mask_type="dilated",
        mask_prob=0.0,
        dilation_rate=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.scaled = scaled
        self.attention_weights = None
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

        self.mask_type = mask_type
        self.mask_prob = mask_prob
        self.dilation_rate = dilation_rate
        self.global_mask = GlobalAttentionMask(
            mask_type=self.mask_type,
            mask_prob=self.mask_prob,
            dilation_rate=self.dilation_rate,
        )

    def build(self, input_shape):
        super().build(input_shape)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of attention_mask: (batch_size,) or (batch_size, no. of queries)
    def call(
        self,
        queries: tf.Tensor,
        keys: tf.Tensor,
        values: tf.Tensor,
        attention_mask: tf.Tensor = None,
        causal_mask: bool = None,
        training=None,
        **kwargs,
    ) -> tf.Tensor:
        scores = tf.matmul(queries, keys, transpose_b=True)
        if self.scaled:
            depth = queries.shape[-1]

            scores = scores / tf.math.sqrt(tf.cast(depth, dtype=tf.float32))

        # apply causal mask
        if causal_mask:
            seq_len = tf.shape(queries)[2]
            heads = tf.shape(queries)[1]
            causal_mask = tf.ones((heads, seq_len)) * -1e9
            causal_mask = tf.linalg.LinearOperatorLowerTriangular(
                causal_mask
            ).to_dense()
            causal_mask = tf.expand_dims(causal_mask, axis=0)  # add batch dimension
            scores += tf.broadcast_to(
                tf.expand_dims(causal_mask, -1), scores.shape
            )  # broadcast across batch dimension

        gmask = self.global_mask.get_mask(keys.shape)
        masked_attention_scores = tf.math.multiply(scores, gmask)
        attention_probs = tf.nn.softmax(masked_attention_scores, axis=-1)
        # self.attention_weights = masked_softmax(scores, attention_mask)
        # self.attention_weights = tf.nn.softmax(scores, axis=-1, mask=attention_mask)
        # scores = tf.matmul(self.dropout(self.attention_weights, **kwargs), values)
        scores = tf.matmul(self.dropout(attention_probs, **kwargs), values)

        return scores

    def get_attention_weights(self):
        return self.attention_weights
