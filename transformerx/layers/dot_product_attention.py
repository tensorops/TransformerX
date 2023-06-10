import tensorflow as tf

from transformerx.layers.masks.global_attention_mask import GlobalAttentionMask
from transformerx.utils import masked_softmax
from transformerx.layers.masks import LookAheadMask


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

    .. math::
        Attention(Q, K, V) = softmax(Q K^T) V

    And scaled dot-product attention [1]_ is formulated as:

    .. math::
        Attention(Q, K, V) = softmax(\\frac{QK^T}{\\sqrt{d_k}}) V


    Examples
    --------
    Scaled dot-product (scaled multiplicative) self-attention of tensor `x` (we feed `x` to queries, keys, and
    values).
    >>> tf.random.set_seed(1)
    >>> x = tf.cast(tf.random.uniform([2, 3, 2]), dtype=tf.float32)
    >>> print(x)
    tf.Tensor(
    [[[0.16513085 0.9014813 ]
      [0.6309742  0.4345461 ]
      [0.29193902 0.64250207]]
    <BLANKLINE>
     [[0.9757855  0.43509948]
      [0.6601019  0.60489583]
      [0.6366315  0.6144488 ]]], shape=(2, 3, 2), dtype=float32)

    >>> dot_product = DotProductAttention(0.2)
    >>> queries, keys, values = x, x, x
    >>> output, attn_weights = dot_product(queries, keys, values)
    >>> print(output)
    tf.Tensor(
    [[[0.34450796 0.6787753 ]
      [0.36907017 0.65472305]
      [0.35440704 0.66882825]]
    <BLANKLINE>
     [[0.77042043 0.5446019 ]
      [0.7632908  0.5484005 ]
      [0.7627964  0.5486638 ]]], shape=(2, 3, 2), dtype=float32)

    The next example shows the dot-product (multiplicative) self-attention of tensor `x`.

    >>> dot_product = DotProductAttention(dropout_rate=0.1, scaled=False)
    >>> output, attn_weights = dot_product(queries, keys, values)
    >>> print(output)
    tf.Tensor(
    [[[0.33704066 0.6868143 ]
      [0.37176722 0.6526886 ]
      [0.35094902 0.6727435 ]]
    <BLANKLINE>
     [[0.7759446  0.54165894]
      [0.7657266  0.54710305]
      [0.7650213  0.5474789 ]]], shape=(2, 3, 2), dtype=float32)

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
        causal_mask: bool = None,
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
        self.causal_mask = causal_mask

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

    # Shape of queries: (batch_size, num_heads, seq_len, head_size) or (batch_size, q_seq_len, d_model)
    # Shape of keys: (batch_size, num_heads, seq_len, head_size) or (batch_size, k_seq_len, d_model)
    # Shape of values: (batch_size, num_heads, seq_len, head_size) or (batch_size, v_seq_len, d_model)
    # Shape of attention_mask: (batch_size,) or (batch_size, no. of queries)
    def call(
        self,
        queries: tf.Tensor,
        keys: tf.Tensor,
        values: tf.Tensor,
        attention_mask: tf.Tensor = None,
        training=None,
        **kwargs,
    ) -> tf.Tensor:
        scores = tf.matmul(queries, keys, transpose_b=True)
        if self.scaled:
            d_model = queries.shape[-1]

            scores = scores / tf.math.sqrt(tf.cast(d_model, dtype=queries.dtype))

        # apply causal mask
        if self.causal_mask:
            # Obsolete version of masking. To be removed in the upcomming updates
            # seq_len = tf.shape(queries)[2]
            # heads = tf.shape(queries)[1]
            # batch_size, num_heads, seq_len, _ = tf.unstack(tf.shape(queries))
            # causal_mask = tf.ones((num_heads, seq_len)) * -1e9
            # causal_mask = tf.linalg.LinearOperatorLowerTriangular(
            #     causal_mask
            # ).to_dense()
            # causal_mask = tf.expand_dims(causal_mask, axis=0)  # add batch dimension
            # causal_mask = tf.broadcast_to(
            #     tf.expand_dims(causal_mask, -1), tf.shape(scores)
            # )  # broadcast across batch dimension

            # New version of masking
            look_ahead_mask = LookAheadMask()
            scores = look_ahead_mask(scores)

        # to be uncommented later
        # apply global mask
        # gmask = self.global_mask.get_mask(keys.shape)
        # masked_attention_scores = tf.math.multiply(scores, gmask)
        # attention_probs = tf.nn.softmax(masked_attention_scores, axis=-1)
        # uncomment until here

        self.attention_weights = masked_softmax(scores, attention_mask)
        # self.attention_weights = tf.nn.softmax(scores, axis=-1, mask=attention_mask)
        # scores = tf.matmul(self.dropout(self.attention_weights, **kwargs), values)
        attention_output = tf.matmul(self.dropout(self.attention_weights), values)
        return attention_output, self.attention_weights

    def get_attention_weights(self):
        return self.attention_weights
