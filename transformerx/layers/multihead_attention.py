import numpy as np
import tensorflow as tf
from einops import rearrange

from transformerx.layers.dot_product_attention import DotProductAttention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Compute Multi-Head [1]_ (masked) self- or cross- attention layer.

    An attention class that runs through an attention mechanism (i.e. most commonly scaled dot-product) several
    times in parallel. The independent attention outputs are then concatenated and linearly transformed into the
    expected dimension.

    This class implements a multi-head attention layer to be used in a Transformer model. The layer
    computes self-attention using dot-product attention, and applies linear transformations
    to the queries, keys, and values to project them into different subspaces. The resulting
    attention scores are then combined and transformed to produce the final output.

    The multi-head attention layer allows the model to attend to different positions and contexts
    in the input sequence, and to combine and weight these contexts in different ways to produce
    the final output. This can help the model to better capture long-range dependencies and
    complex patterns in the input data.

    The multi-head attention layer is composed of several attention heads, each of which
    computes an attention score for a subset of the queries, keys, and values. The attention
    scores are then combined and weighted to produce the final output for each attention head,
    and the outputs of the attention heads are concatenated and transformed to produce the
    final output of the layer.

    The layer can optionally use a window mask to prevent attention between elements that are
    too far apart in the input sequence. This can help the model to focus on local contexts and
    avoid attending to irrelevant positions in the input.

    See Also
    --------
    layers.dot_product_attention : The class for the dot-product attention mechanism.
    call : The method for computing the multi-head attention.

    Notes
    -----
    Intuitively, multiple attention heads allows for attending to parts of the sequence differently
    (e.g. longer-term dependencies versus shorter-term dependencies).

    For more please see [2]

    Mathematical equations
    ---------------------

    The multi-head attention mechanism computes the attention scores between the queries and key-value pairs using the dot product of their representations. The attention scores are then combined and transformed to produce the final output:

    .. math::
        Attention(Q, K, V) = softmax(\\frac{QK^T}{\\sqrt{d_k}})V

    Where:

    - :math:`Q` is the queries tensor, with shape (batch_size, no. of queries, depth)
    - :math:`K` is the keys tensor, with shape (batch_size, no. of key-value pairs, depth)
    - :math:`V` is the values tensor, with shape (batch_size, no. of key-value pairs, depth)
    - :math:`d_k` is the depth of the queries and keys
    - :math:`softmax` is the softmax function

    The final output of the multi-head attention is computed as a weighted sum of the transformed queries, keys, and values, with the attention scores as the weights:

    .. math::
        Output = Concat(head_1, \\dots, head_n)W^O

    Where:

    - :math:`head_i` is the output of the $i$-th attention head, with shape (batch_size, no. of queries, depth / num_heads)
    .. math::
        head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)
    - :math:`Concat` is the concatenation operation
    - :math:`W^O` is the linear transformation matrix, with shape (num_heads * depth, depth)

    The attention weights tensor is optional and can be used to visualize and analyze the attention mechanisms in the model. The attention weights tensor has shape (batch_size, no. of queries, no. of key-value pairs).

    Parameters
    ----------
    d_model : int
        The dimension of the model, i.e., the depth of the input and output tensors.
    num_heads : int
        Number of the heads in the multi-head attention
    dropout_rate : float
        The dropout rate to use for regularization. Float between 0 and 1.
    bias : bool, optional
        Whether to use bias terms in the linear transformations i.e. W_q, W_k, W_v, and W_o, by default False.

    Returns
    -------
    output:
        Concatenated tensors

    Methods
    -------
    split_heads(X)
        Transpose tensors for parallel computation of attention heads.
    inverse_transpose_qkv(X)
        Reverse the operation of split_heads.
    call(queries, keys, values, valid_lens, window_mask=None, **kwargs)
        Compute the multi-head attention for the given queries, keys, and values.

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

    >>> attention = MultiHeadAttention(d_model=16, num_heads=4, dropout=0.1)
    >>> queries = tf.random.normal((3, 10, 16))
    >>> keys = tf.random.normal((3, 20, 16))
    >>> values = tf.random.normal((3, 20, 16))
    >>> valid_lens = tf.constant([10, 15, 20])
    >>> output, _ = attention(queries, keys, values, valid_lens)
    >>> output.shape
    (3, 10, 16)

    >>> window_mask = tf.ones((3, 10, 20))
    >>> output, _ = attention(queries, keys, values, valid_lens, window_mask=window_mask)
    >>> output.shape
    (3, 10, 16)


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
            attention: str = "scaled_dotproduct",
            **kwargs,
    ):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.bias = bias
        if attention == "scaled_dotproduct" or attention == None:
            self.attention = DotProductAttention(self.dropout_rate, self.num_heads, scaled=True)
        elif attention == "dotproduct":
            self.attention = DotProductAttention(self.dropout_rate, self.num_heads, scaled=False)
        self.W_q = tf.keras.layers.Dense(self.d_model, use_bias=self.bias)
        self.W_k = tf.keras.layers.Dense(self.d_model, use_bias=self.bias)
        self.W_v = tf.keras.layers.Dense(self.d_model, use_bias=self.bias)
        self.W_o = tf.keras.layers.Dense(self.d_model, use_bias=self.bias)

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
        X = rearrange(X, "b l (h dk) -> b l h dk", h=self.num_heads)
        # x = tf.transpose(x, perm=(0, 2, 1, 3))
        X = rearrange(X, "b l h dk -> b h l dk")
        # return tf.reshape(x, shape=(-1, x.shape[2], x.shape[3]))
        # X = rearrange(X, "b h l dk -> (b h) l dk")
        return X

    def inverse_transpose_qkv(self, X):
        """Reverse the operation of split_heads."""


    def call(self,
             queries: tf.Tensor,
             values: tf.Tensor,
             keys: tf.Tensor,
             attention_mask: tf.Tensor = None,
             causal_mask: bool = None,
             **kwargs) -> tf.Tensor:
        """Compute the multi-head attention for the given queries, keys, and values.

            This method computes the multi-head attention for the given queries, keys, and values,
            using the dot-product attention mechanism and the linear transformations defined
            in the constructor. The attention scores are then combined and transformed to produce
            the final output of the layer.

            The method optionally accepts a window mask, which is used to prevent attention
            between elements that are too far apart in the input sequence. This can help the model
            to focus on local contexts and avoid attending to irrelevant positions in the input.

            The method returns the final output tensor and an optional tensor containing the
            attention weights. The attention weights can be used for visualization and analysis
            of the attention mechanisms in the model.

        Parameters
        ----------
        queries : tf.Tensor
            The queries tensor. This tensor has shape (batch_size, no. of queries, depth).
        keys : tf.Tensor
            The keys tensor. This tensor has shape (batch_size, no. of key-value pairs, depth).
        values : tf.Tensor
            The values tensor. This tensor has shape (batch_size, no. of key-value pairs, depth).
        valid_lens : Union[tf.Tensor, tf.Tensor]
            The valid sequence lengths for the queries and keys. This tensor has shape
            (batch_size,) or (batch_size, no. of queries).
        window_mask : Optional[tf.Tensor], optional
            The window mask tensor, by default None. This tensor has shape
            (batch_size, no. of queries, no. of key-value pairs) and contains zeros
            for positions that should not attend to each other.

        Returns
        -------
        Tuple[tf.Tensor, Optional[tf.Tensor]]
            The final output tensor and the attention weights tensor. The output tensor has
            shape (batch_size, no. of queries, depth), and the attention weights tensor has
            shape (batch_size, no. of queries, no. of key-value pairs).

        Raises
        ------
        ValueError
            If the dimensions of the queries, keys, and values tensors are incompatible.

        Examples
        --------
        >>> import tensorflow as tf

        >>> queries = tf.random.normal([batch_size, no_of_queries, depth])
        >>> keys = tf.random.normal([batch_size, no_of_key_value_pairs, depth])
        >>> values = tf.random.normal([batch_size, no_of_key_value_pairs, depth])
        >>> valid_lens = tf.random.uniform([batch_size], minval=0, maxval=no_of_queries, dtype=tf.int32)

        >>> multihead_attn = MultiHeadAttention(d_model=depth, num_heads=num_heads, dropout=dropout)
        >>> output, attention_weights = multihead_attn(queries, keys, values, valid_lens)

        Here is an example of how to use the call method with a window mask:

        >>> import tensorflow as tf

        >>> queries = tf.random.normal([batch_size, no_of_queries, depth])
        >>> keys = tf.random.normal([batch_size, no_of_key_value_pairs, depth])
        >>> values = tf.random.normal([batch_size, no_of_key_value_pairs, depth])
        >>> valid_lens = tf.random.uniform([batch_size], minval=0, maxval=no_of_queries, dtype=tf.int32)
        >>> window_mask = tf.random.uniform([batch_size, no_of_queries, no_of_key_value_pairs], 0, 2, dtype=tf.int32)

        >>> multihead_attn = MultiHeadAttention(d_model=depth, num_heads=num_heads, dropout=dropout)
        >>> output, attention_weights = multihead_attn(queries, keys, values, valid_lens, window_mask)
        """

        # Shape of queries, keys, or values:
        # (batch_size, no. of queries or key-value pairs, depth)
        # Shape of attention_mask: (batch_size,) or (batch_size, no. of queries)
        # After transposing, shape of output queries, keys, or values:
        # (batch_size * num_heads, no. of queries or key-value pairs,
        # depth / num_heads)

        queries = self.split_heads(self.W_q(queries))
        keys = self.split_heads(self.W_k(keys))
        values = self.split_heads(self.W_v(values))

        if attention_mask is not None:
            # On axis 0, copy the first item (scalar or vector) for num_heads
            # times, then copy the next item, and so on
            attention_mask = tf.repeat(attention_mask, repeats=self.num_heads, axis=0)

        # Shape of output: (batch_size * num_heads, no. of queries,
        # depth / num_heads)
        output = self.attention(
            queries, keys, values, attention_mask, causal_mask, **kwargs
        )

        # Shape of output_concat: (batch_size, no. of queries, depth)
        output_concat = self.inverse_transpose_qkv(output)
        return self.W_o(output_concat)
