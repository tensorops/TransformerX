import tensorflow as tf
from einops import rearrange

from transformerx.layers.dot_product_attention import DotProductAttention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head attention."""

    def __init__(
            self,
            d_model,
            num_heads,
            dropout,
            bias=False,
            **kwargs,
    ):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads

        self.attention = DotProductAttention(dropout, num_heads)
        self.W_q = tf.keras.layers.Dense(d_model, use_bias=bias)
        self.W_k = tf.keras.layers.Dense(d_model, use_bias=bias)
        self.W_v = tf.keras.layers.Dense(d_model, use_bias=bias)
        self.W_o = tf.keras.layers.Dense(d_model, use_bias=bias)

    def split_heads(self, X: tf.Tensor) -> tf.Tensor:
        """Transpose tensors for parallel computation of attention heads.

        First transposition produces a tensor of shape X: (batch_size, num_heads, no. of queries or key-value pairs,
        depth / num_heads).
        Next it is rearranged to a new order (batch_size * num_heads, no. of queries or key-value pairs,
        depth / num_heads) which is then passed to the last rearrangement and returned.

        Parameters
        ----------
        X : Shape (batch_size, no. of queries or key-value pairs, depth).

        Returns
        -------
        X : Transposed tensor of shape ((batch_size * num_heads, no. of queries or key-value pairs,
        depth / num_heads)
                    hape of output X: (batch_size, no. of queries or key-value pairs, num_heads, depth / num_heads)
        """

        # X = tf.reshape(X, shape=(X.shape[0], X.shape[1], self.num_heads, -1))
        X = rearrange(X, "b h (heads hidden) -> b h heads hidden", heads=self.num_heads)
        # print("X reshaped: ", X.shape)
        # X = tf.transpose(X, perm=(0, 2, 1, 3))
        X = rearrange(X, "b d1 d2 d3 -> b d2 d1 d3")
        # print("X transposed: ", X.shape)
        # return tf.reshape(X, shape=(-1, X.shape[2], X.shape[3]))
        X = rearrange(X, "b d1 d2 d3 -> (b d1) d2 d3")
        # print("X reshaped2: ", X.shape)
        return X

    def inverse_transpose_qkv(self, X):
        """Reverse the operation of split_heads."""
        X = tf.reshape(X, shape=(-1, self.num_heads, X.shape[1], X.shape[2]))
        X = tf.transpose(X, perm=(0, 2, 1, 3))
        return tf.reshape(X, shape=(X.shape[0], X.shape[1], -1))

    def call(self, queries, values, keys, valid_lens, window_mask=None, **kwargs):
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
