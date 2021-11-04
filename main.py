import os
import time
import typing
from typing import Tuple
from itertools import cycle

import numpy as np
import tensorflow as tf
from einops import rearrange, reduce
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""

    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.shape[1]
        mask = tf.range(start=0, limit=maxlen, dtype=tf.float32)[None, :] < tf.cast(
            valid_len[:, None], dtype=tf.float32
        )

        if len(X.shape) == 3:
            return tf.where(tf.expand_dims(mask, axis=-1), X, value)
        else:
            return tf.where(mask, X, value)

    if valid_lens is None:
        return tf.nn.softmax(X, axis=-1)
    else:
        shape = X.shape
        if len(valid_lens.shape) == 1:
            valid_lens = tf.repeat(valid_lens, repeats=shape[1])

        else:
            valid_lens = tf.reshape(valid_lens, shape=-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(tf.reshape(X, shape=(-1, shape[-1])), valid_lens, value=-1e6)
        return tf.nn.softmax(tf.reshape(X, shape=shape), axis=-1)


class DotProductAttention(tf.keras.layers.Layer):
    """Scaled dot product attention."""

    def __init__(self, dropout, num_heads=8):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.num_heads = num_heads  # To be covered later

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def call(self, queries, keys, values, valid_lens=None, window_mask=None, **kwargs):
        d = queries.shape[-1]
        scores = tf.matmul(queries, keys, transpose_b=True) / tf.math.sqrt(
            tf.cast(d, dtype=tf.float32)
        )
        if window_mask is not None:  # To be covered later
            num_windows = window_mask.shape[0]
            n, num_queries, num_kv_pairs = scores.shape
            # Shape of window_mask: (num_windows, no. of queries,
            # no. of key-value pairs)
            scores = tf.reshape(
                scores,
                (
                    n // (num_windows * self.num_heads),
                    num_windows,
                    self.num_heads,
                    num_queries,
                    num_kv_pairs,
                ),
            ) + tf.expand_dims(tf.expand_dims(window_mask, 1), 0)
            scores = tf.reshape(scores, (n, num_queries, num_kv_pairs))
        self.attention_weights = masked_softmax(scores, valid_lens)
        return tf.matmul(self.dropout(self.attention_weights, **kwargs), values)


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head attention."""

    def __init__(
        self,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        num_heads,
        dropout,
        bias=False,
        **kwargs,
    ):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads

        self.attention = DotProductAttention(dropout, num_heads)
        self.W_q = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_k = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_v = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_o = tf.keras.layers.Dense(num_hiddens, use_bias=bias)

    def transpose_qkv(self, X: tf.Tensor) -> tf.Tensor:
        """Transpose tensors for parallel computation of attention heads.

        First transposition produces a tensor of shape X: (batch_size, num_heads, no. of queries or key-value pairs,
        num_hiddens / num_heads).
        Next it is rearranged to a new order (batch_size * num_heads, no. of queries or key-value pairs,
        num_hiddens / num_heads) which is then passed to the last rearrangement and returned.

        Parameters
        ----------
        X : Shape (batch_size, no. of queries or key-value pairs, num_hiddens).

        Returns
        -------
        X : Transposed tensor of shape ((batch_size * num_heads, no. of queries or key-value pairs,
        num_hiddens / num_heads)
                    hape of output X: (batch_size, no. of queries or key-value pairs, num_heads, num_hiddens / num_heads)
        """

        # X = tf.reshape(X, shape=(X.shape[0], X.shape[1], self.num_heads, -1))
        X = rearrange(X, "n h (heads hidden) -> n h heads hidden", heads=self.num_heads)
        print("X reshaped: ", X.shape)
        # X = tf.transpose(X, perm=(0, 2, 1, 3))
        X = rearrange(X, "b d1 d2 d3 -> b d2 d1 d3")
        print("X transposed: ", X.shape)
        # return tf.reshape(X, shape=(-1, X.shape[2], X.shape[3]))
        X = rearrange(X, "b d1 d2 d3 -> (b d1) d2 d3")
        print("X reshaped2: ", X.shape)
        return X

    def inverse_transpose_qkv(self, X):
        """Reverse the operation of transpose_qkv."""
        X = tf.reshape(X, shape=(-1, self.num_heads, X.shape[1], X.shape[2]))
        X = tf.transpose(X, perm=(0, 2, 1, 3))
        return tf.reshape(X, shape=(X.shape[0], X.shape[1], -1))

    def call(self, queries, values, keys, valid_lens, window_mask=None, **kwargs):
        # Shape of queries, keys, or values:
        # (batch_size, no. of queries or key-value pairs, num_hiddens)
        # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
        # After transposing, shape of output queries, keys, or values:
        # (batch_size * num_heads, no. of queries or key-value pairs,
        # num_hiddens / num_heads)

        print("wq(queries): ", self.W_q(queries).shape)
        print("queries: ", queries.shape)
        queries = self.transpose_qkv(self.W_q(queries))
        print("keys: ", keys.shape)
        keys = self.transpose_qkv(self.W_k(keys))
        print("values: ", values.shape)
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for num_heads
            # times, then copy the next item, and so on
            valid_lens = tf.repeat(valid_lens, repeats=self.num_heads, axis=0)

        # Shape of output: (batch_size * num_heads, no. of queries,
        # num_hiddens / num_heads)
        output = self.attention(
            queries, keys, values, valid_lens, window_mask, **kwargs
        )

        # Shape of output_concat: (batch_size, no. of queries, num_hiddens)
        output_concat = self.inverse_transpose_qkv(output)
        return self.W_o(output_concat)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        # Create a long enough P

        self.P = np.zeros((2, max_len, num_hiddens))
        print("P.shape", self.P.shape)
        X = np.arange(max_len, dtype=np.float32).reshape(-1, 1) / np.power(
            10000, np.arange(0, num_hiddens, 2, dtype=np.float32) / num_hiddens
        )

        self.P[:, :, 0::2] = tf.sin(
            X
        )  # x[low::stride] -> positions: 0, 2, 4, ... of all rows and columns
        self.P[:, :, 1::2] = tf.cos(
            X
        )  # x[low::stride] -> positions: 1, 3, 5 , ... of all rows and columns

    def call(self, X, **kwargs):
        X = X + self.P[:, : X.shape[1], :]
        return self.dropout(X, **kwargs)


class Plot:
    def plot_pe(
        self,
        cols: Tuple[int, list, np.ndarray],
        pos_encodings,
        num_steps,
        show_grid=True,
    ):
        ax = plt.figure(figsize=(6, 2.5))

        lines = ["-", "--", "-.", ":"]
        self.line_cycler = cycle(lines)

        def plot_line(col):
            plt.plot(
                np.arange(num_steps),
                pos_encodings[0, :, col].T,
                next(self.line_cycler),
                label=f"col {col}",
            )

        if isinstance(cols, (list, np.ndarray)):
            for col in cols:
                plot_line(col)
        else:
            plot_line(cols)
        ax.legend()
        plt.title("Columns 7-10")
        plt.grid(show_grid)
        plt.show()


class PositionWiseFFN(tf.keras.layers.Layer):
    """Position-wise feed-forward network."""

    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(ffn_num_hiddens)
        self.relu = tf.keras.layers.ReLU()
        self.dense2 = tf.keras.layers.Dense(ffn_num_outputs)

    def call(self, X):
        # x.shape: (batch size, number of time steps or sequence length in tokens, number of hidden units or feature dimension)
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(tf.keras.layers.Layer):
    """Add a residual connection followed by a layer normalization"""

    def __init__(self, norm_shape, dropout):
        super(AddNorm, self).__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.ln = tf.keras.layers.LayerNormalization(norm_shape)

    def call(self, X, Y, **kwargs):
        return self.ln(self.dropout(Y, **kwargs) + X)


class TransformerEncoderBlock(tf.keras.layers.Layer):
    """Transformer encoder block."""

    def __init__(
        self,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_hiddens,
        num_heads,
        dropout,
        bias=False,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias
        )
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def call(self, X, valid_lens, **kwargs):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens, **kwargs), **kwargs)
        return self.addnorm2(Y, self.ffn(Y), **kwargs)


class TransformerEncoder(tf.keras.layers.Layer):
    """Transformer encoder that encompasses one or more TransformerEncoderBlock blocks."""

    def __init__(
        self,
        vocab_size,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_hiddens,
        num_heads,
        num_blks,
        dropout,
        bias=False,
    ):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = tf.keras.layers.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = [
            TransformerEncoderBlock(
                key_size,
                query_size,
                value_size,
                num_hiddens,
                norm_shape,
                ffn_num_hiddens,
                num_heads,
                dropout,
                bias,
            )
            for _ in range(num_blks)
        ]

    def call(self, X, valid_lens, **kwargs):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        X = self.pos_encoding(
            self.embedding(X)
            * tf.math.sqrt(tf.cast(self.num_hiddens, dtype=tf.float32)),
            **kwargs,
        )
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens, **kwargs)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X


class TransformerDecoderBlock(tf.keras.layers.Layer):
    """Transformer decoder block."""

    def __init__(
        self,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_hiddens,
        num_heads,
        dropout,
        i,
    ):
        super().__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout
        )
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout
        )
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def call(self, X, state, **kwargs):
        """Forward propagation of the decoder block.

        During training, all the tokens of any output sequence are processed at the same time, so state[2][self.i]
        is None as initialized. When decoding any output sequence token by token during prediction, state[2][self.i]
        contains representations of the decoded output at the i-th block up to the current time step"""
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = tf.concat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if kwargs["training"]:
            batch_size, num_steps, _ = X.shape
            # Shape of dec_valid_lens: (batch_size, num_steps), where every
            # row is [1, 2, ..., num_steps]
            dec_valid_lens = tf.repeat(
                tf.reshape(tf.range(1, num_steps + 1), shape=(-1, num_steps)),
                repeats=batch_size,
                axis=0,
            )
        else:
            dec_valid_lens = None
        # Self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens, **kwargs)
        Y = self.addnorm1(X, X2, **kwargs)
        # Encoder-decoder attention. Shape of enc_outputs:
        # (batch_size, num_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens, **kwargs)
        Z = self.addnorm2(Y, Y2, **kwargs)
        return self.addnorm3(Z, self.ffn(Z), **kwargs), state
