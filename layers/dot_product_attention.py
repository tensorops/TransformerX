import tensorflow as tf

from utils import masked_softmax


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
